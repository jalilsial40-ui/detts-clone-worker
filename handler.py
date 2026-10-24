"""Digital Empire TTS — RunPod serverless worker for voice cloning.

DRAFT FIX 2026-10-24 (not yet reviewed / not deployed).
Replaces handler.py after the owner's review.

One RunPod job = ONE <=3,000-char group, exactly mirroring the Modal
CloneSynth.read() call it replaces: the script is read in <=250-char
pieces with Chatterbox (exaggeration 0.5, cfg_weight 0.5) and the pieces
are joined into a single WAV. The Modal engine stays the coordinator:
it splits the script into groups, starts one RunPod job per group,
downloads the group WAVs, merges them in order and converts to MP3 —
splitting, merging, charging and cancellation logic do not change.

Job input (JSON):
    sample_url : signed URL the worker downloads the clone sample from
                 (or sample_b64: the sample as base64, for small samples)
    upload_url : signed URL the worker PUTs the finished group WAV to
                 (if omitted, the WAV is returned as wav_b64 instead —
                 only sensible for short groups)
    text       : one group of script text (<= 3,000 chars)
    lang       : "en" or "hi" (chosen by the engine from the script)

Returns {"ok": true, "sample_rate": int, "audio_seconds": float} or
{"error": "..."} (the RunPod SDK marks the job failed on "error").

WHAT CHANGED vs handler.py (and why):
    The old file loaded the whole Chatterbox model at import time,
    BEFORE runpod.serverless.start() ever ran. So at container start
    the worker spent its first seconds/minutes downloading weights
    without ever telling RunPod "I am alive". RunPod answered with
    "unhealthy", killed the container, and restarted it — 4 restarts
    in 39 seconds on 2026-10-24 morning, with empty logs, because the
    process died before anything it printed could be captured.

    This file loads the model LAZILY: start() runs within a second of
    container boot, the worker reports healthy immediately, and the
    first job that arrives triggers the (one-time) model download —
    with progress lines printed so the Logs tab is never empty again.
    If the model load itself is what is broken, it now fails as a
    VISIBLE job error with a traceback instead of a silent crash loop.

    Nothing else changed: split_pieces, exaggeration/cfg_weight,
    upload/wav_b64 logic, charging points on the engine side — all
    identical to the previous file.
"""

import io
import os
import tempfile
import time

import numpy as np
import requests
import soundfile as sf

import runpod
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

# Lazy singleton: None until the first job needs it. Loaded ONCE per
# container lifetime, exactly like the old module-level MODEL — but
# AFTER the worker has announced itself to RunPod as healthy.
_MODEL = None
_MODEL_LOAD_ERROR = None


def get_model():
    """Load the Chatterbox model on first use (cuda), reuse after."""
    global _MODEL, _MODEL_LOAD_ERROR
    if _MODEL is not None:
        return _MODEL
    if _MODEL_LOAD_ERROR is not None:
        # Failed once in this container: fail fast instead of retrying
        # the same broken download on every job.
        raise _MODEL_LOAD_ERROR
    print("[worker] First job: loading ChatterboxMultilingualTTS on cuda ...",
          flush=True)
    t0 = time.time()
    try:
        _MODEL = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
    except Exception as e:  # noqa: BLE001 - surfaced to the job as text
        _MODEL_LOAD_ERROR = e
        print(f"[worker] MODEL LOAD FAILED after {time.time() - t0:.1f}s: "
              f"{type(e).__name__}: {e}", flush=True)
        raise
    print(f"[worker] Model ready in {time.time() - t0:.1f}s "
          f"(sample_rate={getattr(_MODEL, 'sr', '?')})", flush=True)
    return _MODEL


def split_pieces(text: str, limit: int) -> list[str]:
    """Copied verbatim from modal_engine.py so group reading behaves
    exactly as it does on Modal: split on sentence ends (including the
    Urdu/Arabic stops); a single long sentence is cut at a word
    boundary, so no piece ever exceeds the limit."""
    import re

    sentences = [s.strip() for s in re.split(r"(?<=[.!?؟۔])\s+", text) if s.strip()] or [text]
    pieces, cur = [], ""

    def _push(piece: str) -> str:
        while len(piece) > limit:
            cut = piece.rfind(" ", 0, limit)
            if cut < limit // 3:
                cut = limit
            pieces.append(piece[:cut].strip())
            piece = piece[cut:].strip()
        return piece

    for s in sentences:
        if len(cur) + len(s) + 1 <= limit:
            cur = (cur + " " + s).strip()
        else:
            if cur:
                pieces.append(cur)
            cur = _push(s)
    if cur:
        pieces.append(cur)
    return [p for p in pieces if p] or [text]


def handler(job):
    inp = job.get("input") or {}
    text = (inp.get("text") or "").strip()
    if not text:
        return {"error": "No text in job input"}
    lang = inp.get("lang") or "en"
    job_id = job.get("id", "?")
    print(f"[worker] Job {job_id}: {len(text)} chars, lang={lang}", flush=True)

    try:
        model = get_model()
    except Exception as e:  # noqa: BLE001
        # Model itself failed to load: visible job error, container stays
        # alive for the next attempt (and its logs).
        return {"error": f"model_load_failed: {type(e).__name__}: {e}"}

    try:
        with tempfile.TemporaryDirectory() as td:
            sample_path = os.path.join(td, "sample.bin")
            if inp.get("sample_url"):
                r = requests.get(inp["sample_url"], timeout=120)
                r.raise_for_status()
                with open(sample_path, "wb") as f:
                    f.write(r.content)
            elif inp.get("sample_b64"):
                import base64

                with open(sample_path, "wb") as f:
                    f.write(base64.b64decode(inp["sample_b64"]))
            else:
                return {"error": "No sample_url or sample_b64 in job input"}

            parts = []
            for piece in split_pieces(text, 250):
                wav = model.generate(piece, language_id=lang,
                                     audio_prompt_path=sample_path,
                                     exaggeration=0.5, cfg_weight=0.5)
                arr = wav.detach().cpu().numpy() if hasattr(wav, "detach") else np.asarray(wav)
                parts.append(np.asarray(arr, dtype=np.float32).reshape(-1))
            audio = np.concatenate(parts)

            wav_path = os.path.join(td, "group.wav")
            sf.write(wav_path, audio, model.sr, format="WAV")
            result = {"ok": True, "sample_rate": int(model.sr),
                      "audio_seconds": round(len(audio) / float(model.sr), 2)}

            if inp.get("upload_url"):
                with open(wav_path, "rb") as f:
                    up = requests.put(inp["upload_url"], data=f,
                                      headers={"Content-Type": "audio/wav"},
                                      timeout=600)
                up.raise_for_status()
                result["uploaded"] = True
            else:
                import base64

                buf = io.BytesIO()
                sf.write(buf, audio, model.sr, format="WAV")
                result["wav_b64"] = base64.b64encode(buf.getvalue()).decode()
            print(f"[worker] Job {job_id} done: {result['audio_seconds']}s audio",
                  flush=True)
            return result
    except Exception as e:  # noqa: BLE001
        # Any job-time failure (sample download, synthesis, upload):
        # visible error, container stays alive.
        print(f"[worker] Job {job_id} FAILED: {type(e).__name__}: {e}",
              flush=True)
        return {"error": f"{type(e).__name__}: {e}"}


# The worker announces itself to RunPod HERE, within a second of boot —
# the model is NOT loaded yet (see get_model). Started at module level,
# not under a __main__ guard: this file is only ever run as the
# worker's entry script (the image CMD), and RunPod's repository scan
# looks for the start call at the top level.
print("[worker] handler module loaded — starting runpod agent", flush=True)
runpod.serverless.start({"handler": handler})
