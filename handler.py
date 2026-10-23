"""Digital Empire TTS — RunPod serverless worker for voice cloning.

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
"""

import io
import os
import tempfile

import numpy as np
import requests
import soundfile as sf

from chatterbox.mtl_tts import ChatterboxMultilingualTTS

# Loaded once when the worker container starts, then reused by every job
# the container serves — the same pattern as the Modal CloneSynth class.
MODEL = ChatterboxMultilingualTTS.from_pretrained(device="cuda")


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
            wav = MODEL.generate(piece, language_id=lang,
                                 audio_prompt_path=sample_path,
                                 exaggeration=0.5, cfg_weight=0.5)
            arr = wav.detach().cpu().numpy() if hasattr(wav, "detach") else np.asarray(wav)
            parts.append(np.asarray(arr, dtype=np.float32).reshape(-1))
        audio = np.concatenate(parts)

        wav_path = os.path.join(td, "group.wav")
        sf.write(wav_path, audio, MODEL.sr, format="WAV")
        result = {"ok": True, "sample_rate": int(MODEL.sr),
                  "audio_seconds": round(len(audio) / float(MODEL.sr), 2)}

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
            sf.write(buf, audio, MODEL.sr, format="WAV")
            result["wav_b64"] = base64.b64encode(buf.getvalue()).decode()
        return result


if __name__ == "__main__":
    import runpod

    runpod.serverless.start({"handler": handler})
