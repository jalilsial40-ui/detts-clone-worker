# Digital Empire TTS — RunPod clone worker image.
# Base: RunPod's own PyTorch image (CUDA + Python 3.11), the standard
# starting point for RunPod serverless workers.
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Same Xet workaround the Modal clone image needs for Hugging Face pulls.
ENV HF_HUB_DISABLE_XET=1

# Pre-download the Chatterbox weights into the image (the same trick the
# Modal image uses) so a cold worker starts reading instead of first
# downloading several GB of model.
RUN python -c "from chatterbox.mtl_tts import ChatterboxMultilingualTTS; ChatterboxMultilingualTTS.from_pretrained(device='cpu')"

COPY handler.py .

CMD ["python", "-u", "handler.py"]
