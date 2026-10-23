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

# The Chatterbox weights are NOT downloaded at build time: RunPod's
# builder failed that step (exit code 1, 2026-10-23), and it was only
# ever an optimisation. The worker downloads the weights on its first
# start instead (handler.py loads the model at import) and keeps them
# in the container for its whole life; FlashBoot then makes later
# starts fast. The build stays small and reliable: apt + pip only.

COPY handler.py .

CMD ["python", "-u", "handler.py"]
