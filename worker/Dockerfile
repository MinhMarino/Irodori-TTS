FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/runpod-volume/huggingface-cache \
    HF_HUB_CACHE=/runpod-volume/huggingface-cache/hub \
    TRANSFORMERS_CACHE=/runpod-volume/huggingface-cache/hub \
    HF_CACHE_ROOT=/runpod-volume/huggingface-cache/hub \
    BAKED_MODEL_ROOT=/models \
    PRELOAD_MODELS=none \
    LOAD_MODELS=both \
    CUSTOM_VOICE_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
    VOICE_DESIGN_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
    RUNPOD_INIT_TIMEOUT=800

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

COPY handler.py /app/handler.py

# Models are NOT baked into the image (keeps pull fast).
# Attach Runpod cached models via endpoint modelReferences instead.

CMD ["python", "-u", "handler.py"]
