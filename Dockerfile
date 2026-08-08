FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/data/hf \
    HF_HUB_CACHE=/data/hf/hub \
    TTS_OUTPUT_DIR=/app/output \
    TTS_DEVICE=auto \
    TTS_MODEL=Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Torch CPU wheels work on both amd64 and arm64 (Apple Silicon Docker).
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir \
      torch torchaudio \
      --index-url https://download.pytorch.org/whl/cpu

COPY app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app /app

RUN mkdir -p /data/hf /app/output \
    && useradd -m -u 1000 tts \
    && chown -R tts:tts /app /data

USER tts

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=5 \
  CMD curl -fsS http://127.0.0.1:8000/api/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
