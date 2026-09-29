# syntax=docker/dockerfile:1
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080 \
    MODEL_KEY=kronos-mini \
    WEB_CONCURRENCY=1 \
    PYTHONPATH=/app \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface

WORKDIR /app

# System deps for torch wheels / scientific stack
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      curl \
      ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# CPU torch first (smaller than default CUDA wheels)
RUN pip install --upgrade pip \
 && pip install --index-url https://download.pytorch.org/whl/cpu \
      torch==2.4.1

COPY requirements.txt /app/requirements.txt
COPY webapp/requirements.txt /app/webapp-requirements.txt
RUN pip install -r /app/requirements.txt \
 && pip install -r /app/webapp-requirements.txt \
 && pip install "gunicorn==23.0.0"

COPY model /app/model
COPY webapp /app/webapp

# Preload Kronos-mini so the first web request is not a long HF download
RUN python - <<'PY'
from model import Kronos, KronosTokenizer
print("Warming Kronos-mini…")
KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-2k")
Kronos.from_pretrained("NeoQuasar/Kronos-mini")
print("Warmup done")
PY

WORKDIR /app/webapp
EXPOSE 8080

# Long timeout: first predict / model move to device can be slow on small dynos
CMD gunicorn app:app \
    --bind 0.0.0.0:${PORT} \
    --workers 1 \
    --threads 2 \
    --timeout 180 \
    --graceful-timeout 30 \
    --keep-alive 5
