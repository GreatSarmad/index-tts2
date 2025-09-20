# syntax=docker/dockerfile:1.7

ARG CUDA_VERSION=11.8.0
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-venv \
        python3-pip \
        git \
        ffmpeg \
        libsndfile1 \
        curl \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /workspace
COPY requirements.txt ./
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir --no-build-isolation -e .

ARG DOWNLOAD_MODELS=true
ENV HF_HOME=/opt/models/hf_cache \
    TRANSFORMERS_CACHE=/opt/models/hf_cache \
    HF_HUB_CACHE=/opt/models/hf_cache \
    MODELSCOPE_CACHE=/opt/models/modelscope_cache
RUN mkdir -p /opt/models && \
    if [ "$DOWNLOAD_MODELS" = "true" ]; then python tools/download_models.py --model-dir /opt/models --skip-modelscope; fi

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        ca-certificates \
        curl && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /workspace /app
COPY --from=builder /opt/models /app/models

RUN groupadd -r app && useradd -m -g app app
RUN mkdir -p /app/output /app/voices && chown -R app:app /app

USER app
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/models/hf_cache \
    HF_HUB_CACHE=/app/models/hf_cache \
    TRANSFORMERS_CACHE=/app/models/hf_cache \
    MODELSCOPE_CACHE=/app/models/modelscope_cache \
    INDEXTTS_MODEL_DIR=/app/models \
    INDEXTTS_CONFIG_PATH=/app/models/config.yaml \
    INDEXTTS_VOICES_DIR=/app/voices \
    INDEXTTS_OUTPUT_DIR=/app/output \
    INDEXTTS_AUTOWARM=false

WORKDIR /app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
