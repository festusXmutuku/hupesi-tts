# syntax=docker/dockerfile:1.7
# Hupesi TTS running as a server (headless). Build from the repository ROOT:
#   docker compose -f packaging/docker/compose.yaml up -d --build          # CPU
#   docker compose -f packaging/docker/compose.yaml -f packaging/docker/compose.gpu.yaml up -d --build   # NVIDIA
# Build context files: packaging/docker/Dockerfile.dockerignore (allowlist).

# ---- 1. React UI -> dist/ ---------------------------------------------------
FROM node:20-bookworm-slim AS ui
WORKDIR /ui
# "prepare": "husky" requires .git, which is not present in the image.
ENV HUSKY=0
COPY app-desktop/hupesi-tts-app/package.json app-desktop/hupesi-tts-app/package-lock.json ./
RUN --mount=type=cache,target=/root/.npm npm ci
COPY app-desktop/hupesi-tts-app/ ./
RUN npm run build

# ---- 2. Python backend ------------------------------------------------------
# torch cu128 bundles its own CUDA libraries (nvidia-* packages on PyPI), so the
# nvidia/cuda base image is not required; the host only needs a driver + NVIDIA
# Container Toolkit.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS app

# GPU=1: add torch cu128 + transformers (~3.5 GB). GPU=0: ONNX on CPU only.
ARG GPU=0

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    # History, settings, API key, pronunciation dictionary: $LOCALAPPDATA/HupesiTTS.
    LOCALAPPDATA=/data

WORKDIR /app

# Install dependencies before source code so code changes don't invalidate the layer.
COPY pyproject.toml uv.lock README.md LICENSE NOTICE ./
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$GPU" = "1" ]; then G="--group gpu-app"; else G=""; fi; \
    uv sync --frozen --no-install-project $G

COPY src ./src
COPY apps ./apps
COPY docs/hupesi-tts.md DIEU-KHOAN.md PRIVACY.md THIRD-PARTY-NOTICES.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$GPU" = "1" ]; then G="--group gpu-app"; else G=""; fi; \
    uv sync --frozen $G

COPY --from=ui /ui/dist ./app-desktop/hupesi-tts-app/dist

# Models (models/hf_cache), cloned voices (models/custom_voices_*.json), and
# add-ons (addons/, gpu_venv/) live next to the app: point them to the volume so
# they persist across container recreation.
RUN ln -s /data/models /app/models && ln -s /data/addons /app/addons

VOLUME /data
EXPOSE 8001

# The first start may need to download models, so allow a long startup period.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10m --retries=3 \
    CMD python -c "import json, os, urllib.request; s = json.load(urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '8001') + '/health', timeout=4)); raise SystemExit(s['status'] != 'ok')"

ENTRYPOINT ["sh", "-c", "mkdir -p /data/models /data/addons && exec \"$@\"", "--"]
CMD ["python", "-m", "apps.web_stream"]