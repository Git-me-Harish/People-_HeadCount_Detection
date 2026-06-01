# syntax=docker/dockerfile:1.6

# ---- Stage 1: build frontend ----
FROM node:20-alpine AS frontend-build
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install --no-audit --no-fund --progress=false
COPY frontend/ ./
RUN npm run build

# ---- Stage 2: backend runtime ----
FROM python:3.11-slim AS runtime
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# OpenCV / Pillow runtime deps
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgtk-3-0 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && pip install -r /tmp/requirements.txt

COPY backend/ /app/backend/
COPY yolov8n.pt /app/yolov8n.pt
COPY --from=frontend-build /web/dist /app/static

# Serve the built SPA from the API container by mounting via reverse proxy in
# compose, OR copy the built assets and let FastAPI serve them (simple mode).
COPY backend/app /app/app

EXPOSE 8000
ENV YOLO_MODEL_PATH=/app/yolov8n.pt \
    STORAGE_DIR=/app/storage \
    UPLOAD_DIR=/app/storage/uploads \
    OUTPUT_DIR=/app/storage/outputs

VOLUME ["/app/storage"]

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
