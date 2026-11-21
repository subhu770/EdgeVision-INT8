# syntax=docker/dockerfile:1.4
# Multi-stage Edge-Optimized Minimal Container
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir .

# Runtime Stage
FROM python:3.11-slim AS runtime

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SIMULATION_MODE=true \
    MODEL_PATH=models/defect_detector_int8.onnx \
    PORT=8000

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY src/ ./src/
COPY benchmarks/ ./benchmarks/
COPY pyproject.toml .

# Create models directory and pre-generate default quantized models
RUN mkdir -p models && \
    python -m src.models.builder --output models/defect_detector_fp32.onnx && \
    python -m src.models.quantizer --input models/defect_detector_fp32.onnx --output models/defect_detector_int8.onnx

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

ENTRYPOINT ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
