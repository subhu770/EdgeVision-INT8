# EdgeVision-INT8: Production-Grade Edge Defect Detection Pipeline

[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-INT8_Quantized-orange.svg)](https://onnxruntime.ai/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)

**EdgeVision-INT8** is a real-time, edge-optimized industrial visual defect detection system engineered for high-speed conveyor lines, PCB manufacturing, and sheet metal inspection. It features a lightweight PyTorch CNN exported to ONNX, static and dynamic INT8 quantization, an asynchronous FastAPI service with sub-millisecond pre/post-processing, real-time drift telemetry, and deterministic synthetic simulation requiring zero external dataset downloads.

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    subgraph Simulation ["Simulation & Ingestion"]
        SG["IndustrialDefectGenerator<br/>(Deterministic Synthetic Frames)"]
        RawFrame["Clean / Brushed Metal / PCB<br/>+ Cracks, Scratches, Pits"]
        SG --> RawFrame
    end

    subgraph API ["FastAPI Edge Service"]
        Endpoint["POST /detect"]
        RawFrame --> Endpoint
        Letterbox["Letterbox Resize & Normalization<br/>(256x256x3 CHW, float32)"]
        Endpoint --> Letterbox
    end

    subgraph Engine ["ONNX Runtime Engine (CPUExecutionProvider)"]
        ORT["InferenceSession<br/>(ORT_ENABLE_ALL, Multi-threaded)"]
        Letterbox --> ORT
        Head["Vectorized NMS Post-Processing<br/>(IoU Threshold: 0.45)"]
        ORT --> Head
    end

    subgraph Telemetry ["Drift Telemetry & Alerting"]
        Drift["DriftMonitor<br/>(Luminance, Laplacian Blur, Confidence)"]
        RawFrame --> Drift
        Head --> Drift
        Alert["DriftAlert (WARNING / CRITICAL)<br/>GET /telemetry/drift"]
        Drift --> Alert
    end

    subgraph Quant ["Model Optimization"]
        FP32["LightweightDefectDetector (FP32)"]
        Quantizer["quantize_dynamic / quantize_static (INT8)"]
        FP32 --> Quantizer
        Quantizer --> ORT
    end
```

---

## 🚀 Key Features

1. **Lightweight Edge Backbone**: MobileNet-style depthwise separable convolutional pyramid with grid-based anchor-free detection head (`[B, 256, 8]`).
2. **ONNX Runtime INT8 Post-Training Quantization**:
   - Supports both dynamic and static QDQ calibration (`IndustrialCalibrationDataReader`).
   - ~3x-4x model size reduction with < 0.05 mIoU degradation.
3. **High-Speed Vectorized NMS**: Pure NumPy implementation of Non-Maximum Suppression with coordinate un-padding back to original image aspect ratios.
4. **Edge Drift Telemetry**:
   - **Luminance Drift**: Detects lighting changes, ambient light pollution, or illuminator degradation.
   - **Blur Degradation**: Monitors variance of the Laplacian to catch optical defocus, smudging, or conveyor vibration.
   - **Confidence Degradation**: Tracks rolling prediction confidences to detect out-of-distribution materials.
5. **Deterministic Simulation & Zero External Dependencies**:
   - Built-in `IndustrialDefectGenerator` creates brushed metal, PCB solder-mask surfaces, and realistic cracks, scratches, and pits with ground-truth bounding boxes.
   - Fully testable on standard CPU environments without downloading multi-gigabyte external datasets.

---

## 📁 Directory Structure

```
EdgeVision-INT8/
├── Dockerfile                   # Multi-stage edge container
├── docker-compose.yml           # Edge service orchestration with resource limits
├── Makefile                     # Build, test, and benchmark shortcuts
├── pyproject.toml               # Modern packaging & pytest config
├── README.md                    # System documentation
├── src/
│   ├── api/
│   │   ├── main.py              # FastAPI app (/detect, /health, /telemetry/drift)
│   │   └── schemas.py           # Pydantic models for detections & drift alerts
│   ├── models/
│   │   ├── builder.py           # Model architecture & ONNX FP32 exporter
│   │   └── quantizer.py         # Dynamic/Static INT8 quantizer & parity verifier
│   ├── inference/
│   │   └── engine.py            # High-performance ONNX Runtime inference wrapper
│   ├── telemetry/
│   │   └── drift_monitor.py     # Luminance, blur variance, and confidence monitor
│   └── simulation/
│       └── frame_generator.py   # Deterministic synthetic defect generator
├── tests/
│   ├── test_quantization.py     # INT8 compression and output parity tests
│   ├── test_inference_api.py    # FastAPI end-to-end integration tests
│   └── test_drift_monitor.py    # Drift alert verification under simulated degradation
└── benchmarks/
    └── benchmark_fps.py         # FP32 vs INT8 latency, FPS, and IoU retention benchmark
```

---

## 🛠️ Quickstart & Local Installation

### 1. Setup Environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### 2. Build Models & Quantize
```bash
# Build and export FP32 model
python -m src.models.builder --output models/defect_detector_fp32.onnx

# Quantize to INT8
python -m src.models.quantizer --input models/defect_detector_fp32.onnx --output models/defect_detector_int8.onnx
```

### 3. Run Test Suite
```bash
pytest -v
```

### 4. Run Benchmark (FP32 vs INT8)
```bash
python benchmarks/benchmark_fps.py
```

### 5. Launch FastAPI Service
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/detect` | Uploads an image frame; returns defect bounding boxes, latencies, and drift alert. |
| `GET` | `/health` | Service health status, model path, precision (`INT8`/`FP32`), and uptime. |
| `GET` | `/telemetry/drift` | Snapshot of luminance drift, blur variance, and confidence drop metrics. |
| `POST` | `/telemetry/drift/reset` | Resets drift monitor rolling averages and alert queues. |

### Sample Detection Response:
```json
{
  "status": "success",
  "detections": [
    {
      "label": "crack",
      "class_id": 0,
      "confidence": 0.924,
      "bbox": [42.5, 78.0, 114.2, 162.8]
    }
  ],
  "defect_detected": true,
  "defect_count": 1,
  "inference_latency_ms": 7.82,
  "preprocessing_latency_ms": 1.45,
  "postprocessing_latency_ms": 0.38,
  "total_latency_ms": 9.65,
  "drift_alert": null
}
```

---

## 🐳 Docker Deployment

Build and run the edge container:
```bash
docker-compose up --build
```
The container enforces CPU/memory reservations and includes a built-in healthcheck probing `/health`.
