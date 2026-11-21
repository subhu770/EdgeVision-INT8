"""Production-grade FastAPI application for EdgeVision-INT8 industrial defect detection.

Endpoints:
- POST /detect: High-speed edge defect detection on image upload with drift checking.
- GET  /health: Edge inspection node health and model metadata.
- GET  /telemetry/drift: Real-time image distribution and prediction confidence drift telemetry.
- POST /telemetry/drift/reset: Resets drift monitor rolling baselines and alert history.
"""

from contextlib import asynccontextmanager
import logging
import os
import time
from typing import Optional
import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
import numpy as np

from src.api.schemas import (
    DetectionBox,
    DriftAlert,
    DriftMetrics,
    HealthResponse,
    InferenceResult,
    TelemetryResponse,
)
from src.inference.engine import DefectInferenceEngine
from src.models.builder import build_and_export_model
from src.models.quantizer import quantize_onnx_model
from src.telemetry.drift_monitor import DriftMonitor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("api_main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan manager initializing ONNX engine and drift monitor."""
    app.state.start_time = time.time()
    app.state.total_frames_processed = 0

    model_path = os.getenv("MODEL_PATH", "models/defect_detector_int8.onnx")
    use_int8 = os.getenv("USE_INT8", "true").lower() in ("true", "1", "yes")
    conf_thresh = float(os.getenv("CONF_THRESHOLD", "0.25"))
    iou_thresh = float(os.getenv("IOU_THRESHOLD", "0.45"))
    num_threads = int(os.getenv("NUM_THREADS", "4"))

    # Auto-generate models in simulation mode if absent
    if not os.path.exists(model_path):
        fp32_path = "models/defect_detector_fp32.onnx"
        int8_path = "models/defect_detector_int8.onnx"
        logger.info("Model path '%s' not found. Bootstrapping synthetic defect models...", model_path)
        os.makedirs("models", exist_ok=True)
        if not os.path.exists(fp32_path):
            build_and_export_model(output_path=fp32_path, train_synthetic=True)
        if not os.path.exists(int8_path):
            quantize_onnx_model(input_fp32_path=fp32_path, output_int8_path=int8_path, method="dynamic")

        model_path = int8_path if use_int8 else fp32_path

    # Initialize ONNX inference engine
    app.state.engine = DefectInferenceEngine(
        model_path=model_path,
        conf_threshold=conf_thresh,
        iou_threshold=iou_thresh,
        num_threads=num_threads,
    )

    # Initialize Drift Telemetry monitor
    app.state.drift_monitor = DriftMonitor()

    logger.info("EdgeVision-INT8 service initialized with model: %s", model_path)
    yield
    logger.info("EdgeVision-INT8 service shutting down.")


app = FastAPI(
    title="EdgeVision-INT8 API",
    description="Edge-optimized real-time industrial defect detection pipeline with INT8 quantization & drift monitoring.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["Diagnostics"])
async def health_check():
    """Return health status, model metadata, and edge runtime statistics."""
    engine: DefectInferenceEngine = app.state.engine
    is_int8 = "int8" in engine.model_path.lower()
    uptime = time.time() - app.state.start_time

    return HealthResponse(
        status="healthy",
        model_path=engine.model_path,
        precision="INT8" if is_int8 else "FP32",
        execution_provider="CPUExecutionProvider",
        total_frames_processed=app.state.total_frames_processed,
        uptime_seconds=round(uptime, 2),
    )


@app.post("/detect", response_model=InferenceResult, tags=["Inspection"])
async def detect_defects(file: UploadFile = File(...)):
    """Run real-time defect detection on an uploaded inspection frame.

    Args:
        file: Uploaded image file (JPEG, PNG, BMP, or WebP).

    Returns:
        InferenceResult containing detected defect bounding boxes, latencies, and drift status.
    """
    if not file or not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file uploaded or invalid file payload.",
        )

    # Read binary contents
    contents = await file.read()
    if not contents or len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty image payload received.",
        )

    # Decode image via OpenCV
    np_buffer = np.frombuffer(contents, dtype=np.uint8)
    image = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to decode image. Unsupported or corrupted image format.",
        )

    engine: DefectInferenceEngine = app.state.engine
    drift_monitor: DriftMonitor = app.state.drift_monitor

    # Perform inference
    detections, latencies = engine.predict(image)

    # Evaluate drift telemetry
    drift_alert_dict = drift_monitor.process_frame(image, detections)
    app.state.total_frames_processed += 1

    formatted_boxes = [
        DetectionBox(
            label=det["label"],
            class_id=det["class_id"],
            confidence=det["confidence"],
            bbox=det["bbox"],
        )
        for det in detections
    ]

    drift_alert = None
    if drift_alert_dict is not None:
        drift_alert = DriftAlert(
            alert_type=drift_alert_dict["alert_type"],
            severity=drift_alert_dict["severity"],
            message=drift_alert_dict["message"],
            timestamp=drift_alert_dict["timestamp"],
            metrics=DriftMetrics(**drift_alert_dict["metrics"]),
        )

    return InferenceResult(
        status="success",
        detections=formatted_boxes,
        defect_detected=len(formatted_boxes) > 0,
        defect_count=len(formatted_boxes),
        inference_latency_ms=latencies["inference_ms"],
        preprocessing_latency_ms=latencies["preprocess_ms"],
        postprocessing_latency_ms=latencies["postprocess_ms"],
        total_latency_ms=latencies["total_latency_ms"],
        drift_alert=drift_alert,
    )


@app.get("/telemetry/drift", response_model=TelemetryResponse, tags=["Telemetry"])
async def get_drift_telemetry():
    """Retrieve real-time camera stream distribution and confidence drift metrics."""
    drift_monitor: DriftMonitor = app.state.drift_monitor
    metrics = drift_monitor.get_metrics()
    alerts = drift_monitor.get_recent_alerts(limit=25)

    formatted_alerts = [
        DriftAlert(
            alert_type=a["alert_type"],
            severity=a["severity"],
            message=a["message"],
            timestamp=a["timestamp"],
            metrics=DriftMetrics(**a["metrics"]),
        )
        for a in alerts
    ]

    return TelemetryResponse(
        metrics=DriftMetrics(**metrics),
        recent_alerts=formatted_alerts,
    )


@app.post("/telemetry/drift/reset", tags=["Telemetry"])
async def reset_drift_telemetry():
    """Reset telemetry history, rolling averages, and drift alerts."""
    drift_monitor: DriftMonitor = app.state.drift_monitor
    drift_monitor.reset()
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"status": "success", "message": "Telemetry rolling metrics and alert history reset."},
    )
