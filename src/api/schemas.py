"""Pydantic schemas and data transfer objects for EdgeVision-INT8 API."""

from typing import List, Optional
from pydantic import BaseModel, Field


class DetectionBox(BaseModel):
    """Detected defect bounding box and classification."""

    label: str = Field(..., description="Defect class name (crack, scratch, pit)")
    class_id: int = Field(..., description="Integer class identifier")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score")
    bbox: List[float] = Field(..., min_length=4, max_length=4, description="Bounding box [x1, y1, x2, y2] in pixels")


class DriftMetrics(BaseModel):
    """Real-time statistical drift metrics."""

    current_luminance: float = Field(..., description="Instantaneous frame mean luminance")
    smoothed_luminance: float = Field(..., description="Window-smoothed mean luminance")
    baseline_luminance: float = Field(..., description="Baseline reference luminance")
    luminance_drift_ratio: float = Field(..., description="Fractional luminance shift")
    current_blur_variance: float = Field(..., description="Instantaneous Laplacian blur variance")
    smoothed_blur_variance: float = Field(..., description="Window-smoothed blur variance")
    baseline_blur_variance: float = Field(..., description="Baseline focus blur variance")
    blur_drift_ratio: float = Field(..., description="Fractional blur degradation ratio")
    recent_avg_confidence: float = Field(..., description="Rolling average prediction confidence")
    baseline_confidence: float = Field(..., description="Baseline target confidence")
    confidence_drop_ratio: float = Field(..., description="Fractional confidence degradation ratio")
    drift_detected: bool = Field(..., description="True if any drift metric exceeds warning threshold")
    total_frames_inspected: int = Field(..., description="Total cumulative frames processed by telemetry")


class DriftAlert(BaseModel):
    """Structured alert emitted when visual or model drift thresholds are breached."""

    alert_type: str = Field(..., description="LIGHTING_DRIFT, BLUR_DRIFT, or CONFIDENCE_DRIFT")
    severity: str = Field(..., description="WARNING or CRITICAL")
    message: str = Field(..., description="Human-readable alert diagnostic description")
    timestamp: float = Field(..., description="Unix timestamp of alert generation")
    metrics: DriftMetrics = Field(..., description="Drift metrics snapshot at alert time")


class InferenceResult(BaseModel):
    """End-to-end inference and defect detection response."""

    status: str = Field(default="success", description="Status code")
    detections: List[DetectionBox] = Field(default_factory=list, description="List of detected defect objects")
    defect_detected: bool = Field(..., description="True if one or more defects were found")
    defect_count: int = Field(..., description="Total number of defects detected")
    inference_latency_ms: float = Field(..., description="ONNX Runtime CPU compute latency in ms")
    preprocessing_latency_ms: float = Field(..., description="Pre-processing letterbox & normalization latency in ms")
    postprocessing_latency_ms: float = Field(..., description="Post-processing NMS latency in ms")
    total_latency_ms: float = Field(..., description="Total pipeline latency in ms")
    drift_alert: Optional[DriftAlert] = Field(default=None, description="Active drift alert if triggered")


class HealthResponse(BaseModel):
    """Edge node health check and runtime state."""

    status: str = Field(default="healthy", description="Service health state")
    model_path: str = Field(..., description="Path to active ONNX model")
    precision: str = Field(..., description="Model precision (INT8 or FP32)")
    execution_provider: str = Field(default="CPUExecutionProvider", description="ONNX Runtime execution provider")
    total_frames_processed: int = Field(..., description="Total frames processed since service startup")
    uptime_seconds: float = Field(..., description="Uptime duration in seconds")


class TelemetryResponse(BaseModel):
    """Telemetry report containing current drift metrics and historical alerts."""

    metrics: DriftMetrics = Field(..., description="Current drift metrics snapshot")
    recent_alerts: List[DriftAlert] = Field(default_factory=list, description="Recent drift alerts")
