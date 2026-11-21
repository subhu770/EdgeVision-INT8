"""API module for EdgeVision-INT8 inference server."""

from src.api.schemas import (
    DetectionBox,
    DriftAlert,
    DriftMetrics,
    HealthResponse,
    InferenceResult,
    TelemetryResponse,
)

__all__ = [
    "DetectionBox",
    "DriftAlert",
    "DriftMetrics",
    "HealthResponse",
    "InferenceResult",
    "TelemetryResponse",
]
