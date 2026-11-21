"""End-to-end integration tests for EdgeVision-INT8 FastAPI service."""

import io
import cv2
from fastapi.testclient import TestClient
import numpy as np
import pytest

from src.api.main import app
from src.simulation.frame_generator import IndustrialDefectGenerator


@pytest.fixture(scope="module")
def client(model_paths):
    """Test client configured with the test INT8 model."""
    import os
    os.environ["MODEL_PATH"] = model_paths["int8"]

    with TestClient(app) as test_client:
        yield test_client


def _encode_image_to_bytes(img: np.ndarray, format: str = ".png") -> io.BytesIO:
    """Encode OpenCV image array to in-memory bytes buffer."""
    success, encoded = cv2.imencode(format, img)
    assert success, "Failed to encode image to memory"
    return io.BytesIO(encoded.tobytes())


def test_health_check_endpoint(client):
    """Test GET /health returns healthy status and model precision."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["precision"] in ("INT8", "FP32")
    assert data["execution_provider"] == "CPUExecutionProvider"
    assert "total_frames_processed" in data
    assert "uptime_seconds" in data


def test_detect_clean_surface(client, frame_generator):
    """Test POST /detect with pristine metal surface returns zero defects."""
    clean_frame, _ = frame_generator.generate_frame(surface_type="metal", defect_types=[])
    buf = _encode_image_to_bytes(clean_frame)

    response = client.post(
        "/detect",
        files={"file": ("clean_sheet.png", buf, "image/png")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["defect_detected"] is False
    assert data["defect_count"] == 0
    assert len(data["detections"]) == 0
    assert data["total_latency_ms"] > 0.0
    assert data["inference_latency_ms"] > 0.0


def test_detect_defective_surface(client, frame_generator):
    """Test POST /detect with defective frame accurately identifies defect and bounding box."""
    defective_frame, gt_annos = frame_generator.generate_frame(
        surface_type="metal",
        defect_types=["crack", "scratch"],
    )
    buf = _encode_image_to_bytes(defective_frame)

    response = client.post(
        "/detect",
        files={"file": ("defective_part.png", buf, "image/png")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["defect_detected"] is True
    assert data["defect_count"] >= 1
    assert len(data["detections"]) >= 1

    # Validate detection coordinates and confidence
    for det in data["detections"]:
        assert det["label"] in ["crack", "scratch", "pit"]
        assert 0.0 <= det["confidence"] <= 1.0
        x1, y1, x2, y2 = det["bbox"]
        assert 0.0 <= x1 < x2 <= 256.0
        assert 0.0 <= y1 < y2 <= 256.0


def test_detect_invalid_empty_payload(client):
    """Test POST /detect handles empty payload gracefully with HTTP 400."""
    empty_buf = io.BytesIO(b"")
    response = client.post(
        "/detect",
        files={"file": ("empty.png", empty_buf, "image/png")},
    )
    assert response.status_code == 400
    assert "Empty image" in response.json()["detail"]


def test_detect_corrupted_payload(client):
    """Test POST /detect handles non-image bytes gracefully with HTTP 400."""
    corrupted_buf = io.BytesIO(b"Not An Image Byte Stream")
    response = client.post(
        "/detect",
        files={"file": ("corrupt.png", corrupted_buf, "image/png")},
    )
    assert response.status_code == 400
    assert "Failed to decode image" in response.json()["detail"]


def test_telemetry_drift_endpoints(client, frame_generator):
    """Test GET /telemetry/drift and POST /telemetry/drift/reset."""
    # Reset first
    reset_resp = client.post("/telemetry/drift/reset")
    assert reset_resp.status_code == 200

    # Query drift
    drift_resp = client.get("/telemetry/drift")
    assert drift_resp.status_code == 200
    data = drift_resp.json()
    assert "metrics" in data
    assert "recent_alerts" in data
    assert data["metrics"]["total_frames_inspected"] == 0

    # Feed a frame
    frame, _ = frame_generator.generate_frame(surface_type="metal", defect_types=[])
    buf = _encode_image_to_bytes(frame)
    client.post("/detect", files={"file": ("frame.png", buf, "image/png")})

    # Check updated telemetry
    updated_resp = client.get("/telemetry/drift")
    assert updated_resp.status_code == 200
    updated_data = updated_resp.json()
    assert updated_data["metrics"]["total_frames_inspected"] >= 1
