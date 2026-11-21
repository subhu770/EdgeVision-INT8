"""Tests for drift monitor: lighting shifts, optical blur, and confidence degradation."""

import numpy as np
import pytest

from src.simulation.frame_generator import IndustrialDefectGenerator
from src.telemetry.drift_monitor import DriftMonitor


@pytest.fixture
def monitor():
    """Construct a clean drift monitor instance."""
    return DriftMonitor(
        baseline_luminance=150.0,
        baseline_blur_var=200.0,
        baseline_confidence=0.85,
        lighting_warn_threshold=0.20,
        lighting_crit_threshold=0.35,
        blur_warn_threshold=0.35,
        blur_crit_threshold=0.55,
        conf_drop_warn_threshold=0.25,
        conf_drop_crit_threshold=0.45,
        window_size=20,
    )


def test_baseline_calibration(monitor, frame_generator):
    """Verify reference calibration properly establishes baseline values."""
    pristine_frames = [
        frame_generator.generate_frame(surface_type="metal", defect_types=[])[0]
        for _ in range(10)
    ]
    calib = monitor.calibrate_baseline(pristine_frames, baseline_confidence=0.90)

    assert monitor.is_calibrated is True
    assert 100.0 <= calib["baseline_luminance"] <= 200.0
    assert calib["baseline_blur_var"] > 50.0
    assert calib["baseline_confidence"] == 0.90


def test_nominal_frame_no_drift(monitor, frame_generator):
    """Verify normal frames within tolerance do not trigger drift alerts."""
    frame, _ = frame_generator.generate_frame(surface_type="metal", defect_types=[])
    alert = monitor.process_frame(frame, detections=[{"confidence": 0.88}])

    assert alert is None
    metrics = monitor.get_metrics()
    assert metrics["drift_detected"] is False
    assert metrics["total_frames_inspected"] == 1


def test_lighting_drift_alert(monitor, frame_generator):
    """Verify simulated lighting shifts trigger LIGHTING_DRIFT alert."""
    frame, _ = frame_generator.generate_frame(surface_type="metal", defect_types=[])

    # Severe underexposure (lighting drop factor 0.35)
    dark_frame = frame_generator.apply_lighting_drift(frame, factor=0.35)
    alert = monitor.process_frame(dark_frame)

    assert alert is not None
    assert alert["alert_type"] == "LIGHTING_DRIFT"
    assert alert["severity"] in ("WARNING", "CRITICAL")
    assert alert["metrics"]["luminance_drift_ratio"] > 0.20


def test_blur_drift_alert(monitor, frame_generator):
    """Verify simulated optical defocus / vibration triggers BLUR_DRIFT alert."""
    frame, _ = frame_generator.generate_frame(surface_type="metal", defect_types=[])

    # Severe Gaussian blur to collapse Laplacian variance
    blurred_frame = frame_generator.apply_blur_drift(frame, kernel_size=25, blur_type="gaussian")
    alert = monitor.process_frame(blurred_frame)

    assert alert is not None
    assert alert["alert_type"] == "BLUR_DRIFT"
    assert alert["severity"] in ("WARNING", "CRITICAL")
    assert alert["metrics"]["blur_drift_ratio"] > 0.35


def test_confidence_drift_alert(monitor, frame_generator):
    """Verify degraded prediction confidences trigger CONFIDENCE_DRIFT alert."""
    frame, _ = frame_generator.generate_frame(surface_type="metal", defect_types=[])

    # Feed severely degraded confidence (0.35 vs baseline 0.85 -> > 50% drop)
    degraded_dets = [{"confidence": 0.35, "label": "crack", "class_id": 0, "bbox": [10, 10, 30, 30]}]
    alert = monitor.process_frame(frame, detections=degraded_dets)

    assert alert is not None
    assert alert["alert_type"] == "CONFIDENCE_DRIFT"
    assert alert["severity"] in ("WARNING", "CRITICAL")
    assert alert["metrics"]["confidence_drop_ratio"] > 0.25


def test_monitor_reset(monitor, frame_generator):
    """Verify monitor reset flushes history and alert queues."""
    frame, _ = frame_generator.generate_frame(surface_type="metal", defect_types=[])
    dark_frame = frame_generator.apply_lighting_drift(frame, factor=0.3)
    monitor.process_frame(dark_frame)

    assert len(monitor.get_recent_alerts()) > 0
    assert monitor.total_frames_inspected == 1

    monitor.reset()
    assert len(monitor.get_recent_alerts()) == 0
    assert monitor.total_frames_inspected == 0
    assert len(monitor.luminance_history) == 0
