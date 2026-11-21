"""Custom edge frame distribution and prediction confidence drift monitor.

Monitors real-time camera stream for:
1. Lighting shifts (ambient illumination changes, overexposure, underexposure)
2. Optical blur / vibration (Laplacian variance degradation)
3. Model prediction confidence degradation (distribution shift or out-of-distribution inputs)
"""

from collections import deque
import logging
import time
from typing import Any, Dict, List, Optional
import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("drift_monitor")


class DriftMonitor:
    """Real-time industrial camera drift and prediction telemetry monitor."""

    def __init__(
        self,
        baseline_luminance: float = 155.0,
        baseline_blur_var: float = 220.0,
        baseline_confidence: float = 0.85,
        lighting_warn_threshold: float = 0.20,
        lighting_crit_threshold: float = 0.35,
        blur_warn_threshold: float = 0.35,
        blur_crit_threshold: float = 0.55,
        conf_drop_warn_threshold: float = 0.25,
        conf_drop_crit_threshold: float = 0.45,
        window_size: int = 50,
    ):
        """Initialize drift monitor with baseline thresholds.

        Args:
            baseline_luminance: Expected mean grayscale intensity.
            baseline_blur_var: Expected Laplacian variance for focused image.
            baseline_confidence: Expected average confidence for positive detections.
            lighting_warn_threshold: Fractional luminance deviation triggering warning.
            lighting_crit_threshold: Fractional luminance deviation triggering critical alert.
            blur_warn_threshold: Fractional blur drop triggering warning.
            blur_crit_threshold: Fractional blur drop triggering critical alert.
            conf_drop_warn_threshold: Fractional confidence drop triggering warning.
            conf_drop_crit_threshold: Fractional confidence drop triggering critical alert.
            window_size: Rolling window length for metrics smoothing.
        """
        self.baseline_luminance = float(baseline_luminance)
        self.baseline_blur_var = float(baseline_blur_var)
        self.baseline_confidence = float(baseline_confidence)

        self.lighting_warn_threshold = lighting_warn_threshold
        self.lighting_crit_threshold = lighting_crit_threshold
        self.blur_warn_threshold = blur_warn_threshold
        self.blur_crit_threshold = blur_crit_threshold
        self.conf_drop_warn_threshold = conf_drop_warn_threshold
        self.conf_drop_crit_threshold = conf_drop_crit_threshold

        self.window_size = window_size

        # Rolling history buffers
        self.luminance_history: deque = deque(maxlen=window_size)
        self.blur_history: deque = deque(maxlen=window_size)
        self.confidence_history: deque = deque(maxlen=window_size)
        self.alerts_history: deque = deque(maxlen=100)

        self.total_frames_inspected: int = 0
        self.is_calibrated: bool = False

    def calibrate_baseline(
        self,
        reference_frames: List[np.ndarray],
        baseline_confidence: Optional[float] = None,
    ) -> Dict[str, float]:
        """Calibrate reference baselines from known pristine edge conveyor frames.

        Args:
            reference_frames: List of pristine calibration frames [H, W, 3] or [H, W].
            baseline_confidence: Optional expected reference confidence.

        Returns:
            Dict containing calibrated baseline metrics.
        """
        if not reference_frames:
            raise ValueError("Reference frames list cannot be empty for calibration.")

        lum_vals = []
        blur_vals = []

        for frame in reference_frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
            lum_vals.append(float(np.mean(gray)))
            blur_vals.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))

        self.baseline_luminance = float(np.mean(lum_vals))
        self.baseline_blur_var = float(np.mean(blur_vals))
        if baseline_confidence is not None:
            self.baseline_confidence = float(baseline_confidence)

        self.is_calibrated = True
        logger.info(
            "Drift monitor calibrated: Baseline Luminance=%.2f, Blur Var=%.2f, Confidence=%.2f",
            self.baseline_luminance,
            self.baseline_blur_var,
            self.baseline_confidence,
        )

        return {
            "baseline_luminance": round(self.baseline_luminance, 2),
            "baseline_blur_var": round(self.baseline_blur_var, 2),
            "baseline_confidence": round(self.baseline_confidence, 2),
        }

    def process_frame(
        self,
        frame: np.ndarray,
        detections: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Analyze incoming edge frame for visual and model confidence drift.

        Args:
            frame: Raw inspection frame [H, W, 3] or [H, W].
            detections: List of inference detections produced by engine.

        Returns:
            Drift alert dict if drift detected above threshold, otherwise None.
        """
        self.total_frames_inspected += 1

        # 1. Compute frame luminance
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        current_luminance = float(np.mean(gray))
        self.luminance_history.append(current_luminance)

        # 2. Compute optical blur variance (Laplacian)
        current_blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        self.blur_history.append(current_blur_var)

        # 3. Track model confidence
        if detections and len(detections) > 0:
            for det in detections:
                self.confidence_history.append(float(det.get("confidence", 0.0)))

        # Evaluate drift metrics
        metrics = self.get_metrics()
        alert = None

        # Check Lighting Drift
        lum_drift = metrics["luminance_drift_ratio"]
        if lum_drift >= self.lighting_crit_threshold:
            alert = self._create_alert(
                alert_type="LIGHTING_DRIFT",
                severity="CRITICAL",
                message=(
                    f"Severe lighting deviation detected ({round(lum_drift * 100, 1)}% shift). "
                    f"Current: {round(current_luminance, 1)} vs Baseline: {round(self.baseline_luminance, 1)}."
                ),
                metrics=metrics,
            )
        elif lum_drift >= self.lighting_warn_threshold:
            alert = self._create_alert(
                alert_type="LIGHTING_DRIFT",
                severity="WARNING",
                message=(
                    f"Moderate lighting shift detected ({round(lum_drift * 100, 1)}% shift). "
                    f"Check industrial illuminator lamp or enclosure."
                ),
                metrics=metrics,
            )

        # Check Optical Blur Drift
        blur_drift = metrics["blur_drift_ratio"]
        if alert is None or alert["severity"] != "CRITICAL":
            if blur_drift >= self.blur_crit_threshold:
                alert = self._create_alert(
                    alert_type="BLUR_DRIFT",
                    severity="CRITICAL",
                    message=(
                        f"Critical lens blur / vibration detected ({round(blur_drift * 100, 1)}% variance loss). "
                        f"Current var: {round(current_blur_var, 1)} vs Baseline: {round(self.baseline_blur_var, 1)}."
                    ),
                    metrics=metrics,
                )
            elif blur_drift >= self.blur_warn_threshold:
                alert = self._create_alert(
                    alert_type="BLUR_DRIFT",
                    severity="WARNING",
                    message=(
                        f"Optical focus degradation warning ({round(blur_drift * 100, 1)}% drop). "
                        f"Inspect lens surface for smudges or vibration."
                    ),
                    metrics=metrics,
                )

        # Check Prediction Confidence Drift
        conf_drop = metrics["confidence_drop_ratio"]
        if alert is None or alert["severity"] != "CRITICAL":
            if conf_drop >= self.conf_drop_crit_threshold:
                alert = self._create_alert(
                    alert_type="CONFIDENCE_DRIFT",
                    severity="CRITICAL",
                    message=(
                        f"Critical prediction confidence collapse ({round(conf_drop * 100, 1)}% drop). "
                        f"Model may be encountering out-of-distribution defects or altered surface material."
                    ),
                    metrics=metrics,
                )
            elif conf_drop >= self.conf_drop_warn_threshold:
                alert = self._create_alert(
                    alert_type="CONFIDENCE_DRIFT",
                    severity="WARNING",
                    message=(
                        f"Prediction confidence degradation warning ({round(conf_drop * 100, 1)}% drop). "
                        f"Recent avg: {metrics['recent_avg_confidence']:.3f} vs Baseline: {self.baseline_confidence:.3f}."
                    ),
                    metrics=metrics,
                )

        if alert:
            self.alerts_history.append(alert)

        return alert

    def _create_alert(
        self,
        alert_type: str,
        severity: str,
        message: str,
        metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Construct a structured telemetry alert dictionary."""
        return {
            "alert_type": alert_type,
            "severity": severity,
            "message": message,
            "timestamp": time.time(),
            "metrics": metrics,
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Compute instantaneous and smoothed drift statistics."""
        current_lum = self.luminance_history[-1] if self.luminance_history else self.baseline_luminance
        current_blur = self.blur_history[-1] if self.blur_history else self.baseline_blur_var

        # Mean smoothed values over recent window
        mean_lum = float(np.mean(self.luminance_history)) if self.luminance_history else current_lum
        mean_blur = float(np.mean(self.blur_history)) if self.blur_history else current_blur
        recent_conf = float(np.mean(self.confidence_history)) if self.confidence_history else self.baseline_confidence

        # Fractional drift ratios
        lum_drift_ratio = abs(mean_lum - self.baseline_luminance) / max(self.baseline_luminance, 1.0)
        blur_drift_ratio = max(0.0, (self.baseline_blur_var - mean_blur) / max(self.baseline_blur_var, 1.0))
        conf_drop_ratio = max(0.0, (self.baseline_confidence - recent_conf) / max(self.baseline_confidence, 0.01))

        drift_detected = (
            lum_drift_ratio >= self.lighting_warn_threshold
            or blur_drift_ratio >= self.blur_warn_threshold
            or conf_drop_ratio >= self.conf_drop_warn_threshold
        )

        return {
            "current_luminance": round(current_lum, 2),
            "smoothed_luminance": round(mean_lum, 2),
            "baseline_luminance": round(self.baseline_luminance, 2),
            "luminance_drift_ratio": round(lum_drift_ratio, 4),
            "current_blur_variance": round(current_blur, 2),
            "smoothed_blur_variance": round(mean_blur, 2),
            "baseline_blur_variance": round(self.baseline_blur_var, 2),
            "blur_drift_ratio": round(blur_drift_ratio, 4),
            "recent_avg_confidence": round(recent_conf, 4),
            "baseline_confidence": round(self.baseline_confidence, 4),
            "confidence_drop_ratio": round(conf_drop_ratio, 4),
            "drift_detected": drift_detected,
            "total_frames_inspected": self.total_frames_inspected,
        }

    def get_recent_alerts(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return list of recent alerts up to specified limit."""
        return list(self.alerts_history)[-limit:]

    def reset(self) -> None:
        """Clear rolling metrics history and alerts."""
        self.luminance_history.clear()
        self.blur_history.clear()
        self.confidence_history.clear()
        self.alerts_history.clear()
        self.total_frames_inspected = 0
