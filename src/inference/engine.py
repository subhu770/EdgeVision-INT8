"""High-throughput edge ONNX Runtime inference engine.

Provides zero-copy / minimal-allocation pre-processing, multi-threaded CPU execution,
and vectorized Non-Maximum Suppression (NMS) post-processing with latency telemetry.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
import onnxruntime as ort

from src.simulation.frame_generator import DEFECT_CLASSES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("inference_engine")


def nms_fast(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45) -> List[int]:
    """Vectorized Non-Maximum Suppression (NMS) in pure NumPy.

    Args:
        boxes: Coordinates array [N, 4] formatted as [x1, y1, x2, y2].
        scores: Confidence scores array [N].
        iou_threshold: Overlap IoU threshold for box suppression.

    Returns:
        List of selected detection indices.
    """
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        y2_min = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, y2_min - yy1)
        intersection = w * h
        union = areas[i] + areas[order[1:]] - intersection
        iou = intersection / np.maximum(union, 1e-6)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


class DefectInferenceEngine:
    """Production-grade edge inference engine utilizing ONNX Runtime."""

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        num_threads: int = 4,
    ):
        """Initialize the ONNX Runtime inference engine.

        Args:
            model_path: Path to FP32 or quantized INT8 ONNX model file.
            conf_threshold: Minimum confidence score to accept detection.
            iou_threshold: NMS intersection-over-union threshold.
            num_threads: Thread pool size for CPU execution provider.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        self.model_path = os.path.abspath(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.num_threads = num_threads

        # Configure session options for edge CPU execution
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        opts.inter_op_num_threads = num_threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        logger.info("Loading ONNX session: %s (threads: %d)", self.model_path, num_threads)
        self.session = ort.InferenceSession(
            self.model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )

        # Inspect input metadata
        input_meta = self.session.get_inputs()[0]
        self.input_name = input_meta.name
        self.input_shape = input_meta.shape  # e.g., ['batch_size', 3, 256, 256]
        self.target_height = int(self.input_shape[2]) if isinstance(self.input_shape[2], int) else 256
        self.target_width = int(self.input_shape[3]) if isinstance(self.input_shape[3], int) else 256

        self.output_name = self.session.get_outputs()[0].name
        logger.info(
            "Engine ready | Input: %s %s -> Output: %s",
            self.input_name,
            (self.target_height, self.target_width),
            self.output_name,
        )

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
        """Preprocess an input frame with letterbox aspect ratio preservation.

        Args:
            image: Input image [H, W, 3] or [H, W] in BGR or grayscale.

        Returns:
            Tuple of (preprocessed tensor [1, 3, H, W] float32, (scale_ratio, pad_x, pad_y, orig_dims)).
        """
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        orig_h, orig_w = image.shape[:2]
        target_w, target_h = self.target_width, self.target_height

        # Fast letterbox scale calculation
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = int(round(orig_w * scale))
        new_h = int(round(orig_h * scale))

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Letterbox padding
        pad_x = (target_w - new_w) / 2.0
        pad_y = (target_h - new_h) / 2.0
        top = int(round(pad_y - 0.1))
        bottom = int(round(target_h - new_h - top))
        left = int(round(pad_x - 0.1))
        right = int(round(target_w - new_w - left))

        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )

        # BGR -> RGB, HWC -> CHW, normalize to [0, 1]
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) * (1.0 / 255.0)
        chw = np.transpose(rgb, (2, 0, 1))
        tensor = np.expand_dims(chw, axis=0)

        metadata = (scale, pad_x, pad_y, float(orig_w), float(orig_h))
        return tensor, metadata

    def postprocess(
        self,
        raw_output: np.ndarray,
        metadata: Tuple[float, float, float, float, float],
    ) -> List[Dict[str, Any]]:
        """Decode bounding boxes, apply confidence thresholding and vectorized NMS.

        Args:
            raw_output: Tensor [1, 256, 8] from ONNX model.
            metadata: (scale, pad_x, pad_y, orig_w, orig_h).

        Returns:
            List of detected defect objects with label, confidence, and bounding box.
        """
        preds = raw_output[0]  # [256, 8]
        scale, pad_x, pad_y, orig_w, orig_h = metadata

        cx = preds[:, 0]
        cy = preds[:, 1]
        w = preds[:, 2]
        h = preds[:, 3]
        obj = preds[:, 4]
        cls_probs = preds[:, 5:8]

        class_ids = np.argmax(cls_probs, axis=-1)
        class_confs = np.max(cls_probs, axis=-1)
        total_scores = obj * class_confs

        # Filter by confidence threshold
        mask = total_scores >= self.conf_threshold
        if not np.any(mask):
            return []

        cx = cx[mask]
        cy = cy[mask]
        w = w[mask]
        h = h[mask]
        scores = total_scores[mask]
        cids = class_ids[mask]

        # Convert normalized center-width-height to pixel coordinates
        x1_px = (cx - w / 2.0) * self.target_width
        y1_px = (cy - h / 2.0) * self.target_height
        x2_px = (cx + w / 2.0) * self.target_width
        y2_px = (cy + h / 2.0) * self.target_height

        # Invert letterbox transform back to original image coordinates
        x1_orig = np.clip((x1_px - pad_x) / scale, 0.0, orig_w)
        y1_orig = np.clip((y1_px - pad_y) / scale, 0.0, orig_h)
        x2_orig = np.clip((x2_px - pad_x) / scale, 0.0, orig_w)
        y2_orig = np.clip((y2_px - pad_y) / scale, 0.0, orig_h)

        boxes = np.stack([x1_orig, y1_orig, x2_orig, y2_orig], axis=1)

        # Apply multi-class NMS
        final_detections: List[Dict[str, Any]] = []
        unique_classes = np.unique(cids)

        for cid in unique_classes:
            cls_mask = (cids == cid)
            cls_boxes = boxes[cls_mask]
            cls_scores = scores[cls_mask]

            keep_indices = nms_fast(cls_boxes, cls_scores, self.iou_threshold)
            for idx in keep_indices:
                box = cls_boxes[idx]
                # Filter degenerate boxes
                if (box[2] - box[0]) < 2 or (box[3] - box[1]) < 2:
                    continue
                label = DEFECT_CLASSES[cid] if cid < len(DEFECT_CLASSES) else f"defect_{cid}"
                final_detections.append({
                    "label": label,
                    "class_id": int(cid),
                    "confidence": round(float(cls_scores[idx]), 4),
                    "bbox": [round(float(coord), 2) for coord in box],
                })

        # Sort detections descending by confidence
        final_detections.sort(key=lambda d: d["confidence"], reverse=True)
        return final_detections

    def predict(self, image: np.ndarray) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """Perform end-to-end inference with fine-grained stage latency measurements.

        Args:
            image: Raw input frame [H, W, 3] uint8.

        Returns:
            Tuple of (detections list, latency metrics dict in milliseconds).
        """
        t0 = time.perf_counter()
        tensor, meta = self.preprocess(image)
        t1 = time.perf_counter()

        raw_outputs = self.session.run([self.output_name], {self.input_name: tensor})
        t2 = time.perf_counter()

        detections = self.postprocess(raw_outputs[0], meta)
        t3 = time.perf_counter()

        latency_breakdown = {
            "preprocess_ms": round((t1 - t0) * 1000.0, 3),
            "inference_ms": round((t2 - t1) * 1000.0, 3),
            "postprocess_ms": round((t3 - t2) * 1000.0, 3),
            "total_latency_ms": round((t3 - t0) * 1000.0, 3),
        }

        return detections, latency_breakdown
