"""ONNX Runtime Dynamic and Static INT8 Quantization module.

Applies post-training quantization to reduce edge model footprint, improve inference
throughput, and compare FP32 vs INT8 latency, compression, and numerical parity.
"""

import argparse
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)

from src.simulation.frame_generator import IndustrialDefectGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("model_quantizer")


class IndustrialCalibrationDataReader(CalibrationDataReader):
    """Feeds synthetic industrial surface images into ONNX Runtime static calibrator."""

    def __init__(self, input_name: str = "images", num_samples: int = 50, seed: int = 123):
        super().__init__()
        self.input_name = input_name
        self.generator = IndustrialDefectGenerator(width=256, height=256, seed=seed)
        self.samples = self.generator.generate_calibration_samples(num_samples=num_samples, seed=seed)
        self.enum_data = iter(self.samples)

    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        """Return next input dict or None when calibration dataset is exhausted."""
        try:
            sample = next(self.enum_data)
            return {self.input_name: sample}
        except StopIteration:
            return None

    def rewind(self) -> None:
        """Rewind the data iterator for multi-pass calibration."""
        self.enum_data = iter(self.samples)


def _get_total_model_size(path: str) -> int:
    """Get total model size on disk including external tensor data files if present."""
    total = os.path.getsize(path)
    data_file = path + ".data"
    if os.path.exists(data_file):
        total += os.path.getsize(data_file)
    return total


def evaluate_compression(fp32_path: str, int8_path: str) -> Dict[str, float]:
    """Calculate file size reduction and compression ratio between FP32 and INT8 models.

    Args:
        fp32_path: Filepath to FP32 ONNX model.
        int8_path: Filepath to INT8 ONNX model.

    Returns:
        Dict with fp32_size_mb, int8_size_mb, compression_ratio, and reduction_pct.
    """
    if not os.path.exists(fp32_path):
        raise FileNotFoundError(f"FP32 model not found at: {fp32_path}")
    if not os.path.exists(int8_path):
        raise FileNotFoundError(f"INT8 model not found at: {int8_path}")

    fp32_bytes = _get_total_model_size(fp32_path)
    int8_bytes = _get_total_model_size(int8_path)

    fp32_mb = fp32_bytes / (1024 * 1024)
    int8_mb = int8_bytes / (1024 * 1024)
    compression_ratio = fp32_bytes / max(int8_bytes, 1)
    reduction_pct = ((fp32_bytes - int8_bytes) / max(fp32_bytes, 1)) * 100.0

    return {
        "fp32_size_bytes": float(fp32_bytes),
        "int8_size_bytes": float(int8_bytes),
        "fp32_size_mb": round(fp32_mb, 3),
        "int8_size_mb": round(int8_mb, 3),
        "compression_ratio": round(compression_ratio, 2),
        "reduction_pct": round(reduction_pct, 2),
    }


def verify_parity(
    fp32_path: str,
    int8_path: str,
    sample_inputs: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Verify numerical parity and output correlation between FP32 and INT8 models.

    Args:
        fp32_path: Path to FP32 ONNX model.
        int8_path: Path to INT8 ONNX model.
        sample_inputs: Optional [B, 3, 256, 256] input tensor. If None, generated automatically.

    Returns:
        Dict with max_absolute_error, mean_absolute_error, cosine_similarity.
    """
    if sample_inputs is None:
        generator = IndustrialDefectGenerator(width=256, height=256, seed=777)
        sample_inputs = generator.generate_calibration_samples(num_samples=1)[0]

    session_fp32 = ort.InferenceSession(fp32_path, providers=["CPUExecutionProvider"])
    session_int8 = ort.InferenceSession(int8_path, providers=["CPUExecutionProvider"])

    input_name_fp32 = session_fp32.get_inputs()[0].name
    input_name_int8 = session_int8.get_inputs()[0].name

    out_fp32 = session_fp32.run(None, {input_name_fp32: sample_inputs})[0].flatten()
    out_int8 = session_int8.run(None, {input_name_int8: sample_inputs})[0].flatten()

    max_err = float(np.max(np.abs(out_fp32 - out_int8)))
    mean_err = float(np.mean(np.abs(out_fp32 - out_int8)))

    # Cosine similarity
    norm_fp32 = np.linalg.norm(out_fp32)
    norm_int8 = np.linalg.norm(out_int8)
    cos_sim = float(np.dot(out_fp32, out_int8) / (norm_fp32 * norm_int8 + 1e-8))

    return {
        "max_absolute_error": round(max_err, 5),
        "mean_absolute_error": round(mean_err, 5),
        "cosine_similarity": round(cos_sim, 5),
    }


def quantize_onnx_model(
    input_fp32_path: str,
    output_int8_path: str,
    method: str = "dynamic",
    calibration_samples: int = 50,
) -> str:
    """Quantize an FP32 ONNX defect detection model to INT8 precision.

    Args:
        input_fp32_path: Path to source FP32 ONNX model.
        output_int8_path: Destination path for quantized INT8 ONNX model.
        method: Quantization method ('dynamic' or 'static').
        calibration_samples: Number of synthetic industrial calibration frames for static mode.

    Returns:
        Absolute path to quantized INT8 ONNX model.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_int8_path)), exist_ok=True)
    logger.info("Starting INT8 quantization (%s method): %s -> %s", method, input_fp32_path, output_int8_path)

    if method == "static":
        try:
            logger.info("Generating %d synthetic calibration frames for static QDQ quantization...", calibration_samples)
            data_reader = IndustrialCalibrationDataReader(
                input_name="images",
                num_samples=calibration_samples,
                seed=123,
            )
            quantize_static(
                model_input=input_fp32_path,
                model_output=output_int8_path,
                calibration_data_reader=data_reader,
                quant_format=QuantFormat.QDQ,
                activation_type=QuantType.QInt8,
                weight_type=QuantType.QInt8,
                per_channel=True,
            )
            logger.info("Static QDQ quantization completed successfully.")
        except Exception as exc:
            logger.warning("Static quantization encountered an issue (%s); falling back to Dynamic quantization.", exc)
            quantize_dynamic(
                model_input=input_fp32_path,
                model_output=output_int8_path,
                weight_type=QuantType.QInt8,
            )
    else:
        quantize_dynamic(
            model_input=input_fp32_path,
            model_output=output_int8_path,
            weight_type=QuantType.QInt8,
        )
        logger.info("Dynamic INT8 quantization completed successfully.")

    # Validate output
    comp = evaluate_compression(input_fp32_path, output_int8_path)
    logger.info(
        "Quantization Results: FP32: %.2f MB -> INT8: %.2f MB (%.1f%% reduction, %.2fx compression ratio)",
        comp["fp32_size_mb"],
        comp["int8_size_mb"],
        comp["reduction_pct"],
        comp["compression_ratio"],
    )

    parity = verify_parity(input_fp32_path, output_int8_path)
    logger.info(
        "Parity Check: Mean Abs Error: %.5f | Cosine Similarity: %.5f",
        parity["mean_absolute_error"],
        parity["cosine_similarity"],
    )

    return os.path.abspath(output_int8_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize EdgeVision-INT8 defect detector model.")
    parser.add_argument("--input", type=str, default="models/defect_detector_fp32.onnx", help="FP32 ONNX model path")
    parser.add_argument("--output", type=str, default="models/defect_detector_int8.onnx", help="INT8 ONNX model path")
    parser.add_argument("--method", type=str, default="dynamic", choices=["dynamic", "static"], help="Quantization method")
    parser.add_argument("--samples", type=int, default=50, help="Calibration sample count for static quantization")
    args = parser.parse_args()

    quantize_onnx_model(
        input_fp32_path=args.input,
        output_int8_path=args.output,
        method=args.method,
        calibration_samples=args.samples,
    )
