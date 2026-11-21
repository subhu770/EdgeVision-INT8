"""Models module: PyTorch architecture definition, ONNX export, and INT8 quantization."""

from src.models.builder import LightweightDefectDetector, build_and_export_model
from src.models.quantizer import quantize_onnx_model, evaluate_compression, verify_parity

__all__ = [
    "LightweightDefectDetector",
    "build_and_export_model",
    "quantize_onnx_model",
    "evaluate_compression",
    "verify_parity",
]
