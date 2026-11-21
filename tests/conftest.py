"""Shared test fixtures and setup for EdgeVision-INT8 test suite."""

import os
import pytest
import numpy as np

# Set simulation mode environment variable
os.environ["SIMULATION_MODE"] = "true"

from src.models.builder import build_and_export_model
from src.models.quantizer import quantize_onnx_model
from src.simulation.frame_generator import IndustrialDefectGenerator


@pytest.fixture(scope="session")
def model_paths(tmp_path_factory):
    """Build and quantize session-wide test models, reusing pre-built models when present."""
    default_fp32 = "models/defect_detector_fp32.onnx"
    default_int8 = "models/defect_detector_int8.onnx"

    if os.path.exists(default_fp32) and os.path.exists(default_int8):
        return {
            "fp32": os.path.abspath(default_fp32),
            "int8": os.path.abspath(default_int8),
        }

    tmp_dir = tmp_path_factory.mktemp("test_models")
    fp32_path = str(tmp_dir / "defect_detector_fp32.onnx")
    int8_path = str(tmp_dir / "defect_detector_int8.onnx")

    # Build FP32 ONNX model
    build_and_export_model(output_path=fp32_path, train_synthetic=True, seed=42)

    # Quantize to INT8
    quantize_onnx_model(input_fp32_path=fp32_path, output_int8_path=int8_path, method="dynamic")

    return {
        "fp32": fp32_path,
        "int8": int8_path,
    }


@pytest.fixture
def frame_generator():
    """Deterministic synthetic defect generator instance."""
    return IndustrialDefectGenerator(width=256, height=256, seed=42)
