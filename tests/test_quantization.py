"""Tests for INT8 quantization, compression ratios, and numerical parity."""

import os
import numpy as np
import onnx
import onnxruntime as ort
import pytest

from src.models.quantizer import (
    IndustrialCalibrationDataReader,
    evaluate_compression,
    quantize_onnx_model,
    verify_parity,
)


def test_fp32_model_export(model_paths):
    """Verify FP32 model exists, is valid ONNX, and runs on CPUExecutionProvider."""
    fp32_path = model_paths["fp32"]
    assert os.path.exists(fp32_path), "FP32 ONNX model file does not exist"

    # Validate ONNX protobuf
    onnx_model = onnx.load(fp32_path)
    onnx.checker.check_model(onnx_model)

    # Validate inference
    session = ort.InferenceSession(fp32_path, providers=["CPUExecutionProvider"])
    dummy_input = np.random.rand(1, 3, 256, 256).astype(np.float32)
    outs = session.run(None, {session.get_inputs()[0].name: dummy_input})

    assert len(outs) == 1
    assert outs[0].shape == (1, 256, 8)


def test_int8_quantization_compression(model_paths):
    """Verify INT8 model achieves significant size reduction compared to FP32."""
    fp32_path = model_paths["fp32"]
    int8_path = model_paths["int8"]

    assert os.path.exists(int8_path), "INT8 ONNX model file does not exist"

    stats = evaluate_compression(fp32_path, int8_path)
    assert stats["int8_size_bytes"] < stats["fp32_size_bytes"], "INT8 model is not smaller than FP32"
    assert stats["reduction_pct"] >= 20.0, f"Expected at least 20% size reduction, got {stats['reduction_pct']}%"
    assert stats["compression_ratio"] > 1.25, f"Expected compression ratio > 1.25x, got {stats['compression_ratio']}x"


def test_int8_output_shape_and_numerical_parity(model_paths):
    """Verify FP32 and INT8 produce identical output shapes and high correlation."""
    fp32_path = model_paths["fp32"]
    int8_path = model_paths["int8"]

    parity = verify_parity(fp32_path, int8_path)

    # Cosine similarity between FP32 and INT8 outputs should be high
    assert parity["cosine_similarity"] > 0.85, (
        f"Cosine similarity too low: {parity['cosine_similarity']}"
    )
    # Mean absolute error should remain bounded
    assert parity["mean_absolute_error"] < 0.20, (
        f"Mean absolute error too high: {parity['mean_absolute_error']}"
    )


def test_calibration_data_reader():
    """Verify industrial calibration data reader yields valid batches."""
    reader = IndustrialCalibrationDataReader(input_name="images", num_samples=10, seed=42)
    count = 0

    batch = reader.get_next()
    while batch is not None:
        count += 1
        assert "images" in batch
        assert batch["images"].shape == (1, 3, 256, 256)
        assert batch["images"].dtype == np.float32
        assert 0.0 <= np.min(batch["images"]) <= 1.0
        assert 0.0 <= np.max(batch["images"]) <= 1.0
        batch = reader.get_next()

    assert count == 10
