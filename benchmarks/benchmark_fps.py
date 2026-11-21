"""EdgeVision-INT8: High-Performance Industrial Defect Detection Benchmark.

Benchmarks FP32 vs INT8:
- Model size in MB and compression ratio
- P50, P90, P95, and P99 latency in milliseconds
- Real-time throughput (Frames Per Second)
- Detection retention (mean IoU and defect recall parity) across 200 synthetic industrial frames.

Zero GPU or external dataset downloads required. Runs completely on CPU.
"""

import argparse
import logging
import os
import sys
import time
from typing import Any, Dict, List, Tuple
import numpy as np

# Ensure root directory is on pythonpath
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.inference.engine import DefectInferenceEngine
from src.models.builder import build_and_export_model
from src.models.quantizer import evaluate_compression, quantize_onnx_model
from src.simulation.frame_generator import IndustrialDefectGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark")


def compute_box_iou(box1: List[float], box2: List[float]) -> float:
    """Compute Intersection-over-Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

    union = area1 + area2 - intersection
    if union <= 1e-6:
        return 0.0
    return float(intersection / union)


def evaluate_detection_retention(
    fp32_dets_list: List[List[Dict[str, Any]]],
    int8_dets_list: List[List[Dict[str, Any]]],
) -> Dict[str, float]:
    """Calculate bounding box IoU retention and class consistency between FP32 and INT8."""
    all_ious = []
    class_matches = 0
    total_comparisons = 0

    for fp32_dets, int8_dets in zip(fp32_dets_list, int8_dets_list):
        if not fp32_dets and not int8_dets:
            all_ious.append(1.0)
            continue
        if not fp32_dets or not int8_dets:
            all_ious.append(0.0)
            continue

        for f_det in fp32_dets:
            best_iou = 0.0
            best_match_cls = False
            for i_det in int8_dets:
                iou = compute_box_iou(f_det["bbox"], i_det["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_match_cls = (f_det["class_id"] == i_det["class_id"])

            all_ious.append(best_iou)
            total_comparisons += 1
            if best_match_cls and best_iou > 0.4:
                class_matches += 1

    mean_iou = float(np.mean(all_ious)) if all_ious else 1.0
    cls_retention = (class_matches / max(total_comparisons, 1)) * 100.0

    return {
        "mean_iou_retention": round(mean_iou, 4),
        "class_retention_pct": round(cls_retention, 2),
    }


def run_benchmark(
    num_frames: int = 200,
    fp32_model_path: str = "models/defect_detector_fp32.onnx",
    int8_model_path: str = "models/defect_detector_int8.onnx",
    warmup_runs: int = 20,
    num_threads: int = 4,
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute complete FP32 vs INT8 edge benchmark across synthetic frames."""
    print("=" * 80)
    print(" " * 18 + "EdgeVision-INT8: CPU Latency & Parity Benchmark")
    print("=" * 80)

    # 1. Ensure models exist
    if not os.path.exists(fp32_model_path):
        logger.info("FP32 model not found at '%s'. Building model...", fp32_model_path)
        build_and_export_model(output_path=fp32_model_path, train_synthetic=True, seed=seed)

    if not os.path.exists(int8_model_path):
        logger.info("INT8 model not found at '%s'. Quantizing model...", int8_model_path)
        quantize_onnx_model(input_fp32_path=fp32_model_path, output_int8_path=int8_model_path, method="dynamic")

    # 2. Model Compression Stats
    compression_stats = evaluate_compression(fp32_model_path, int8_model_path)
    fp32_mb = compression_stats["fp32_size_mb"]
    int8_mb = compression_stats["int8_size_mb"]
    reduction = compression_stats["reduction_pct"]
    ratio = compression_stats["compression_ratio"]

    print(f"\n[Model Footprint]")
    print(f"  • FP32 Model Size: {fp32_mb:.2f} MB")
    print(f"  • INT8 Model Size: {int8_mb:.2f} MB")
    print(f"  • Size Reduction:  {reduction:.1f}% ({ratio:.2f}x compression)")

    # 3. Generate deterministic synthetic dataset
    logger.info("Generating %d deterministic synthetic industrial frames...", num_frames)
    generator = IndustrialDefectGenerator(width=256, height=256, seed=seed)
    dataset = generator.generate_batch(num_frames=num_frames, defect_ratio=0.6, seed=seed)
    images = [item[0] for item in dataset]
    ground_truths = [item[1] for item in dataset]

    # 4. Initialize Inference Engines
    logger.info("Initializing FP32 and INT8 ONNX Runtime inference engines (%d threads)...", num_threads)
    engine_fp32 = DefectInferenceEngine(fp32_model_path, conf_threshold=0.35, num_threads=num_threads)
    engine_int8 = DefectInferenceEngine(int8_model_path, conf_threshold=0.35, num_threads=num_threads)

    # 5. Warmup
    logger.info("Warming up inference engines (%d runs)...", warmup_runs)
    for _ in range(warmup_runs):
        _ = engine_fp32.predict(images[0])
        _ = engine_int8.predict(images[0])

    # 6. Benchmark FP32
    logger.info("Benchmarking FP32 engine over %d frames...", num_frames)
    fp32_latencies = []
    fp32_all_dets = []

    start_total_fp32 = time.perf_counter()
    for frame in images:
        dets, lat = engine_fp32.predict(frame)
        fp32_latencies.append(lat["total_latency_ms"])
        fp32_all_dets.append(dets)
    total_time_fp32 = time.perf_counter() - start_total_fp32
    fps_fp32 = num_frames / total_time_fp32

    # 7. Benchmark INT8
    logger.info("Benchmarking INT8 engine over %d frames...", num_frames)
    int8_latencies = []
    int8_all_dets = []

    start_total_int8 = time.perf_counter()
    for frame in images:
        dets, lat = engine_int8.predict(frame)
        int8_latencies.append(lat["total_latency_ms"])
        int8_all_dets.append(dets)
    total_time_int8 = time.perf_counter() - start_total_int8
    fps_int8 = num_frames / total_time_int8

    # 8. Compute Retention and IoU Parity
    retention_stats = evaluate_detection_retention(fp32_all_dets, int8_all_dets)

    # 9. Aggregate Latency Percentiles
    fp32_lat_arr = np.array(fp32_latencies)
    int8_lat_arr = np.array(int8_latencies)

    p50_fp32 = float(np.percentile(fp32_lat_arr, 50))
    p90_fp32 = float(np.percentile(fp32_lat_arr, 90))
    p95_fp32 = float(np.percentile(fp32_lat_arr, 95))
    p99_fp32 = float(np.percentile(fp32_lat_arr, 99))
    mean_fp32 = float(np.mean(fp32_lat_arr))

    p50_int8 = float(np.percentile(int8_lat_arr, 50))
    p90_int8 = float(np.percentile(int8_lat_arr, 90))
    p95_int8 = float(np.percentile(int8_lat_arr, 95))
    p99_int8 = float(np.percentile(int8_lat_arr, 99))
    mean_int8 = float(np.mean(int8_lat_arr))

    speedup_fps = fps_int8 / max(fps_fp32, 1e-6)
    p50_gain = ((p50_fp32 - p50_int8) / max(p50_fp32, 1e-6)) * 100.0

    # 10. Display Benchmark Table
    ratio_str = f"{ratio:.2f}x"
    cls_str = f"{retention_stats['class_retention_pct']:.1f}%"

    print("\n" + "=" * 80)
    print(f"{'Metric':<30} | {'FP32 Baseline':<20} | {'INT8 Quantized':<20}")
    print("-" * 80)
    print(f"{'Model File Size (MB)':<30} | {fp32_mb:<20.2f} | {int8_mb:<20.2f}")
    print(f"{'Compression Ratio':<30} | {'1.00x':<20} | {ratio_str:<20}")
    print(f"{'Throughput (FPS)':<30} | {fps_fp32:<20.1f} | {fps_int8:<20.1f}")
    print(f"{'Latency Mean (ms)':<30} | {mean_fp32:<20.2f} | {mean_int8:<20.2f}")
    print(f"{'Latency P50 (ms)':<30} | {p50_fp32:<20.2f} | {p50_int8:<20.2f}")
    print(f"{'Latency P90 (ms)':<30} | {p90_fp32:<20.2f} | {p90_int8:<20.2f}")
    print(f"{'Latency P95 (ms)':<30} | {p95_fp32:<20.2f} | {p95_int8:<20.2f}")
    print(f"{'Latency P99 (ms)':<30} | {p99_fp32:<20.2f} | {p99_int8:<20.2f}")
    print("-" * 80)
    print(f"{'Mean IoU Retention':<30} | {'1.0000':<20} | {retention_stats['mean_iou_retention']:<20.4f}")
    print(f"{'Class Agreement Rate':<30} | {'100.0%':<20} | {cls_str:<20}")
    print("=" * 80)
    print(f"\n[Summary Insights]")
    print(f"  • Throughput Gain: {speedup_fps:.2f}x speedup ({fps_fp32:.1f} FPS -> {fps_int8:.1f} FPS)")
    print(f"  • P50 Latency Reduction: {p50_gain:.1f}% drop ({p50_fp32:.2f} ms -> {p50_int8:.2f} ms)")
    print(f"  • Edge Memory Footprint: Saved {fp32_mb - int8_mb:.2f} MB on disk and RAM")
    print(f"  • Parity Retention: {retention_stats['mean_iou_retention']:.4f} IoU retained")
    print("=" * 80 + "\n")

    return {
        "fp32_size_mb": fp32_mb,
        "int8_size_mb": int8_mb,
        "fps_fp32": fps_fp32,
        "fps_int8": fps_int8,
        "p50_fp32_ms": p50_fp32,
        "p50_int8_ms": p50_int8,
        "p95_fp32_ms": p95_fp32,
        "p95_int8_ms": p95_int8,
        "mean_iou_retention": retention_stats["mean_iou_retention"],
        "class_retention_pct": retention_stats["class_retention_pct"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark FP32 vs INT8 EdgeVision defect detector.")
    parser.add_argument("--frames", type=int, default=200, help="Number of benchmark evaluation frames")
    parser.add_argument("--threads", type=int, default=4, help="CPU execution thread count")
    parser.add_argument("--fp32", type=str, default="models/defect_detector_fp32.onnx", help="FP32 model path")
    parser.add_argument("--int8", type=str, default="models/defect_detector_int8.onnx", help="INT8 model path")
    parser.add_argument("--seed", type=int, default=42, help="Synthetic generator seed")
    args = parser.parse_args()

    results = run_benchmark(
        num_frames=args.frames,
        fp32_model_path=args.fp32,
        int8_model_path=args.int8,
        num_threads=args.threads,
        seed=args.seed,
    )
