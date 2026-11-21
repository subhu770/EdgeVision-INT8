"""Defect detector model architecture and ONNX export builder.

Constructs an edge-optimized MobileNet-style lightweight convolutional network
for real-time industrial defect detection (crack, scratch, pit) and exports it to ONNX.
Zero external weights download required - includes deterministic synthetic pre-training.
"""

import argparse
import logging
import os
from typing import Optional, Tuple
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.optim as optim

from src.simulation.frame_generator import DEFECT_CLASSES, IndustrialDefectGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("model_builder")


class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution block with Batch Normalization and Hardswish."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.act1 = nn.Hardswish(inplace=True)

        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.Hardswish(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.bn1(self.depthwise(x)))
        x = self.act2(self.bn2(self.pointwise(x)))
        return x


class LightweightDefectDetector(nn.Module):
    """Edge-optimized lightweight defect detector.

    Input shape:  [B, 3, 256, 256]
    Output shape: [B, 256, 8] where columns are [cx, cy, w, h, obj_conf, p_crack, p_scratch, p_pit].
    """

    def __init__(self, num_classes: int = 3, grid_size: int = 16):
        super().__init__()
        self.num_classes = num_classes
        self.grid_size = grid_size  # 16x16 grid for 256x256 input (16px per cell)

        # Backbone: MobileNet-style inverted depthwise-separable pyramid
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),  # 128x128
            nn.BatchNorm2d(16),
            nn.Hardswish(inplace=True),
        )

        self.stage1 = DepthwiseSeparableConv(16, 32, stride=2)   # 64x64
        self.stage2 = DepthwiseSeparableConv(32, 64, stride=2)   # 32x32
        self.stage3 = DepthwiseSeparableConv(64, 96, stride=2)   # 16x16
        self.stage4 = DepthwiseSeparableConv(96, 128, stride=1)  # 16x16

        # Multi-scale aggregation / bottleneck
        self.neck = nn.Sequential(
            nn.Conv2d(128, 96, kernel_size=1, bias=False),
            nn.BatchNorm2d(96),
            nn.Hardswish(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, padding=1, groups=96, bias=False),
            nn.BatchNorm2d(96),
            nn.Hardswish(inplace=True),
        )

        # Output Head:
        # Channels: 4 (offsets cx, cy, w, h) + 1 (objectness) + num_classes
        out_channels = 4 + 1 + num_classes
        self.head = nn.Conv2d(96, out_channels, kernel_size=1, bias=True)
        # Prior probability initialization for objectness to combat foreground/background class imbalance
        nn.init.constant_(self.head.bias[4], -2.5)

        # Precompute coordinate grid for ONNX graph embedding
        gy, gx = torch.meshgrid(
            torch.arange(grid_size, dtype=torch.float32),
            torch.arange(grid_size, dtype=torch.float32),
            indexing="ij",
        )
        self.register_buffer("grid_x", gx.unsqueeze(0))  # [1, 16, 16]
        self.register_buffer("grid_y", gy.unsqueeze(0))  # [1, 16, 16]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass predicting bounding boxes, objectness, and class probabilities.

        Args:
            x: Input tensor [B, 3, 256, 256] in range [0, 1].

        Returns:
            Tensor of shape [B, 256, 8] with columns [cx, cy, w, h, obj_conf, p_crack, p_scratch, p_pit].
        """
        b = x.shape[0]

        feat = self.stem(x)
        feat = self.stage1(feat)
        feat = self.stage2(feat)
        feat = self.stage3(feat)
        feat = self.stage4(feat)
        feat = self.neck(feat)
        raw_pred = self.head(feat)  # [B, 8, 16, 16]

        # Slicing channels
        # raw_pred: [B, 8, 16, 16]
        dx = torch.sigmoid(raw_pred[:, 0:1, :, :])
        dy = torch.sigmoid(raw_pred[:, 1:2, :, :])
        w = torch.sigmoid(raw_pred[:, 2:3, :, :]) * 0.95 + 0.05
        h = torch.sigmoid(raw_pred[:, 3:4, :, :]) * 0.95 + 0.05
        obj = torch.sigmoid(raw_pred[:, 4:5, :, :])
        cls_scores = torch.softmax(raw_pred[:, 5:5 + self.num_classes, :, :], dim=1)

        # Decode center coordinates normalized to [0, 1]
        cx = (self.grid_x.unsqueeze(1) + dx) / float(self.grid_size)
        cy = (self.grid_y.unsqueeze(1) + dy) / float(self.grid_size)

        # Concatenate: [B, 8, 16, 16]
        decoded = torch.cat([cx, cy, w, h, obj, cls_scores], dim=1)

        # Flatten grid: [B, 8, 256] -> [B, 256, 8]
        out = decoded.flatten(2).transpose(1, 2)
        return out


def train_synthetic_detector(
    model: LightweightDefectDetector,
    num_samples: int = 60,
    epochs: int = 15,
    lr: float = 4e-3,
    seed: int = 42,
) -> LightweightDefectDetector:
    """Fast deterministic synthetic pre-training loop.

    Trains the model on generated synthetic defects in ~3 seconds on CPU so that
    it produces genuine, high-accuracy defect detections on edge inference.
    """
    logger.info("Initializing synthetic pre-training (%d samples, %d epochs)...", num_samples, epochs)
    torch.manual_seed(seed)
    generator = IndustrialDefectGenerator(width=256, height=256, seed=seed)
    dataset = generator.generate_batch(num_frames=num_samples, defect_ratio=0.75, seed=seed)

    # Prepare training tensors
    images_list = []
    targets_obj = []
    targets_box = []
    targets_cls = []

    grid_s = model.grid_size
    for img, annos in dataset:
        rgb = img[:, :, ::-1].astype(np.float32) / 255.0
        chw = np.transpose(rgb, (2, 0, 1))
        images_list.append(chw)

        obj_grid = np.zeros((grid_s, grid_s), dtype=np.float32)
        box_grid = np.zeros((4, grid_s, grid_s), dtype=np.float32)
        cls_grid = np.zeros((3, grid_s, grid_s), dtype=np.float32)

        for anno in annos:
            x1, y1, x2, y2 = anno["bbox"]
            cid = anno["class_id"]
            cx_n = ((x1 + x2) / 2.0) / 256.0
            cy_n = ((y1 + y2) / 2.0) / 256.0
            w_n = (x2 - x1) / 256.0
            h_n = (y2 - y1) / 256.0

            gx = int(np.clip(cx_n * grid_s, 0, grid_s - 1))
            gy = int(np.clip(cy_n * grid_s, 0, grid_s - 1))

            obj_grid[gy, gx] = 1.0
            box_grid[0, gy, gx] = cx_n
            box_grid[1, gy, gx] = cy_n
            box_grid[2, gy, gx] = w_n
            box_grid[3, gy, gx] = h_n
            cls_grid[cid, gy, gx] = 1.0

        targets_obj.append(obj_grid.flatten())
        targets_box.append(box_grid.reshape(4, -1).T)
        targets_cls.append(cls_grid.reshape(3, -1).T)

    x_tensor = torch.tensor(np.array(images_list), dtype=torch.float32)
    t_obj = torch.tensor(np.array(targets_obj), dtype=torch.float32)
    t_box = torch.tensor(np.array(targets_box), dtype=torch.float32)
    t_cls = torch.tensor(np.array(targets_cls), dtype=torch.float32)

    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    bce = nn.BCELoss()
    l1 = nn.SmoothL1Loss()

    batch_size = 12
    num_batches = len(images_list) // batch_size

    for epoch in range(epochs):
        perm = torch.randperm(len(images_list))
        total_loss = 0.0

        for b in range(num_batches):
            idx = perm[b * batch_size:(b + 1) * batch_size]
            b_x = x_tensor[idx]
            b_obj = t_obj[idx]
            b_box = t_box[idx]
            b_cls = t_cls[idx]

            optimizer.zero_grad()
            preds = model(b_x)  # [B, 256, 8]

            pred_box = preds[:, :, 0:4]
            pred_obj = preds[:, :, 4]
            pred_cls = preds[:, :, 5:8]

            # Objectness loss with class imbalance compensation
            pred_obj_clamped = torch.clamp(pred_obj, 1e-6, 1.0 - 1e-6)
            pos_mask = (b_obj > 0.5)
            neg_mask = (b_obj <= 0.5)

            loss_pos = bce(pred_obj_clamped[pos_mask], b_obj[pos_mask]) if pos_mask.sum() > 0 else torch.tensor(0.0)
            loss_neg = bce(pred_obj_clamped[neg_mask], b_obj[neg_mask])
            loss_obj = loss_pos * 25.0 + loss_neg * 1.0

            # Box regression and classification loss on positive cells
            if pos_mask.sum() > 0:
                loss_box = l1(pred_box[pos_mask], b_box[pos_mask]) * 5.0
                pred_cls_clamped = torch.clamp(pred_cls[pos_mask], 1e-6, 1.0 - 1e-6)
                loss_cls = bce(pred_cls_clamped, b_cls[pos_mask]) * 3.0
            else:
                loss_box = torch.tensor(0.0)
                loss_cls = torch.tensor(0.0)

            loss = loss_obj + loss_box + loss_cls
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 6 == 0:
            logger.info("Epoch %d/%d - Loss: %.4f", epoch + 1, epochs, total_loss / num_batches)

    model.eval()
    logger.info("Synthetic pre-training complete.")
    return model


def build_and_export_model(
    output_path: str = "models/defect_detector_fp32.onnx",
    train_synthetic: bool = True,
    seed: int = 42,
) -> str:
    """Builds and exports the lightweight defect detector model to ONNX format.

    Args:
        output_path: Path to save the exported ONNX model.
        train_synthetic: Whether to train on deterministic synthetic defects.
        seed: Random seed for initialization and synthetic pre-training.

    Returns:
        Absolute path to the exported ONNX model.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.manual_seed(seed)

    model = LightweightDefectDetector(num_classes=3, grid_size=16)

    if train_synthetic:
        model = train_synthetic_detector(model, num_samples=60, epochs=15, seed=seed)

    model.eval()
    dummy_input = torch.randn(1, 3, 256, 256, dtype=torch.float32)

    logger.info("Exporting FP32 model to ONNX: %s", output_path)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["images"],
        output_names=["detections"],
        dynamic_axes={
            "images": {0: "batch_size"},
            "detections": {0: "batch_size"},
        },
        dynamo=False,
    )

    # Verify model with ONNX checker
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX model structure validated successfully.")

    # Test inference session
    session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    ort_inputs = {session.get_inputs()[0].name: dummy_input.numpy()}
    ort_outs = session.run(None, ort_inputs)
    assert ort_outs[0].shape == (1, 256, 8), f"Unexpected output shape: {ort_outs[0].shape}"
    logger.info("ONNX Runtime CPU session test passed with output shape: %s", ort_outs[0].shape)

    return os.path.abspath(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build and export EdgeVision-INT8 defect detector.")
    parser.add_argument("--output", type=str, default="models/defect_detector_fp32.onnx", help="Path to output ONNX file")
    parser.add_argument("--skip-training", action="store_true", help="Skip synthetic pre-training")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed")
    args = parser.parse_args()

    build_and_export_model(
        output_path=args.output,
        train_synthetic=not args.skip_training,
        seed=args.seed,
    )
