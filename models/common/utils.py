"""Shared utility functions for geometry, YAML loading, and reproducibility."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dictionary."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def make_divisible(x: float, divisor: int = 8) -> int:
    """Round channel count to a hardware-friendly divisor."""
    return int(np.ceil(x / divisor) * divisor)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def xywhn_to_xyxy(boxes: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Convert normalized YOLO `xywh` boxes to absolute `xyxy` format."""
    if boxes.numel() == 0:
        return boxes.reshape(0, 4)
    cx, cy, bw, bh = boxes.unbind(dim=-1)
    x1 = (cx - bw / 2.0) * width
    y1 = (cy - bh / 2.0) * height
    x2 = (cx + bw / 2.0) * width
    y2 = (cy + bh / 2.0) * height
    return torch.stack((x1, y1, x2, y2), dim=-1)


def xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    """Convert `xyxy` boxes to `xywh`."""
    if boxes.numel() == 0:
        return boxes.reshape(0, 4)
    x1, y1, x2, y2 = boxes.unbind(dim=-1)
    return torch.stack(((x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1), dim=-1)


def bbox_iou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Compute pairwise IoU for aligned `xyxy` boxes."""
    x1 = torch.maximum(box1[..., 0], box2[..., 0])
    y1 = torch.maximum(box1[..., 1], box2[..., 1])
    x2 = torch.minimum(box1[..., 2], box2[..., 2])
    y2 = torch.minimum(box1[..., 3], box2[..., 3])
    inter = (x2 - x1).clamp_min(0) * (y2 - y1).clamp_min(0)
    area1 = (box1[..., 2] - box1[..., 0]).clamp_min(0) * (box1[..., 3] - box1[..., 1]).clamp_min(0)
    area2 = (box2[..., 2] - box2[..., 0]).clamp_min(0) * (box2[..., 3] - box2[..., 1]).clamp_min(0)
    return inter / (area1 + area2 - inter + eps)


def generalized_box_iou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Compute aligned Generalized IoU for `xyxy` boxes."""
    iou = bbox_iou(box1, box2, eps=eps)
    cx1 = torch.minimum(box1[..., 0], box2[..., 0])
    cy1 = torch.minimum(box1[..., 1], box2[..., 1])
    cx2 = torch.maximum(box1[..., 2], box2[..., 2])
    cy2 = torch.maximum(box1[..., 3], box2[..., 3])
    c_area = (cx2 - cx1).clamp_min(0) * (cy2 - cy1).clamp_min(0) + eps
    area1 = (box1[..., 2] - box1[..., 0]).clamp_min(0) * (box1[..., 3] - box1[..., 1]).clamp_min(0)
    area2 = (box2[..., 2] - box2[..., 0]).clamp_min(0) * (box2[..., 3] - box2[..., 1]).clamp_min(0)
    x1 = torch.maximum(box1[..., 0], box2[..., 0])
    y1 = torch.maximum(box1[..., 1], box2[..., 1])
    x2 = torch.minimum(box1[..., 2], box2[..., 2])
    y2 = torch.minimum(box1[..., 3], box2[..., 3])
    inter = (x2 - x1).clamp_min(0) * (y2 - y1).clamp_min(0)
    union = area1 + area2 - inter + eps
    return iou - (c_area - union) / c_area


def dist2bbox(distance: torch.Tensor, anchor_points: torch.Tensor) -> torch.Tensor:
    """Decode left-top-right-bottom distances into `xyxy` boxes."""
    left_top = anchor_points - distance[..., :2]
    right_bottom = anchor_points + distance[..., 2:]
    return torch.cat((left_top, right_bottom), dim=-1)


def make_anchors(features: list[torch.Tensor], strides: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Create anchor-point centers and per-anchor stride values from feature maps."""
    anchors = []
    stride_tensors = []
    device = features[0].device
    dtype = features[0].dtype
    for feat, stride in zip(features, strides):
        _, _, h, w = feat.shape
        ys, xs = torch.meshgrid(
            torch.arange(h, device=device, dtype=dtype),
            torch.arange(w, device=device, dtype=dtype),
            indexing="ij",
        )
        anchor = torch.stack(((xs + 0.5) * stride, (ys + 0.5) * stride), dim=-1).reshape(-1, 2)
        anchors.append(anchor)
        stride_tensors.append(torch.full((h * w, 1), float(stride), device=device, dtype=dtype))
    return torch.cat(anchors, dim=0), torch.cat(stride_tensors, dim=0)
