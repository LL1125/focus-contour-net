"""Utility helpers for detection head decoding and end-to-end top-k selection."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from models.common.utils import dist2bbox, make_anchors


def flatten_map(x: torch.Tensor) -> torch.Tensor:
    """Convert a BCHW tensor to BAC format, where A is the flattened spatial axis."""
    b, c, h, w = x.shape
    return x.view(b, c, h * w).permute(0, 2, 1).contiguous()


def decode_distance_maps(
    box_maps: list[torch.Tensor],
    features: list[torch.Tensor],
    strides: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode per-level distance maps into flattened boxes and anchor metadata."""
    anchors, stride_tensor = make_anchors(features, strides)
    flat_dists = []
    for box_map, stride in zip(box_maps, strides):
        dist = F.softplus(flatten_map(box_map)) * float(stride)
        flat_dists.append(dist)
    distances = torch.cat(flat_dists, dim=1)
    pred_boxes = dist2bbox(distances, anchors.unsqueeze(0))
    return pred_boxes, anchors, stride_tensor


def select_topk_detections(pred_boxes: torch.Tensor, pred_scores: torch.Tensor, max_det: int = 300) -> torch.Tensor:
    """Select top scoring end-to-end detections without NMS."""
    batch, num_anchors, num_classes = pred_scores.shape
    scores = pred_scores.sigmoid()
    k = min(max_det, num_anchors * num_classes)
    flat_scores = scores.reshape(batch, -1)
    conf, indices = torch.topk(flat_scores, k=k, dim=1)
    anchor_idx = torch.div(indices, num_classes, rounding_mode="floor")
    class_idx = indices % num_classes
    gather_idx = anchor_idx.unsqueeze(-1).expand(-1, -1, 4)
    boxes = torch.gather(pred_boxes, 1, gather_idx)
    return torch.cat((boxes, conf.unsqueeze(-1), class_idx.float().unsqueeze(-1)), dim=-1)
