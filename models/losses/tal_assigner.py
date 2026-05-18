"""Minimal task-aligned assigner for the standalone detector."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.common.utils import bbox_iou


class TaskAlignedAssigner(nn.Module):
    """Assign anchors to ground truth with a simplified TAL strategy."""

    def __init__(self, topk: int = 10, num_classes: int = 80, alpha: float = 1.0, beta: float = 6.0) -> None:
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta

    @torch.no_grad()
    def forward(
        self,
        pred_scores: torch.Tensor,
        pred_boxes: torch.Tensor,
        anchor_points: torch.Tensor,
        gt_labels: list[torch.Tensor],
        gt_boxes: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Build per-anchor targets for a batch."""
        batch, num_anchors, _ = pred_scores.shape
        device = pred_scores.device
        target_labels = torch.full((batch, num_anchors), -1, dtype=torch.long, device=device)
        target_boxes = torch.zeros((batch, num_anchors, 4), dtype=pred_boxes.dtype, device=device)
        target_scores = torch.zeros((batch, num_anchors, self.num_classes), dtype=pred_scores.dtype, device=device)
        fg_mask = torch.zeros((batch, num_anchors), dtype=torch.bool, device=device)
        matched_gt = torch.full((batch, num_anchors), -1, dtype=torch.long, device=device)

        for b in range(batch):
            if gt_boxes[b].numel() == 0:
                continue
            gtb = gt_boxes[b].to(device)
            gtl = gt_labels[b].to(device)
            in_gt = self._select_candidates_in_gts(anchor_points, gtb)
            if not in_gt.any():
                continue
            ious = self._pairwise_iou(pred_boxes[b], gtb)
            cls_scores = pred_scores[b].sigmoid()[:, gtl]
            metrics = cls_scores.pow(self.alpha) * ious.pow(self.beta)
            metrics = metrics * in_gt.float()

            pos_mask = torch.zeros_like(metrics, dtype=torch.bool)
            topk = min(self.topk, num_anchors)
            for gt_idx in range(gtb.shape[0]):
                values, indices = torch.topk(metrics[:, gt_idx], k=topk, largest=True)
                valid = values > 0
                pos_mask[indices[valid], gt_idx] = True

            if not pos_mask.any():
                continue

            overlaps = ious.masked_fill(~pos_mask, -1.0)
            best_gt = overlaps.argmax(dim=1)
            best_iou = overlaps.max(dim=1).values
            fg = best_iou > 0
            fg_mask[b] = fg
            matched_gt[b, fg] = best_gt[fg]
            target_labels[b, fg] = gtl[best_gt[fg]]
            target_boxes[b, fg] = gtb[best_gt[fg]]
            target_scores[b, fg, gtl[best_gt[fg]]] = best_iou[fg].clamp_(0.0, 1.0)

        return {
            "target_labels": target_labels,
            "target_boxes": target_boxes,
            "target_scores": target_scores,
            "fg_mask": fg_mask,
            "matched_gt_indices": matched_gt,
        }

    def _select_candidates_in_gts(self, anchors: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
        x = anchors[:, 0:1]
        y = anchors[:, 1:2]
        left = x >= gt_boxes[:, 0]
        right = x <= gt_boxes[:, 2]
        top = y >= gt_boxes[:, 1]
        bottom = y <= gt_boxes[:, 3]
        return left & right & top & bottom

    def _pairwise_iou(self, pred_boxes: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
        pred = pred_boxes.unsqueeze(1).expand(-1, gt_boxes.shape[0], -1)
        gt = gt_boxes.unsqueeze(0).expand(pred_boxes.shape[0], -1, -1)
        return bbox_iou(pred, gt)
