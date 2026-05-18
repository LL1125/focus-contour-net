"""Minimal dual-branch detection loss for the standalone YOLO26-style detector."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common.utils import generalized_box_iou
from models.losses.tal_assigner import TaskAlignedAssigner


class DetectLossE2E(nn.Module):
    """Compute one-to-many and one-to-one detection losses separately."""

    def __init__(self, num_classes: int, loss_cfg: dict[str, Any] | None = None) -> None:
        super().__init__()
        cfg = loss_cfg or {}
        self.num_classes = num_classes
        self.cls_weight = float(cfg.get("cls_weight", 1.0))
        self.box_weight = float(cfg.get("box_weight", 5.0))
        self.iou_weight = float(cfg.get("iou_weight", 2.0))
        self.one2many_weight = float(cfg.get("one2many_weight", 1.0))
        self.one2one_weight = float(cfg.get("one2one_weight", 1.0))
        self.one2many_assigner = TaskAlignedAssigner(topk=int(cfg.get("one2many_topk", 10)), num_classes=num_classes)
        self.one2one_assigner = TaskAlignedAssigner(topk=int(cfg.get("one2one_topk", 1)), num_classes=num_classes)

    def _split_targets(self, targets: list[dict[str, torch.Tensor]]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        labels = [target["labels"].long() for target in targets]
        boxes = [target["boxes"].float() for target in targets]
        return labels, boxes

    def _compute_branch_loss(
        self,
        preds: dict[str, torch.Tensor],
        gt_labels: list[torch.Tensor],
        gt_boxes: list[torch.Tensor],
        assigner: TaskAlignedAssigner,
        branch_name: str,
        branch_weight: float,
    ) -> dict[str, torch.Tensor]:
        assignment = assigner(preds["pred_scores"], preds["pred_boxes"], preds["anchors"], gt_labels, gt_boxes)
        target_scores = assignment["target_scores"]
        fg_mask = assignment["fg_mask"]
        cls_loss = F.binary_cross_entropy_with_logits(preds["pred_scores"], target_scores, reduction="mean")

        if fg_mask.any():
            pred_boxes = preds["pred_boxes"][fg_mask]
            target_boxes = assignment["target_boxes"][fg_mask]
            box_loss = F.l1_loss(pred_boxes, target_boxes, reduction="mean")
            iou_loss = (1.0 - generalized_box_iou(pred_boxes, target_boxes)).mean()
        else:
            zero = preds["pred_scores"].new_tensor(0.0)
            box_loss = zero
            iou_loss = zero

        total = branch_weight * (self.cls_weight * cls_loss + self.box_weight * box_loss + self.iou_weight * iou_loss)
        return {
            f"{branch_name}_cls": cls_loss.detach(),
            f"{branch_name}_box": box_loss.detach(),
            f"{branch_name}_iou": iou_loss.detach(),
            f"{branch_name}_loss": total,
        }

    def forward(self, preds: dict[str, dict[str, torch.Tensor]], targets: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Return branch-wise and total detection loss terms."""
        gt_labels, gt_boxes = self._split_targets(targets)
        one2many = self._compute_branch_loss(
            preds["one2many"], gt_labels, gt_boxes, self.one2many_assigner, "one2many", self.one2many_weight
        )
        one2one = self._compute_branch_loss(
            preds["one2one"], gt_labels, gt_boxes, self.one2one_assigner, "one2one", self.one2one_weight
        )
        total = one2many["one2many_loss"] + one2one["one2one_loss"]
        return {**one2many, **one2one, "loss": total}
