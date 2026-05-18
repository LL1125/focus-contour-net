"""Minimal validation loop for detect-only training."""

from __future__ import annotations

from typing import Any

import torch
from tqdm import tqdm

from models.common.utils import bbox_iou


class ValidatorDetect:
    """Run validation for the detect-only baseline."""

    def __init__(self, device: torch.device) -> None:
        self.device = device

    def _mean_best_iou(self, detections: torch.Tensor, targets: list[dict[str, torch.Tensor]]) -> float:
        scores = []
        for det, target in zip(detections, targets):
            gt_boxes = target["boxes"].to(det.device)
            if gt_boxes.numel() == 0 or det.numel() == 0:
                continue
            ious = bbox_iou(det[:, None, :4], gt_boxes[None, :, :])
            scores.append(float(ious.max(dim=0).values.mean().item()))
        return sum(scores) / max(len(scores), 1)

    def validate(self, model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, criterion: torch.nn.Module) -> dict[str, float]:
        """Evaluate loss and a lightweight IoU proxy metric."""
        model.eval()
        total_loss = 0.0
        total_batches = 0
        total_iou = 0.0
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="val", leave=False):
                images = batch["images"].to(self.device)
                targets = batch["targets"]
                for target in targets:
                    target["labels"] = target["labels"].to(self.device)
                    target["boxes"] = target["boxes"].to(self.device)
                outputs = model.forward_train(images)
                loss_dict = criterion(outputs["detection_outputs"], targets)
                infer = model.forward_infer(images)
                total_loss += float(loss_dict["loss"].item())
                total_iou += self._mean_best_iou(infer["detections"], targets)
                total_batches += 1
        return {
            "val_loss": total_loss / max(total_batches, 1),
            "val_mean_best_iou": total_iou / max(total_batches, 1),
        }
