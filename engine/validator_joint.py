"""Validation loop for joint detection + contour training."""

from __future__ import annotations

import torch

from engine.validator_detect import ValidatorDetect


class ValidatorJoint(ValidatorDetect):
    """Joint validator that aggregates contour-related losses too."""

    def validate(self, model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, criterion: torch.nn.Module) -> dict[str, float]:
        model.eval()
        total_loss = 0.0
        total_iou = 0.0
        total_batches = 0
        total_contour = 0.0
        with torch.no_grad():
            for batch in dataloader:
                images = batch["images"].to(self.device)
                targets = batch["targets"]
                for target in targets:
                    for key in ("labels", "boxes", "contours", "boundary"):
                        if key in target and isinstance(target[key], torch.Tensor):
                            target[key] = target[key].to(self.device)
                outputs = model.forward_train(images)
                loss_dict = criterion(outputs, targets)
                infer = model.forward_infer(images)
                total_loss += float(loss_dict["loss"].item())
                total_contour += float(loss_dict.get("contour_loss", torch.tensor(0.0)).item())
                total_iou += self._mean_best_iou(infer["detections"], targets)
                total_batches += 1
        return {
            "val_loss": total_loss / max(total_batches, 1),
            "val_contour_loss": total_contour / max(total_batches, 1),
            "val_mean_best_iou": total_iou / max(total_batches, 1),
        }
