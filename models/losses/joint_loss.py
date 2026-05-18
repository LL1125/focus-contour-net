"""Composite loss for detect-only and detect+contour training."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from models.losses.contour_loss import ContourLoss
from models.losses.detect_loss_e2e import DetectLossE2E


class JointLoss(nn.Module):
    """Combine detection loss with optional contour loss."""

    def __init__(self, num_classes: int, loss_cfg: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.detect_loss = DetectLossE2E(num_classes=num_classes, loss_cfg=loss_cfg)
        self.contour_loss = ContourLoss(loss_cfg=loss_cfg)

    def forward(self, outputs: dict[str, Any], targets: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Compute detection loss and optionally contour loss."""
        detect_terms = self.detect_loss(outputs["detection_outputs"], targets)
        if "contour_outputs" not in outputs or outputs["contour_outputs"] is None:
            return detect_terms
        contour_terms = self.contour_loss(outputs["contour_outputs"], targets)
        total = detect_terms["loss"] + contour_terms["loss"]
        return {**detect_terms, **contour_terms, "loss": total}
