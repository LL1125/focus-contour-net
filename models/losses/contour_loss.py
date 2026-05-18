"""Placeholder contour loss for future Fourier / boundary / residual objectives."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContourLoss(nn.Module):
    """Executable placeholder for future formal contour supervision."""

    def __init__(self, loss_cfg: dict[str, Any] | None = None) -> None:
        super().__init__()
        cfg = loss_cfg or {}
        self.contour_weight = float(cfg.get("contour_weight", 1.0))
        self.boundary_weight = float(cfg.get("boundary_weight", 0.25))

    def forward(self, preds: dict[str, torch.Tensor | None], targets: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Return placeholder contour losses.

        TODO:
        Replace this with the formal Fourier coefficient, boundary, and local residual contour objectives.
        """
        device = preds["base_fourier"].device
        contour_targets = [target.get("contours", torch.empty(0, device=device)) for target in targets]
        boundary_targets = [target.get("boundary", torch.empty(0, device=device)) for target in targets]

        contour_loss = torch.tensor(0.0, device=device)
        boundary_loss = torch.tensor(0.0, device=device)
        valid_contours = 0
        valid_boundaries = 0

        for batch_index, contour_gt in enumerate(contour_targets):
            if contour_gt.numel() == 0:
                continue
            pred = preds["refined_contour"][batch_index, 0]
            gt = contour_gt[0].to(device)
            if gt.shape[0] != pred.shape[0]:
                if gt.shape[0] > pred.shape[0]:
                    gt = gt[: pred.shape[0]]
                else:
                    pad = pred.new_zeros((pred.shape[0] - gt.shape[0], gt.shape[1]))
                    gt = torch.cat([gt, pad], dim=0)
            contour_loss = contour_loss + F.l1_loss(pred, gt, reduction="mean")
            valid_contours += 1

        if preds.get("boundary_logits") is not None:
            for batch_index, boundary_gt in enumerate(boundary_targets):
                if boundary_gt.numel() == 0:
                    continue
                pred = preds["boundary_logits"][batch_index, :1].reshape(-1)
                gt = boundary_gt[:1].to(device).reshape(-1).float()
                boundary_loss = boundary_loss + F.binary_cross_entropy_with_logits(pred, gt)
                valid_boundaries += 1

        if valid_contours:
            contour_loss = contour_loss / valid_contours
        if valid_boundaries:
            boundary_loss = boundary_loss / valid_boundaries

        total = self.contour_weight * contour_loss + self.boundary_weight * boundary_loss
        return {
            "contour_loss": contour_loss.detach(),
            "boundary_loss": boundary_loss.detach(),
            "loss": total,
        }
