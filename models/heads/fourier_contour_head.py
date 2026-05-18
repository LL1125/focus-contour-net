"""Interface shell for a future formal Fourier base + local normal residual contour head."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.common.conv import Conv


class FourierContourHead(nn.Module):
    """Runnable placeholder contour branch sharing neck features with detection."""

    def __init__(
        self,
        in_channels: list[int],
        hidden_dim: int = 128,
        num_coeffs: int = 16,
        num_points: int = 64,
        boundary_head: bool = True,
    ) -> None:
        super().__init__()
        self.num_coeffs = num_coeffs
        self.num_points = num_points
        self.boundary_head_enabled = boundary_head

        self.stems = nn.ModuleList(nn.Sequential(Conv(c, hidden_dim, 3, 1), Conv(hidden_dim, hidden_dim, 3, 1)) for c in in_channels)
        self.fourier_heads = nn.ModuleList(nn.Conv2d(hidden_dim, num_coeffs * 2, 1) for _ in in_channels)
        self.contour_heads = nn.ModuleList(nn.Conv2d(hidden_dim, num_points * 2, 1) for _ in in_channels)
        self.boundary_heads = (
            nn.ModuleList(nn.Conv2d(hidden_dim, 1, 1) for _ in in_channels) if boundary_head else None
        )

    def _flatten(self, x: torch.Tensor, channels: int) -> torch.Tensor:
        batch = x.shape[0]
        return x.view(batch, channels, -1).permute(0, 2, 1).contiguous()

    def forward(self, features: list[torch.Tensor]) -> dict[str, torch.Tensor | None]:
        """Return placeholder contour predictions with future-proof field names."""
        base_fourier = []
        refined_contour = []
        boundary_logits = []
        for idx, feat in enumerate(features):
            stem = self.stems[idx](feat)
            coeffs = self._flatten(self.fourier_heads[idx](stem), self.num_coeffs * 2).view(feat.shape[0], -1, self.num_coeffs, 2)
            contour = self._flatten(self.contour_heads[idx](stem), self.num_points * 2).view(feat.shape[0], -1, self.num_points, 2)
            base_fourier.append(coeffs)
            refined_contour.append(contour)
            if self.boundary_heads is not None:
                boundary_logits.append(self._flatten(self.boundary_heads[idx](stem), 1))

        return {
            "base_fourier": torch.cat(base_fourier, dim=1),
            "refined_contour": torch.cat(refined_contour, dim=1),
            "boundary_logits": torch.cat(boundary_logits, dim=1) if boundary_logits else None,
        }
