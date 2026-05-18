"""Lightweight spatial-channel gating helpers for focus-region enhancement."""

from __future__ import annotations

import torch
import torch.nn as nn


class KeyRegionGate(nn.Module):
    """Small gate that mixes channel attention and spatial attention."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels, max(channels // reduction, 8), 3, padding=1, bias=False),
            nn.BatchNorm2d(max(channels // reduction, 8)),
            nn.SiLU(),
            nn.Conv2d(max(channels // reduction, 8), 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return multiplicative gating weights."""
        return self.channel_gate(x) * self.spatial_gate(x)
