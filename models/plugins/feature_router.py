"""Utility wrapper for routing optional plugins into named feature stages."""

from __future__ import annotations

import torch
import torch.nn as nn


class FeatureRouter(nn.Module):
    """Apply named plugin modules to selected features if they are available."""

    def __init__(self, modules: dict[str, nn.Module] | None = None) -> None:
        super().__init__()
        self.modules_map = nn.ModuleDict(modules or {})

    def has(self, name: str) -> bool:
        """Check whether a plugin exists for the given route."""
        return name in self.modules_map

    def route(self, name: str, x: torch.Tensor) -> torch.Tensor:
        """Apply a named plugin if present, otherwise return the input unchanged."""
        if name not in self.modules_map:
            return x
        return self.modules_map[name](x)
