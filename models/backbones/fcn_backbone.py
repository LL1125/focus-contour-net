"""Minimal FocusContourNet backbone + neck inspired by the YOLO26 design philosophy."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from models.common.blocks import C2f, SCDown, SPPF
from models.common.conv import Conv
from models.common.neck import PathAggregationNeck
from models.plugins.feature_router import FeatureRouter


class FCNBackbone(nn.Module):
    """Backbone-neck wrapper that exposes multi-scale features clearly."""

    def __init__(self, cfg: dict[str, Any], plugin_router: FeatureRouter | None = None) -> None:
        super().__init__()
        channels = cfg.get("backbone_channels", [32, 64, 128, 256, 512])
        neck_channels = cfg.get("neck_channels", [128, 256, 512])
        in_channels = int(cfg.get("in_channels", 3))
        self.enable_p2 = bool(cfg.get("enable_p2", False))
        self.plugin_router = plugin_router or FeatureRouter()

        c1, c2, c3, c4, c5 = channels
        self.stem = Conv(in_channels, c1, 3, 2)
        self.stage2_down = SCDown(c1, c2)
        self.stage2 = C2f(c2, c2, n=2, shortcut=True)
        self.stage3_down = SCDown(c2, c3)
        self.stage3 = C2f(c3, c3, n=2, shortcut=True)
        self.stage4_down = SCDown(c3, c4)
        self.stage4 = C2f(c4, c4, n=3, shortcut=True)
        self.stage5_down = SCDown(c4, c5)
        self.stage5 = nn.Sequential(C2f(c5, c5, n=3, shortcut=True), SPPF(c5, c5))

        neck_in = [c2, c3, c4, c5] if self.enable_p2 else [c3, c4, c5]
        self.neck = PathAggregationNeck(neck_in, neck_channels, fusion_plugin=self._neck_plugin)

    def _backbone_plugin(self, x: torch.Tensor) -> torch.Tensor:
        return self.plugin_router.route("backbone_late", x)

    def _neck_plugin(self, x: torch.Tensor) -> torch.Tensor:
        return self.plugin_router.route("neck_fusion", x)

    def forward(self, x: torch.Tensor) -> dict[str, dict[str, torch.Tensor] | list[torch.Tensor]]:
        """Return backbone features and neck outputs with stable names."""
        x = self.stem(x)
        p2 = self.stage2(self.stage2_down(x))
        p3 = self.stage3(self.stage3_down(p2))
        p4 = self._backbone_plugin(self.stage4(self.stage4_down(p3)))
        p5 = self.stage5(self.stage5_down(p4))

        if self.enable_p2:
            neck_out = self.neck(p2, p3, p4, p5)
            ordered = [neck_out["p2"], neck_out["p3"], neck_out["p4"], neck_out["p5"]]
            backbone_features = {"p2": p2, "p3": p3, "p4": p4, "p5": p5}
        else:
            neck_out = self.neck(p3, p4, p5)
            ordered = [neck_out["p3"], neck_out["p4"], neck_out["p5"]]
            backbone_features = {"p3": p3, "p4": p4, "p5": p5}

        return {"backbone": backbone_features, "neck": neck_out, "features": ordered}
