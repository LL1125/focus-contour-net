"""Standalone detect-only FocusContourNet base detector."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from models.backbones.fcn_backbone import FCNBackbone
from models.heads.detect_head_e2e import DetectHeadE2E


class FCNBaseDetector(nn.Module):
    """Baseline detect-only FocusContourNet detector."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = FCNBackbone(cfg)
        self.detect_head = DetectHeadE2E(
            num_classes=int(cfg["num_classes"]),
            ch=cfg["neck_channels"],
            strides=cfg["strides"],
            hidden_dim=int(cfg.get("head_channels", 128)),
            max_det=int(cfg.get("max_det", 300)),
        )

    def forward_train(self, images: torch.Tensor) -> dict[str, Any]:
        """Forward path used during optimization."""
        features = self.backbone(images)
        det_outputs = self.detect_head.forward_train(features["features"])
        return {"feature_pyramid": features, "detection_outputs": det_outputs}

    def forward_infer(self, images: torch.Tensor, return_raw: bool = False) -> dict[str, Any]:
        """Inference path that defaults to one-to-one end-to-end detections."""
        features = self.backbone(images)
        det_outputs = self.detect_head.forward_infer(features["features"])
        if return_raw:
            return {"feature_pyramid": features, **det_outputs}
        return {"detections": det_outputs["detections"]}

    def forward(self, images: torch.Tensor, return_raw: bool = False) -> dict[str, Any]:
        """Dispatch by module mode."""
        if self.training:
            return self.forward_train(images)
        return self.forward_infer(images, return_raw=return_raw)
