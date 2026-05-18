"""Minimal YOLO26-style end-to-end detect head with one-to-many and one-to-one branches."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from models.common.conv import Conv, DWConv
from models.heads.head_utils import decode_distance_maps, flatten_map, select_topk_detections


class DetectHeadE2E(nn.Module):
    """Dual-branch detection head.

    Training:
    - one-to-many branch
    - one-to-one branch

    Inference:
    - defaults to one-to-one branch
    """

    def __init__(self, num_classes: int, ch: list[int], strides: list[int], hidden_dim: int = 128, max_det: int = 300) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.channels = ch
        self.strides = strides
        self.max_det = max_det
        self.deploy = False

        self.one2many_box_heads, self.one2many_cls_heads = self._make_heads(ch, hidden_dim)
        self.one2one_box_heads = copy.deepcopy(self.one2many_box_heads)
        self.one2one_cls_heads = copy.deepcopy(self.one2many_cls_heads)

    def _make_heads(self, channels: list[int], hidden_dim: int) -> tuple[nn.ModuleList, nn.ModuleList]:
        box_heads = nn.ModuleList()
        cls_heads = nn.ModuleList()
        for c in channels:
            box_heads.append(nn.Sequential(Conv(c, hidden_dim, 3, 1), Conv(hidden_dim, hidden_dim, 3, 1), nn.Conv2d(hidden_dim, 4, 1)))
            cls_heads.append(
                nn.Sequential(
                    DWConv(c, c, 3, 1),
                    Conv(c, hidden_dim, 1, 1),
                    DWConv(hidden_dim, hidden_dim, 3, 1),
                    Conv(hidden_dim, hidden_dim, 1, 1),
                    nn.Conv2d(hidden_dim, self.num_classes, 1),
                )
            )
        return box_heads, cls_heads

    def _forward_branch(
        self,
        features: list[torch.Tensor],
        box_heads: nn.ModuleList,
        cls_heads: nn.ModuleList,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        box_maps = [head(feat) for head, feat in zip(box_heads, features)]
        cls_maps = [head(feat) for head, feat in zip(cls_heads, features)]
        pred_boxes, anchors, stride_tensor = decode_distance_maps(box_maps, features, self.strides)
        pred_scores = torch.cat([flatten_map(m) for m in cls_maps], dim=1)
        return {
            "box_maps": box_maps,
            "cls_maps": cls_maps,
            "pred_boxes": pred_boxes,
            "pred_scores": pred_scores,
            "anchors": anchors,
            "stride_tensor": stride_tensor,
        }

    def forward_train(self, features: list[torch.Tensor]) -> dict[str, dict[str, torch.Tensor | list[torch.Tensor]]]:
        """Return both training branches."""
        one2many = self._forward_branch(features, self.one2many_box_heads, self.one2many_cls_heads)
        detached = [feat.detach() for feat in features]
        one2one = self._forward_branch(detached, self.one2one_box_heads, self.one2one_cls_heads)
        return {"one2many": one2many, "one2one": one2one}

    def forward_infer(self, features: list[torch.Tensor], branch: str = "one2one") -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Run inference with the selected branch and return NMS-free detections."""
        if branch == "one2many":
            outputs = self._forward_branch(features, self.one2many_box_heads, self.one2many_cls_heads)
        else:
            outputs = self._forward_branch(features, self.one2one_box_heads, self.one2one_cls_heads)
        detections = select_topk_detections(outputs["pred_boxes"], outputs["pred_scores"], max_det=self.max_det)
        return {"detections": detections, "branch_outputs": outputs}

    def fuse_for_infer(self) -> None:
        """Mark the head as deployment-oriented and conceptually drop one-to-many usage."""
        self.deploy = True

    def forward(self, features: list[torch.Tensor]) -> dict[str, torch.Tensor] | dict[str, dict[str, torch.Tensor]]:
        """Dispatch to training or inference path based on module mode."""
        if self.training:
            return self.forward_train(features)
        return self.forward_infer(features)
