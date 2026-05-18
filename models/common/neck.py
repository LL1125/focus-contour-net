"""Neck definitions for the standalone YOLO26-style detector."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn

from models.common.blocks import C3
from models.common.conv import Concat, Conv


class PathAggregationNeck(nn.Module):
    """Minimal FPN/PAN neck with optional plugin insertion points."""

    def __init__(
        self,
        in_channels: list[int],
        out_channels: list[int],
        fusion_plugin: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        if len(in_channels) not in {3, 4} or len(out_channels) != len(in_channels):
            raise ValueError("This minimal neck expects matching 3-scale or 4-scale channel definitions.")
        self.enable_p2 = len(in_channels) == 4
        self.concat = Concat(1)
        self.fusion_plugin = fusion_plugin

        if self.enable_p2:
            c2, c3, c4, c5 = in_channels
            o2, o3, o4, o5 = out_channels
            self.p5_reduce = Conv(c5, o4, 1, 1)
            self.p4_fuse = C3(c4 + o4, o4, n=2, shortcut=False)
            self.p4_reduce = Conv(o4, o3, 1, 1)
            self.p3_fuse = C3(c3 + o3, o3, n=2, shortcut=False)
            self.p3_reduce = Conv(o3, o2, 1, 1)
            self.p2_fuse = C3(c2 + o2, o2, n=2, shortcut=False)
            self.p2_down = Conv(o2, o2, 3, 2)
            self.n3_fuse = C3(o2 + o3, o3, n=2, shortcut=False)
            self.n3_down = Conv(o3, o3, 3, 2)
            self.n4_fuse = C3(o3 + o4, o4, n=2, shortcut=False)
            self.n4_down = Conv(o4, o4, 3, 2)
            self.n5_fuse = C3(o4 + c5, o5, n=2, shortcut=False)
        else:
            c3, c4, c5 = in_channels
            o3, o4, o5 = out_channels
            self.p5_reduce = Conv(c5, o4, 1, 1)
            self.p4_fuse = C3(c4 + o4, o4, n=2, shortcut=False)
            self.p4_reduce = Conv(o4, o3, 1, 1)
            self.p3_fuse = C3(c3 + o3, o3, n=2, shortcut=False)
            self.p3_down = Conv(o3, o3, 3, 2)
            self.n4_fuse = C3(o3 + o4, o4, n=2, shortcut=False)
            self.n4_down = Conv(o4, o4, 3, 2)
            self.n5_fuse = C3(o4 + c5, o5, n=2, shortcut=False)

    def forward(self, *features: torch.Tensor) -> dict[str, torch.Tensor]:
        """Produce P3/P4/P5 neck outputs."""
        if self.enable_p2:
            p2, p3, p4, p5 = features
            p5_up = torch.nn.functional.interpolate(self.p5_reduce(p5), scale_factor=2.0, mode="nearest")
            p4_td = self.p4_fuse(self.concat([p5_up, p4]))
            if self.fusion_plugin is not None:
                p4_td = self.fusion_plugin(p4_td)

            p4_up = torch.nn.functional.interpolate(self.p4_reduce(p4_td), scale_factor=2.0, mode="nearest")
            p3_td = self.p3_fuse(self.concat([p4_up, p3]))

            p3_up = torch.nn.functional.interpolate(self.p3_reduce(p3_td), scale_factor=2.0, mode="nearest")
            p2_out = self.p2_fuse(self.concat([p3_up, p2]))

            p3_out = self.n3_fuse(self.concat([self.p2_down(p2_out), p3_td]))
            p4_out = self.n4_fuse(self.concat([self.n3_down(p3_out), p4_td]))
            p5_out = self.n5_fuse(self.concat([self.n4_down(p4_out), p5]))
            return {"p2": p2_out, "p3": p3_out, "p4": p4_out, "p5": p5_out}

        p3, p4, p5 = features
        p5_up = torch.nn.functional.interpolate(self.p5_reduce(p5), scale_factor=2.0, mode="nearest")
        p4_td = self.p4_fuse(self.concat([p5_up, p4]))
        if self.fusion_plugin is not None:
            p4_td = self.fusion_plugin(p4_td)

        p4_up = torch.nn.functional.interpolate(self.p4_reduce(p4_td), scale_factor=2.0, mode="nearest")
        p3_out = self.p3_fuse(self.concat([p4_up, p3]))

        p4_out = self.n4_fuse(self.concat([self.p3_down(p3_out), p4_td]))
        p5_out = self.n5_fuse(self.concat([self.n4_down(p4_out), p5]))
        return {"p3": p3_out, "p4": p4_out, "p5": p5_out}
