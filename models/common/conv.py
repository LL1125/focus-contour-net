"""Minimal convolution blocks used by the standalone YOLO26-style project."""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn


def autopad(k: int | Iterable[int], p: int | None = None, d: int = 1) -> int | list[int]:
    """Return padding that preserves spatial shape for the given kernel and dilation."""
    if isinstance(k, Iterable):
        kernel = [d * (x - 1) + 1 if d > 1 else x for x in k]
        return [x // 2 for x in kernel] if p is None else p
    kernel = d * (k - 1) + 1 if d > 1 else k
    return kernel // 2 if p is None else p


class Conv(nn.Module):
    """Convolution + BatchNorm + activation."""

    default_act = nn.SiLU()

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 1,
        s: int = 1,
        p: int | None = None,
        g: int = 1,
        d: int = 1,
        act: bool | nn.Module = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution, normalization, and activation."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        """Apply fused convolution for deployment-style inference."""
        return self.act(self.conv(x))


class DWConv(Conv):
    """Depth-wise style convolution with automatically chosen group count."""

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, d: int = 1, act: bool | nn.Module = True) -> None:
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class ConvTranspose(nn.Module):
    """ConvTranspose2d with optional normalization and activation."""

    def __init__(self, c1: int, c2: int, k: int = 2, s: int = 2, act: bool | nn.Module = True) -> None:
        super().__init__()
        self.deconv = nn.ConvTranspose2d(c1, c2, kernel_size=k, stride=s, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = Conv.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample and refine the feature map."""
        return self.act(self.bn(self.deconv(x)))


class RepConv(nn.Module):
    """A lightweight re-parameterizable conv block."""

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1, g: int = 1, act: bool | nn.Module = True) -> None:
        super().__init__()
        if k != 3:
            raise ValueError("RepConv currently supports k=3 only in this minimal project.")
        self.c1 = c1
        self.c2 = c2
        self.g = g
        self.s = s
        self.act = Conv.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.branch_3x3 = Conv(c1, c2, 3, s, g=g, act=False)
        self.branch_1x1 = Conv(c1, c2, 1, s, g=g, act=False)
        self.use_identity = c1 == c2 and s == 1
        self.id_bn = nn.BatchNorm2d(c1) if self.use_identity else None
        self.deploy = False
        self.reparam = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Combine 3x3, 1x1, and optional identity branches."""
        if self.deploy and self.reparam is not None:
            return self.act(self.reparam(x))
        y = self.branch_3x3(x) + self.branch_1x1(x)
        if self.id_bn is not None:
            y = y + self.id_bn(x)
        return self.act(y)

    def fuse_for_infer(self) -> None:
        """Switch to a single convolutional branch for inference."""
        if self.deploy:
            return
        self.reparam = nn.Conv2d(self.c1, self.c2, 3, self.s, 1, groups=self.g, bias=True)
        nn.init.zeros_(self.reparam.weight)
        nn.init.zeros_(self.reparam.bias)
        self.deploy = True


class Concat(nn.Module):
    """Concatenate a list of tensors along a fixed dimension."""

    def __init__(self, dimension: int = 1) -> None:
        super().__init__()
        self.dimension = dimension

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        """Concatenate tensors."""
        return torch.cat(xs, dim=self.dimension)
