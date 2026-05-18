"""Minimal structural blocks for the standalone YOLO26-style detector."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.common.conv import Conv, DWConv


class Bottleneck(nn.Module):
    """Standard bottleneck block with optional shortcut."""

    def __init__(self, c1: int, c2: int, shortcut: bool = True, e: float = 0.5) -> None:
        super().__init__()
        hidden = int(c2 * e)
        self.cv1 = Conv(c1, hidden, 1, 1)
        self.cv2 = Conv(hidden, c2, 3, 1)
        self.use_shortcut = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the bottleneck block."""
        y = self.cv2(self.cv1(x))
        return x + y if self.use_shortcut else y


class C2f(nn.Module):
    """Cross-stage partial block with lightweight feature accumulation."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, e: float = 0.5) -> None:
        super().__init__()
        hidden = int(c2 * e)
        self.cv1 = Conv(c1, 2 * hidden, 1, 1)
        self.blocks = nn.ModuleList(Bottleneck(hidden, hidden, shortcut=shortcut, e=1.0) for _ in range(n))
        self.cv2 = Conv((2 + n) * hidden, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Split, transform, and fuse partial features."""
        y = list(self.cv1(x).chunk(2, dim=1))
        for block in self.blocks:
            y.append(block(y[-1]))
        return self.cv2(torch.cat(y, dim=1))


class C3(nn.Module):
    """Classic CSP-style C3 block."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, e: float = 0.5) -> None:
        super().__init__()
        hidden = int(c2 * e)
        self.cv1 = Conv(c1, hidden, 1, 1)
        self.cv2 = Conv(c1, hidden, 1, 1)
        self.m = nn.Sequential(*(Bottleneck(hidden, hidden, shortcut=shortcut, e=1.0) for _ in range(n)))
        self.cv3 = Conv(hidden * 2, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fuse transformed and skip branches."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class SPPF(nn.Module):
    """Fast spatial pyramid pooling block."""

    def __init__(self, c1: int, c2: int, k: int = 5) -> None:
        super().__init__()
        hidden = c1 // 2
        self.cv1 = Conv(c1, hidden, 1, 1)
        self.pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv2 = Conv(hidden * 4, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply repeated pyramid pooling."""
        y1 = self.cv1(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        y4 = self.pool(y3)
        return self.cv2(torch.cat((y1, y2, y3, y4), dim=1))


class SCDown(nn.Module):
    """Separable convolution downsampling block."""

    def __init__(self, c1: int, c2: int) -> None:
        super().__init__()
        hidden = max(c1, c2 // 2)
        self.cv1 = Conv(c1, hidden, 1, 1)
        self.cv2 = DWConv(hidden, c2, 3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Downsample while keeping computation modest."""
        return self.cv2(self.cv1(x))
