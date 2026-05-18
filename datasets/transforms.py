"""Dataset transforms used by the standalone training loops."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image


class ResizeToTensor:
    """Resize to a square canvas and convert to a float tensor in `[0, 1]`."""

    def __init__(self, img_size: int) -> None:
        self.img_size = int(img_size)

    def __call__(self, image: Image.Image) -> tuple[torch.Tensor, dict[str, Any]]:
        original_size = image.size
        image = image.resize((self.img_size, self.img_size), resample=Image.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        if array.ndim == 2:
            array = np.repeat(array[..., None], repeats=3, axis=-1)
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        meta = {"original_size": original_size, "resized_size": (self.img_size, self.img_size)}
        return tensor, meta
