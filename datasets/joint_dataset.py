"""Joint detection + contour dataset scaffold."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from datasets.detect_dataset import DetectDataset


class JointDataset(DetectDataset):
    """Detection dataset extended with contour and boundary placeholder supervision."""

    def __init__(self, data_cfg: dict[str, Any], split: str = "train", img_size: int | None = None) -> None:
        super().__init__(data_cfg=data_cfg, split=split, img_size=img_size)
        contour_cfg = data_cfg.get("contour", {})
        self.contour_dir = self.root / contour_cfg.get("annotation_dir", "contours")
        self.boundary_dir = self.root / contour_cfg.get("boundary_dir", "boundaries")

    def _load_json_tensor(self, path: Path, key: str, default_shape: tuple[int, ...]) -> torch.Tensor:
        if not path.exists():
            return torch.zeros(default_shape, dtype=torch.float32)
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        data = payload.get(key, [])
        tensor = torch.tensor(data, dtype=torch.float32)
        if tensor.numel() == 0:
            return torch.zeros(default_shape, dtype=torch.float32)
        return tensor

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = super().__getitem__(index)
        image_path = self.image_paths[index]
        relative = image_path.relative_to(self.image_dir).with_suffix(".json")
        contour_path = self.contour_dir / relative
        boundary_path = self.boundary_dir / relative

        target = sample["target"]
        target["contours"] = self._load_json_tensor(contour_path, key="contours", default_shape=(0, 0, 2))
        target["boundary"] = self._load_json_tensor(boundary_path, key="boundary", default_shape=(0,))
        return sample
