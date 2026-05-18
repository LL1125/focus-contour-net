"""YOLO-style detection dataset with pathlib-based path handling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset

from datasets.transforms import ResizeToTensor
from models.common.utils import xywhn_to_xyxy


class DetectDataset(Dataset):
    """Load standard YOLO detection data from `images/` and `labels/` folders."""

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    def __init__(self, data_cfg: dict[str, Any], split: str = "train", img_size: int | None = None) -> None:
        self.data_cfg = data_cfg
        self.split = split
        self.root = Path(data_cfg["path"])
        self.image_dir = self.root / data_cfg[split]
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory does not exist: {self.image_dir}")
        self.label_dir = self.root / str(data_cfg[split]).replace("images", "labels", 1)
        self.img_size = int(img_size or data_cfg.get("img_size", 640))
        self.transform = ResizeToTensor(self.img_size)
        self.image_paths = sorted(path for path in self.image_dir.rglob("*") if path.suffix.lower() in self.IMG_EXTS)
        if not self.image_paths:
            raise FileNotFoundError(f"No images found under: {self.image_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def _label_path(self, image_path: Path) -> Path:
        relative = image_path.relative_to(self.image_dir).with_suffix(".txt")
        return self.label_dir / relative

    def _read_labels(self, image_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
        label_path = self._label_path(image_path)
        if not label_path.exists():
            return torch.zeros(0, dtype=torch.long), torch.zeros((0, 4), dtype=torch.float32)

        labels = []
        boxes = []
        with label_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(float(parts[0]))
                box = torch.tensor([float(v) for v in parts[1:5]], dtype=torch.float32)
                labels.append(cls_id)
                boxes.append(box)
        if not boxes:
            return torch.zeros(0, dtype=torch.long), torch.zeros((0, 4), dtype=torch.float32)
        boxes_xywhn = torch.stack(boxes, dim=0)
        boxes_xyxy = xywhn_to_xyxy(boxes_xywhn, width=self.img_size, height=self.img_size)
        return torch.tensor(labels, dtype=torch.long), boxes_xyxy

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        image_tensor, meta = self.transform(image)
        labels, boxes = self._read_labels(image_path)
        target = {
            "labels": labels,
            "boxes": boxes,
            "image_id": image_path.stem,
            "image_path": image_path,
            "original_size": meta["original_size"],
            "resized_size": meta["resized_size"],
        }
        return {"image": image_tensor, "target": target}

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Stack images and keep targets as a list."""
        return {
            "images": torch.stack([sample["image"] for sample in batch], dim=0),
            "targets": [sample["target"] for sample in batch],
        }
