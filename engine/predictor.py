"""Prediction helper for standalone inference scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image

from datasets.transforms import ResizeToTensor
from engine import build_model, load_checkpoint, load_model_config, resolve_device, resolve_project_path


class Predictor:
    """Load a model and run end-to-end detection inference."""

    def __init__(self, model_config: str | Path | dict[str, Any], weights: str | Path | None = None, device: str = "auto") -> None:
        self.model_cfg = load_model_config(model_config)
        self.device = resolve_device(device)
        self.model = build_model(self.model_cfg).to(self.device)
        self.model.eval()
        if weights is not None:
            load_checkpoint(weights, self.model, optimizer=None, map_location=self.device)
        self.transform = ResizeToTensor(int(self.model_cfg.get("img_size", 640) or 640))

    def predict_image(self, image_path: str | Path, return_raw: bool = False) -> dict[str, Any]:
        """Run model inference for a single image."""
        image = Image.open(resolve_project_path(image_path)).convert("RGB")
        tensor, meta = self.transform(image)
        inputs = tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model.forward_infer(inputs, return_raw=return_raw)
        outputs["meta"] = meta
        return outputs
