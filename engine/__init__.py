"""Engine helpers for model construction, checkpointing, and device selection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from models.common.utils import load_yaml
from models.detectors import YOLO26Base, YOLO26FocusFourier

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_device(device: str = "auto") -> torch.device:
    """Resolve a user-facing device string into a torch device."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a path relative to the project root when needed."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()


def load_model_config(model_config: str | Path | dict[str, Any]) -> dict[str, Any]:
    """Load model config from YAML or pass through a dict."""
    if isinstance(model_config, dict):
        return model_config
    return load_yaml(resolve_project_path(model_config))


def build_model(model_cfg: dict[str, Any]) -> torch.nn.Module:
    """Instantiate the configured detector."""
    model_type = model_cfg.get("model_type", "yolo26_base")
    if model_type == "yolo26_base":
        return YOLO26Base(model_cfg)
    if model_type == "yolo26_focus_fourier":
        return YOLO26FocusFourier(model_cfg)
    raise ValueError(f"Unsupported model_type: {model_type}")


def save_checkpoint(path: str | Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, extra: dict[str, Any] | None = None) -> None:
    """Persist a training checkpoint."""
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, resolve_project_path(path))


def load_checkpoint(path: str | Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Restore model and optional optimizer states."""
    checkpoint = torch.load(resolve_project_path(path), map_location=map_location)
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint
