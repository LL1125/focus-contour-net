"""Sanity-check FocusContourNet model construction and a dummy forward pass."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine import build_model, load_model_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check model creation and forward shapes.")
    parser.add_argument("--model-config", type=Path, default=ROOT / "configs" / "model" / "fcn_base.yaml")
    parser.add_argument("--img-size", type=int, default=320)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_cfg = load_model_config(args.model_config)
    model = build_model(model_cfg)
    dummy = torch.randn(2, int(model_cfg.get("in_channels", 3)), args.img_size, args.img_size)

    model.train()
    train_out = model(dummy)
    print("train keys:", list(train_out.keys()))

    model.eval()
    with torch.no_grad():
        infer_out = model(dummy, return_raw=True)
    print("infer keys:", list(infer_out.keys()))
    print("detections:", infer_out["detections"].shape)


if __name__ == "__main__":
    main()
