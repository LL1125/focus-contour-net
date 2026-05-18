"""Inspect one dataloader batch for quick debugging."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.detect_dataset import DetectDataset
from models.common.utils import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load one batch and print shapes.")
    parser.add_argument("--data-config", type=Path, default=ROOT / "configs" / "data" / "detect_data.yaml")
    parser.add_argument("--img-size", type=int, default=320)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_cfg = load_yaml(args.data_config)
    dataset = DetectDataset(data_cfg, split="train", img_size=args.img_size)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=DetectDataset.collate_fn)
    batch = next(iter(loader))
    print(batch["images"].shape)
    print([{k: (v.shape if hasattr(v, "shape") else v) for k, v in target.items() if k in {"labels", "boxes"}} for target in batch["targets"]])


if __name__ == "__main__":
    main()
