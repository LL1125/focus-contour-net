"""CLI entrypoint for detect-only validation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.detect_dataset import DetectDataset
from engine import build_model, load_checkpoint, load_model_config, resolve_device, resolve_project_path
from engine.validator_detect import ValidatorDetect
from models.common.utils import load_yaml
from models.losses.detect_loss_e2e import DetectLossE2E
from torch.utils.data import DataLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the detect-only baseline.")
    parser.add_argument("--train-config", type=Path, default=ROOT / "configs" / "train" / "train_detect.yaml")
    parser.add_argument("--weights", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.train_config)
    model_cfg = load_model_config(cfg["model_config"])
    data_cfg = load_yaml(resolve_project_path(cfg["data_config"]))
    device = resolve_device(cfg.get("device", "auto"))
    model = build_model(model_cfg).to(device)
    if args.weights:
        load_checkpoint(args.weights, model, optimizer=None, map_location=device)
    criterion = DetectLossE2E(num_classes=int(model_cfg["num_classes"]), loss_cfg=cfg.get("loss")).to(device)
    val_set = DetectDataset(data_cfg, split="val", img_size=int(cfg.get("img_size", 640)))
    val_loader = DataLoader(val_set, batch_size=int(cfg.get("batch_size", 8)), shuffle=False, num_workers=int(cfg.get("num_workers", 4)), collate_fn=DetectDataset.collate_fn)
    metrics = ValidatorDetect(device).validate(model, val_loader, criterion)
    print(metrics)


if __name__ == "__main__":
    main()
