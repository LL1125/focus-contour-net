"""CLI entrypoint for detect-only training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.trainer_detect import TrainerDetect


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the detect-only baseline.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "train" / "train_detect.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    TrainerDetect(args.config).train()


if __name__ == "__main__":
    main()
