"""CLI entrypoint for end-to-end prediction."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.predictor import Predictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run end-to-end prediction.")
    parser.add_argument("--model-config", type=Path, default=ROOT / "configs" / "model" / "fcn_base.yaml")
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--raw", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictor = Predictor(args.model_config, weights=args.weights, device=args.device)
    outputs = predictor.predict_image(args.image, return_raw=args.raw)
    print(outputs["detections"][0][:10])


if __name__ == "__main__":
    main()
