# my_yolo26_fourier

A clean, standalone, PyTorch-based YOLO26-style minimal detection project with:

- detect-only baseline training and validation
- dual-head end-to-end detection design
- pluggable backbone/neck focus-region enhancement hooks
- parallel Fourier contour head interfaces for future joint training

This project does not depend on the `ultralytics` Python package and does not modify any existing repository in-place.

## Current status

- Detect-only baseline: runnable
- Joint detect + contour branch: runnable scaffold with placeholder contour supervision
- Focus plugin: lightweight replaceable implementation
- Fourier contour head: interface shell with executable placeholder logic

## Quick start

```bash
python tools/train_detect.py --config configs/train/train_debug.yaml
python scripts/check_model.py --model-config configs/model/y26_base.yaml
```

## Ablation targets

- `y26_base`: baseline detect-only
- `y26_focus_fourier`: + focus plugin + contour branch
- `y26_focus_fourier_p2`: reserved small-object variant with P2 output flag in config

## Notes

- Paths use `pathlib` throughout for Ubuntu/Windows compatibility.
- The contour branch is intentionally lightweight for now and marked with TODOs where your formal module should be inserted later.
