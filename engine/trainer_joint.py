"""Joint trainer scaffold for detect + contour experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.joint_dataset import JointDataset
from engine import build_model, load_checkpoint, load_model_config, resolve_device, resolve_project_path, save_checkpoint
from engine.validator_joint import ValidatorJoint
from models.common.utils import load_yaml, set_seed
from models.losses.joint_loss import JointLoss


class TrainerJoint:
    """Train the focus + Fourier detector scaffold."""

    def __init__(self, train_config: str | Path | dict[str, Any]) -> None:
        self.cfg = load_yaml(train_config) if not isinstance(train_config, dict) else train_config
        set_seed(int(self.cfg.get("seed", 42)))
        self.device = resolve_device(self.cfg.get("device", "auto"))
        self.output_dir = resolve_project_path(self.cfg.get("output_dir", "runs/joint"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model_cfg = load_model_config(self.cfg["model_config"])
        self.data_cfg = load_yaml(resolve_project_path(self.cfg["data_config"]))
        self.model = build_model(self.model_cfg).to(self.device)
        self.criterion = JointLoss(num_classes=int(self.model_cfg["num_classes"]), loss_cfg=self.cfg.get("loss")).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.cfg.get("learning_rate", 1e-3)),
            weight_decay=float(self.cfg.get("weight_decay", 5e-4)),
        )
        self.validator = ValidatorJoint(self.device)
        self.start_epoch = 0
        if self.cfg.get("resume"):
            checkpoint = load_checkpoint(self.cfg["resume"], self.model, self.optimizer, map_location=self.device)
            self.start_epoch = int(checkpoint.get("epoch", 0)) + 1

    def build_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        train_set = JointDataset(self.data_cfg, split="train", img_size=int(self.cfg.get("img_size", 640)))
        val_set = JointDataset(self.data_cfg, split="val", img_size=int(self.cfg.get("img_size", 640)))
        batch_size = int(self.cfg.get("batch_size", 4))
        num_workers = int(self.cfg.get("num_workers", 4))
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=JointDataset.collate_fn)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=JointDataset.collate_fn)
        return train_loader, val_loader

    def _move_targets_to_device(self, targets: list[dict[str, Any]]) -> None:
        for target in targets:
            for key in ("labels", "boxes", "contours", "boundary"):
                if key in target and isinstance(target[key], torch.Tensor):
                    target[key] = target[key].to(self.device)

    def train(self) -> None:
        train_loader, val_loader = self.build_dataloaders()
        epochs = int(self.cfg.get("epochs", 100))
        eval_every = int(self.cfg.get("eval_every", 1))
        save_every = int(self.cfg.get("save_every", 10))

        for epoch in range(self.start_epoch, epochs):
            self.model.train()
            running_loss = 0.0
            for batch in tqdm(train_loader, desc=f"joint {epoch}", leave=False):
                images = batch["images"].to(self.device)
                targets = batch["targets"]
                self._move_targets_to_device(targets)
                outputs = self.model(images)
                loss_dict = self.criterion(outputs, targets)
                self.optimizer.zero_grad(set_to_none=True)
                loss_dict["loss"].backward()
                self.optimizer.step()
                running_loss += float(loss_dict["loss"].item())

            train_loss = running_loss / max(len(train_loader), 1)
            print(f"epoch={epoch} train_loss={train_loss:.4f}")
            if (epoch + 1) % eval_every == 0:
                metrics = self.validator.validate(self.model, val_loader, self.criterion)
                print(
                    f"epoch={epoch} val_loss={metrics['val_loss']:.4f} "
                    f"val_contour_loss={metrics['val_contour_loss']:.4f} "
                    f"val_mean_best_iou={metrics['val_mean_best_iou']:.4f}"
                )
            if (epoch + 1) % save_every == 0 or epoch + 1 == epochs:
                save_checkpoint(self.output_dir / f"epoch_{epoch}.pt", self.model, self.optimizer, epoch, extra={"model_cfg": self.model_cfg})
