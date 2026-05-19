"""Validation loop for detect-only training with standard detection metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from models.common.utils import bbox_iou


class ValidatorDetect:
    """Run validation for the detect-only baseline."""

    def __init__(self, device: torch.device, iou_thresholds: torch.Tensor | None = None) -> None:
        self.device = device
        self.iouv = iou_thresholds if iou_thresholds is not None else torch.linspace(0.5, 0.95, 10)

    def _mean_best_iou(self, detections: torch.Tensor, targets: list[dict[str, torch.Tensor]]) -> float:
        scores = []
        for det, target in zip(detections, targets):
            gt_boxes = target["boxes"].to(det.device)
            if gt_boxes.numel() == 0 or det.numel() == 0:
                continue
            ious = bbox_iou(det[:, None, :4], gt_boxes[None, :, :])
            scores.append(float(ious.max(dim=0).values.mean().item()))
        return sum(scores) / max(len(scores), 1)

    @staticmethod
    def _compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0]))
        mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
        x = np.linspace(0.0, 1.0, 101)
        return float(np.trapz(np.interp(x, mrec, mpre), x))

    def _match_detections(
        self,
        detections: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        detections = detections.detach().cpu()
        gt_boxes = gt_boxes.detach().cpu()
        gt_labels = gt_labels.detach().cpu().long()

        num_det = int(detections.shape[0])
        correct = torch.zeros((num_det, int(self.iouv.numel())), dtype=torch.bool)
        if num_det == 0:
            return correct, torch.zeros(0), torch.zeros(0, dtype=torch.long), gt_labels
        if gt_boxes.numel() == 0:
            return correct, detections[:, 4], detections[:, 5].long(), gt_labels

        det_boxes = detections[:, :4]
        det_conf = detections[:, 4]
        det_cls = detections[:, 5].long()

        ious = bbox_iou(det_boxes[:, None, :], gt_boxes[None, :, :]).cpu()
        class_mask = det_cls[:, None] == gt_labels[None, :]
        ious = ious * class_mask.float()

        for thr_idx, thr in enumerate(self.iouv):
            det_idx, gt_idx = torch.where(ious >= float(thr))
            if det_idx.numel() == 0:
                continue
            match_ious = ious[det_idx, gt_idx]
            order = torch.argsort(match_ious, descending=True)
            used_det: set[int] = set()
            used_gt: set[int] = set()
            for order_idx in order.tolist():
                di = int(det_idx[order_idx].item())
                gi = int(gt_idx[order_idx].item())
                if di in used_det or gi in used_gt:
                    continue
                correct[di, thr_idx] = True
                used_det.add(di)
                used_gt.add(gi)
        return correct, det_conf, det_cls, gt_labels

    def _compute_detection_metrics(
        self,
        correct_list: list[torch.Tensor],
        conf_list: list[torch.Tensor],
        pred_cls_list: list[torch.Tensor],
        target_cls_list: list[torch.Tensor],
    ) -> dict[str, float]:
        if not target_cls_list:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "map50": 0.0,
                "map50_95": 0.0,
            }

        target_cls = torch.cat(target_cls_list, dim=0).cpu().numpy().astype(np.int64)
        if target_cls.size == 0:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "map50": 0.0,
                "map50_95": 0.0,
            }

        if conf_list:
            conf = torch.cat(conf_list, dim=0).cpu().numpy()
            pred_cls = torch.cat(pred_cls_list, dim=0).cpu().numpy().astype(np.int64)
            correct = torch.cat(correct_list, dim=0).cpu().numpy().astype(np.float32)
        else:
            conf = np.zeros((0,), dtype=np.float32)
            pred_cls = np.zeros((0,), dtype=np.int64)
            correct = np.zeros((0, int(self.iouv.numel())), dtype=np.float32)

        unique_classes = np.unique(target_cls)
        eps = 1e-16
        precision_per_class: list[float] = []
        recall_per_class: list[float] = []
        ap_per_class: list[np.ndarray] = []

        for cls_id in unique_classes:
            gt_count = int((target_cls == cls_id).sum())
            pred_mask = pred_cls == cls_id
            pred_count = int(pred_mask.sum())
            if gt_count == 0:
                continue
            if pred_count == 0:
                precision_per_class.append(0.0)
                recall_per_class.append(0.0)
                ap_per_class.append(np.zeros(int(self.iouv.numel()), dtype=np.float32))
                continue

            cls_conf = conf[pred_mask]
            cls_correct = correct[pred_mask]
            order = np.argsort(-cls_conf)
            cls_correct = cls_correct[order]
            cls_tp = cls_correct.cumsum(0)
            cls_fp = (1.0 - cls_correct).cumsum(0)
            recall_curve = cls_tp / (gt_count + eps)
            precision_curve = cls_tp / (cls_tp + cls_fp + eps)

            ap = np.zeros(int(self.iouv.numel()), dtype=np.float32)
            for thr_idx in range(int(self.iouv.numel())):
                ap[thr_idx] = self._compute_ap(recall_curve[:, thr_idx], precision_curve[:, thr_idx])

            f1 = 2.0 * precision_curve[:, 0] * recall_curve[:, 0] / (precision_curve[:, 0] + recall_curve[:, 0] + eps)
            best_idx = int(np.argmax(f1)) if f1.size else 0
            precision_per_class.append(float(precision_curve[best_idx, 0]) if precision_curve.size else 0.0)
            recall_per_class.append(float(recall_curve[best_idx, 0]) if recall_curve.size else 0.0)
            ap_per_class.append(ap)

        if not ap_per_class:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "map50": 0.0,
                "map50_95": 0.0,
            }

        ap_array = np.stack(ap_per_class, axis=0)
        return {
            "precision": float(np.mean(precision_per_class)) if precision_per_class else 0.0,
            "recall": float(np.mean(recall_per_class)) if recall_per_class else 0.0,
            "map50": float(ap_array[:, 0].mean()),
            "map50_95": float(ap_array.mean()),
        }

    def validate(self, model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, criterion: torch.nn.Module) -> dict[str, float]:
        """Evaluate loss, a lightweight IoU proxy, and standard detection metrics."""
        model.eval()
        total_loss = 0.0
        total_batches = 0
        total_iou = 0.0
        correct_list: list[torch.Tensor] = []
        conf_list: list[torch.Tensor] = []
        pred_cls_list: list[torch.Tensor] = []
        target_cls_list: list[torch.Tensor] = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="val", leave=False):
                images = batch["images"].to(self.device)
                targets = batch["targets"]
                for target in targets:
                    target["labels"] = target["labels"].to(self.device)
                    target["boxes"] = target["boxes"].to(self.device)
                outputs = model.forward_train(images)
                loss_dict = criterion(outputs["detection_outputs"], targets)
                infer = model.forward_infer(images)
                detections = infer["detections"]
                total_loss += float(loss_dict["loss"].item())
                total_iou += self._mean_best_iou(detections, targets)
                total_batches += 1

                for det, target in zip(detections, targets):
                    correct, conf, pred_cls, target_cls = self._match_detections(det, target["boxes"], target["labels"])
                    correct_list.append(correct)
                    conf_list.append(conf)
                    pred_cls_list.append(pred_cls)
                    target_cls_list.append(target_cls)

        metrics = self._compute_detection_metrics(correct_list, conf_list, pred_cls_list, target_cls_list)
        metrics.update(
            {
                "val_loss": total_loss / max(total_batches, 1),
                "val_mean_best_iou": total_iou / max(total_batches, 1),
            }
        )
        return metrics
