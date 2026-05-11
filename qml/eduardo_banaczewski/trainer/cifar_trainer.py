import os
import time
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset, Subset

from qml.eduardo_banaczewski.evaluation.exporters import (
    save_metrics_json,
    save_test_predictions_csv,
)
from qml.eduardo_banaczewski.experiment import CifarExperimentConfig
from qml.eduardo_banaczewski.visualization.plots import (
    plot_confusion_matrix,
    plot_fold_accuracy_comparison,
    plot_training_curves,
)

__all__ = ["TrainResult", "CifarTrainer"]


@dataclass(frozen=True)
class TrainResult:
    fold_metrics: Dict[str, Dict[str, object]]
    summary: Dict[str, object]
    experiment_dir: Path


class CifarTrainer:
    def __init__(
        self,
        model_builder: Callable[[], nn.Module],
        config: CifarExperimentConfig,
        output_dir: Path,
        class_names: Tuple[str, ...],
    ):
        """Initialize trainer state for k-fold CIFAR experiments."""
        self.model_builder = model_builder
        self.config = config
        self.output_dir = output_dir
        self.class_names = class_names
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.CrossEntropyLoss()
        os.makedirs(self.output_dir, exist_ok=True)

    def _loader_from_indices(
        self,
        dataset: Dataset,
        indices: Sequence[int],
        shuffle: bool,
        seed_offset: int = 0,
    ) -> DataLoader:
        """Create a dataloader from a subset of indices."""
        return DataLoader(
            Subset(dataset, indices),
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.config.num_workers > 0,
            generator=torch.Generator().manual_seed(self.config.seed + seed_offset),
        )

    def _run_epoch(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loader: DataLoader,
        training: bool,
    ) -> Tuple[float, float]:
        """Run one epoch and return average loss and accuracy."""
        if training:
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        correct = 0
        total = 0
        with torch.set_grad_enabled(training):
            for images, labels in loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                if training:
                    optimizer.zero_grad()
                logits = model(images)
                loss = self.criterion(logits, labels)
                if training:
                    loss.backward()
                    optimizer.step()
                running_loss += loss.item() * labels.size(0)
                correct += (logits.argmax(dim=1) == labels).sum().item()
                total += labels.size(0)
        return running_loss / total, correct / total

    def _fold_dir(self, fold_index: int) -> Path:
        """Return the output directory for one fold."""
        fold_dir = self.output_dir / f"fold_{fold_index + 1}"
        os.makedirs(fold_dir, exist_ok=True)
        return fold_dir

    def _train_single_fold(
        self,
        fold_index: int,
        train_indices: List[int],
        val_indices: List[int],
        train_dataset_aug: Dataset,
        train_dataset_eval: Dataset,
        test_loader: DataLoader,
    ) -> Path:
        """Train/evaluate one fold with checkpoint resume support."""
        fold_dir = self._fold_dir(fold_index)
        checkpoint_path = fold_dir / "checkpoint.pth"
        best_model_path = fold_dir / "best_model.pth"
        fold_metrics_path = fold_dir / "metrics.json"

        if best_model_path.exists() and fold_metrics_path.exists():
            return fold_metrics_path

        model = self.model_builder().to(self.device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }
        best_val_acc = -1.0
        start_epoch = 0
        elapsed_seconds = 0.0

        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            history = checkpoint["history"]
            best_val_acc = checkpoint["best_val_acc"]
            start_epoch = int(checkpoint["epoch"]) + 1
            elapsed_seconds = float(checkpoint.get("elapsed_seconds", 0.0))

        train_loader = self._loader_from_indices(
            train_dataset_aug,
            train_indices,
            shuffle=True,
            seed_offset=fold_index,
        )
        val_loader = self._loader_from_indices(
            train_dataset_eval,
            val_indices,
            shuffle=False,
            seed_offset=100 + fold_index,
        )

        train_start = time.time()
        for epoch in range(start_epoch, self.config.epochs):
            train_loss, train_acc = self._run_epoch(model, optimizer, train_loader, training=True)
            val_loss, val_acc = self._run_epoch(model, optimizer, val_loader, training=False)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), best_model_path)

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "history": history,
                    "best_val_acc": best_val_acc,
                    "elapsed_seconds": elapsed_seconds + (time.time() - train_start),
                },
                checkpoint_path,
            )

        elapsed_seconds += time.time() - train_start
        plot_training_curves(history, fold_dir / "training_curves.pdf")

        model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        model.eval()
        all_labels: List[np.ndarray] = []
        all_preds: List[np.ndarray] = []
        prediction_rows: List[Dict[str, int]] = []
        with torch.no_grad():
            for batch_index, (images, labels) in enumerate(test_loader):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                logits = model(images)
                preds = logits.argmax(dim=1)
                all_labels.append(labels.cpu().numpy())
                all_preds.append(preds.cpu().numpy())
                for sample_index in range(labels.shape[0]):
                    prediction_rows.append(
                        {
                            "batch": batch_index,
                            "sample": sample_index,
                            "true_label": int(labels[sample_index].cpu()),
                            "pred_label": int(preds[sample_index].cpu()),
                        }
                    )

        labels_np = np.concatenate(all_labels)
        preds_np = np.concatenate(all_preds)
        test_acc = float((labels_np == preds_np).mean())
        conf = confusion_matrix(labels_np, preds_np, labels=np.arange(len(self.class_names)))
        plot_confusion_matrix(conf, self.class_names, fold_dir / "confusion_matrix.pdf")
        save_test_predictions_csv(prediction_rows, fold_dir / "test_predictions.csv")

        fold_metrics = {
            "fold": fold_index + 1,
            "history": history,
            "best_val_acc": best_val_acc,
            "test_acc": test_acc,
            "train_time_seconds": elapsed_seconds,
        }
        save_metrics_json(fold_metrics, fold_metrics_path)
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        return fold_metrics_path

    def run(
        self,
        train_dataset_aug: Dataset,
        train_dataset_eval: Dataset,
        train_targets: torch.Tensor,
        test_loader: DataLoader,
    ) -> TrainResult:
        """Run full stratified k-fold training and aggregate experiment metrics."""
        split = StratifiedKFold(
            n_splits=self.config.n_folds,
            shuffle=True,
            random_state=self.config.seed,
        )

        fold_metrics: Dict[str, Dict[str, object]] = {}
        val_acc_by_fold: List[float] = []
        test_acc_by_fold: List[float] = []
        train_indices_ref = np.arange(len(train_targets))
        for fold_index, (train_idx, val_idx) in enumerate(
            split.split(train_indices_ref, train_targets.numpy())
        ):
            fold_metrics_path = self._train_single_fold(
                fold_index=fold_index,
                train_indices=train_idx.tolist(),
                val_indices=val_idx.tolist(),
                train_dataset_aug=train_dataset_aug,
                train_dataset_eval=train_dataset_eval,
                test_loader=test_loader,
            )
            with fold_metrics_path.open("r", encoding="utf-8") as stream:
                metrics = json.load(stream)
            fold_key = f"fold_{fold_index + 1}"
            fold_metrics[fold_key] = metrics
            val_acc_by_fold.append(metrics["best_val_acc"])
            test_acc_by_fold.append(metrics["test_acc"])

        val_acc_np = np.asarray(val_acc_by_fold, dtype=float)
        test_acc_np = np.asarray(test_acc_by_fold, dtype=float)
        summary = {
            "val_acc_mean": float(val_acc_np.mean()),
            "val_acc_std": float(val_acc_np.std()),
            "test_acc_mean": float(test_acc_np.mean()),
            "test_acc_std": float(test_acc_np.std()),
            "val_acc_by_fold": val_acc_by_fold,
            "test_acc_by_fold": test_acc_by_fold,
        }
        plot_fold_accuracy_comparison(
            val_acc=val_acc_np,
            test_acc=test_acc_np,
            output_path=self.output_dir / "fold_accuracy_comparison.pdf",
        )
        save_metrics_json(
            {"config": self.config.to_dict(), "folds": fold_metrics, "summary": summary},
            self.output_dir / "metrics.json",
        )
        return TrainResult(
            fold_metrics=fold_metrics,
            summary=summary,
            experiment_dir=self.output_dir,
        )
