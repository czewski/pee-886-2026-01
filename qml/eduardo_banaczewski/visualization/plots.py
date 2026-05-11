from pathlib import Path
from typing import Dict, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

__all__ = ["plot_confusion_matrix", "plot_training_curves"]
__all__.append("plot_fold_accuracy_comparison")


def plot_training_curves(history: Dict[str, Sequence[float]], output_path: Path) -> None:
    """Plot training/validation loss and accuracy curves for one fold."""
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="train_loss")
    axes[0].plot(epochs, history["val_loss"], label="val_loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves")
    axes[0].legend()
    axes[0].grid(True, alpha=0.2)

    axes[1].plot(epochs, history["train_acc"], label="train_acc")
    axes[1].plot(epochs, history["val_acc"], label="val_acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy Curves")
    axes[1].legend()
    axes[1].grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: Tuple[str, ...],
    output_path: Path,
) -> None:
    """Plot and export confusion matrix for one fold."""
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(confusion_matrix, cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    threshold = confusion_matrix.max() * 0.5 if confusion_matrix.size else 0.0
    for row in range(confusion_matrix.shape[0]):
        for col in range(confusion_matrix.shape[1]):
            color = "white" if confusion_matrix[row, col] > threshold else "black"
            ax.text(col, row, str(confusion_matrix[row, col]), ha="center", va="center", color=color)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_fold_accuracy_comparison(
    val_acc: np.ndarray,
    test_acc: np.ndarray,
    output_path: Path,
) -> None:
    """Plot grouped bars comparing validation and test accuracy by fold."""
    folds = np.arange(1, val_acc.size + 1)
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(folds - width / 2, val_acc, width=width, label="Validation Accuracy")
    ax.bar(folds + width / 2, test_acc, width=width, label="Test Accuracy")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Accuracy")
    ax.set_title("Validation vs Test Accuracy by Fold")
    ax.set_xticks(folds)
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
