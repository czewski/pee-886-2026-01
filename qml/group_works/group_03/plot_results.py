import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate training curves and fold metrics visualizations."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("outputs"),
        help="Root directory containing experiment folders.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Single experiment folder name to process. If omitted, processes all.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional root directory to write figures. Defaults to each experiment folder.",
    )
    return parser.parse_args()


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _collect_folds(experiment_dir: Path):
    metrics_path = experiment_dir / "metrics.json"
    folds = {}
    evaluation = {}

    if metrics_path.exists():
        metrics = _load_json(metrics_path)
        folds = metrics.get("folds", {})
        evaluation = metrics.get("evaluation", {})

    if not folds:
        for fold_file in sorted(experiment_dir.glob("metrics_fold_*.json")):
            fold_name = fold_file.stem.replace("metrics_", "")
            folds[fold_name] = _load_json(fold_file)

    if not evaluation:
        evaluation_path = experiment_dir / "metrics_evaluation.json"
        if evaluation_path.exists():
            evaluation = _load_json(evaluation_path)

    return folds, evaluation


def _stack_with_nan(series_list):
    max_len = max(len(values) for values in series_list)
    stacked = np.full((len(series_list), max_len), np.nan, dtype=float)
    for idx, values in enumerate(series_list):
        values = np.asarray(values, dtype=float)
        stacked[idx, : len(values)] = values
    return stacked


def _plot_training_curves(folds, out_dir: Path):
    fold_names = sorted(folds.keys())
    train_loss = [folds[name]["train_loss"] for name in fold_names]
    val_loss = [folds[name]["val_loss"] for name in fold_names]
    val_acc = [folds[name]["val_acc"] for name in fold_names]

    train_stack = _stack_with_nan(train_loss)
    val_stack = _stack_with_nan(val_loss)
    acc_stack = _stack_with_nan(val_acc)

    epochs_loss = np.arange(1, train_stack.shape[1] + 1)
    epochs_acc = np.arange(1, acc_stack.shape[1] + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, name in enumerate(fold_names):
        axes[0].plot(
            np.arange(1, len(train_loss[idx]) + 1),
            train_loss[idx],
            alpha=0.25,
            linewidth=1,
            color="tab:blue",
        )
        axes[0].plot(
            np.arange(1, len(val_loss[idx]) + 1),
            val_loss[idx],
            alpha=0.25,
            linewidth=1,
            color="tab:orange",
        )
        axes[1].plot(
            np.arange(1, len(val_acc[idx]) + 1),
            val_acc[idx],
            alpha=0.25,
            linewidth=1,
            label=name,
        )

    train_mean = np.nanmean(train_stack, axis=0)
    train_std = np.nanstd(train_stack, axis=0)
    val_mean = np.nanmean(val_stack, axis=0)
    val_std = np.nanstd(val_stack, axis=0)
    acc_mean = np.nanmean(acc_stack, axis=0)
    acc_std = np.nanstd(acc_stack, axis=0)

    axes[0].plot(epochs_loss, train_mean, color="tab:blue", linewidth=2.5, label="Train mean")
    axes[0].fill_between(
        epochs_loss,
        train_mean - train_std,
        train_mean + train_std,
        color="tab:blue",
        alpha=0.15,
        label="Train ± std",
    )
    axes[0].plot(epochs_loss, val_mean, color="tab:orange", linewidth=2.5, label="Val mean")
    axes[0].fill_between(
        epochs_loss,
        val_mean - val_std,
        val_mean + val_std,
        color="tab:orange",
        alpha=0.15,
        label="Val ± std",
    )
    axes[0].set_title("Loss curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs_acc, acc_mean, color="tab:green", linewidth=2.5, label="Val acc mean")
    axes[1].fill_between(
        epochs_acc,
        acc_mean - acc_std,
        acc_mean + acc_std,
        color="tab:green",
        alpha=0.2,
        label="Val acc ± std",
    )
    axes[1].set_title("Validation accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_fold_metrics(folds, evaluation, out_dir: Path):
    fold_names = sorted(folds.keys())
    val_best = [float(np.max(folds[name]["val_acc"])) for name in fold_names]
    val_last = [float(folds[name]["val_acc"][-1]) for name in fold_names]
    train_seconds = [float(folds[name].get("train_time_seconds", np.nan)) for name in fold_names]

    test_acc = evaluation.get("test_acc_per_fold", [])
    avg_test = evaluation.get("avg_test_acc", np.nan)
    std_test = evaluation.get("std_test_acc", np.nan)

    x = np.arange(len(fold_names))
    width = 0.35
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].bar(x - width / 2, val_last, width, label="Final val acc")
    axes[0].bar(x + width / 2, val_best, width, label="Best val acc")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(fold_names, rotation=30)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("Validation accuracy by fold")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()

    axes[1].bar(x, train_seconds, color="tab:purple")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(fold_names, rotation=30)
    axes[1].set_ylabel("Seconds")
    axes[1].set_title("Training time by fold")
    axes[1].grid(axis="y", alpha=0.25)

    if len(test_acc) == len(fold_names):
        axes[2].bar(x, test_acc, color="tab:red", label="Test acc")
        axes[2].axhline(avg_test, color="black", linestyle="--", linewidth=1.5, label=f"Mean {avg_test:.2f}%")
        axes[2].fill_between(
            [-0.5, len(fold_names) - 0.5],
            avg_test - std_test,
            avg_test + std_test,
            color="gray",
            alpha=0.15,
            label=f"±{std_test:.2f}",
        )
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(fold_names, rotation=30)
        axes[2].set_ylabel("Accuracy (%)")
        axes[2].set_title("Test accuracy by fold")
        axes[2].grid(axis="y", alpha=0.25)
        axes[2].legend()
    else:
        axes[2].axis("off")
        axes[2].text(
            0.5,
            0.5,
            "No test metrics available.",
            ha="center",
            va="center",
            fontsize=12,
        )

    fig.tight_layout()
    fig.savefig(out_dir / "metrics_summary.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _process_experiment(experiment_dir: Path, output_root: Optional[Path]):
    folds, evaluation = _collect_folds(experiment_dir)
    if not folds:
        return False

    if output_root is None:
        out_dir = experiment_dir / "plots"
    else:
        out_dir = output_root / experiment_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    _plot_training_curves(folds, out_dir)
    _plot_fold_metrics(folds, evaluation, out_dir)
    return True


def main():
    args = parse_args()
    results_root = args.results_root

    if args.experiment:
        candidates = [results_root / args.experiment]
    else:
        candidates = [path for path in sorted(results_root.iterdir()) if path.is_dir()]

    processed = []
    for experiment_dir in candidates:
        if _process_experiment(experiment_dir, args.output_root):
            processed.append(experiment_dir.name)

    if not processed:
        raise FileNotFoundError(
            f"No valid experiment metrics were found in: {results_root}"
        )

    print("Generated plots for:", ", ".join(processed))


if __name__ == "__main__":
    main()
