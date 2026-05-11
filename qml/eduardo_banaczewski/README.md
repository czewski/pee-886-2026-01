# Eduardo Banaczewski - CIFAR Experiment

This namespace contains a modular CIFAR-10 experiment with:

- reproducible seeds
- CIFAR dataloaders with deterministic split/shuffle
- CNN benchmark model
- QML hybrid model variants
- trainer and evaluation/export utilities

## Modules

- `experiment.py`: config dataclass, seed setup, output run directory.
- `loaders/cifar10_loader.py`: CIFAR-10 train/val/test loaders.
- `models/cnn_benchmark.py`: baseline CNN.
- `models/qml_hybrid.py`: hybrid quantum-classical model variants.
- `models/factory.py`: model selection by config name.
- `trainer/cifar_trainer.py`: train/eval loop.
- `evaluation/exporters.py`: JSON/CSV result export.
- `visualization/plots.py`: training curves and confusion matrix images.

## Runner

Use:

```bash
python3 scripts/eduardo_banaczewski/run_cifar_experiment.py \
  --experiment-name exp_cnn_baseline \
  --model-name cnn_benchmark \
  --n-folds 5 \
  --seed 42
```

Available model names:

- `cnn_benchmark`
- `qml_baseline`
- `qml_strong`
- `qml_data_reupload`

## Outputs

Each run exports artifacts to:

`outputs/<experiment_name>/`

Expected files include:

- `metrics.json`
- `fold_accuracy_comparison.pdf`
- `fold_1/metrics.json` ... `fold_5/metrics.json`
- `fold_1/training_curves.pdf` ... `fold_5/training_curves.pdf`
- `fold_1/confusion_matrix.pdf` ... `fold_5/confusion_matrix.pdf`
- `fold_1/test_predictions.csv` ... `fold_5/test_predictions.csv`
- `fold_1/best_model.pth` ... `fold_5/best_model.pth`

## Resume support

If execution is interrupted, re-run the same command with the same
`--experiment-name`. The trainer resumes incomplete folds from
`fold_*/checkpoint.pth` and skips completed folds.
