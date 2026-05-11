# Scripts - Eduardo Banaczewski

## CIFAR experiment runner

```bash
python3 scripts/eduardo_banaczewski/run_cifar_experiment.py \
  --experiment-name exp_qml_baseline \
  --model-name qml_baseline \
  --n-folds 5 \
  --epochs 15 \
  --seed 42
```

The command creates `outputs/<experiment_name>/` with per-fold checkpoints, metrics, CSV predictions, and PDF plots.
