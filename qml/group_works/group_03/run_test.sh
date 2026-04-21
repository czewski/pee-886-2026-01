#!/usr/bin/env bash
set -euo pipefail

BATCH_SIZE="${1:-128}"
NQUBITS="${2:-10}"
NLAYERS="${3:-3}"
EPOCHS_PER_FOLD="${4:-40}"
PATIENCE="${5:-5}"
LEARNING_RATE="${6:-0.001}"
SEED="${7:-2026}"
N_FOLDS="${8:-5}"
NUM_WORKERS="${9:-5}"
FIT="${10:-1}"

FIT_ARG=()
if [[ "${FIT}" == "1" || "${FIT}" == "true" || "${FIT}" == "--fit" ]]; then
  FIT_ARG=(--fit)
fi

python -m qml.group_works.group_03.run_test \
  "${FIT_ARG[@]}" \
  --batch-size "${BATCH_SIZE}" \
  --nqubits "${NQUBITS}" \
  --nlayers "${NLAYERS}" \
  --epochs-per-fold "${EPOCHS_PER_FOLD}" \
  --patience "${PATIENCE}" \
  --learning-rate "${LEARNING_RATE}" \
  --seed "${SEED}" \
  --n-folds "${N_FOLDS}" \
  --num-workers "${NUM_WORKERS}"
