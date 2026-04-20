#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LATENT_DIM="${LATENT_DIM:-32}"
DISTANCE_WEIGHT="${DISTANCE_WEIGHT:-1.0}"
FORCE_RETRAIN="${FORCE_RETRAIN:-0}"

if [ "$#" -gt 0 ]; then
  DIMS=("$@")
else
  DIMS=(2 8 16 512)
fi

export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

for dim in "${DIMS[@]}"; do
  experiment_name="toy_ae_lifted_gr_d${dim}"
  if [ "$FORCE_RETRAIN" != "1" ] && run_dir="$(latest_run_dir "$experiment_name")" && [ -f "$run_dir/checkpoints/best.ckpt" ]; then
    echo "skipping train for ${experiment_name} using ${run_dir}"
    continue
  fi
  "$PYTHON_BIN" -m flatcfm.modelcore.train \
    experiment=toy/ae_lifted_gr \
    "experiment_name=${experiment_name}" \
    "space.projections.0.ambient_dim=${dim}" \
    "space.projections.0.seed=${SEED}" \
    "task.epochs=${EPOCHS}" \
    "task.batch_size=${BATCH_SIZE}" \
    "model.latent_dim=${LATENT_DIM}" \
    "loss.weights.distance=${DISTANCE_WEIGHT}" \
    "trainer.precision=bf16-mixed"
done
