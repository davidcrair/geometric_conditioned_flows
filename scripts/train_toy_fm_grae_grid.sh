#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-100}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-50}"
BATCH_SIZE="${BATCH_SIZE:-128}"
FORCE_RETRAIN="${FORCE_RETRAIN:-0}"
FORCE_PREDICT="${FORCE_PREDICT:-0}"

if [ "$#" -gt 0 ]; then
  DIMS=("$@")
else
  DIMS=(2 8 16 512)
fi

export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

for dim in "${DIMS[@]}"; do
  experiment_name="toy_fm_lifted_ae_latent_d${dim}"
  run_dir=""
  if [ "$FORCE_RETRAIN" != "1" ] && run_dir="$(latest_run_dir "$experiment_name")" && [ -f "$run_dir/checkpoints/best.ckpt" ]; then
    echo "skipping train for ${experiment_name} using ${run_dir}"
  else
    "$PYTHON_BIN" -m flatcfm.modelcore.train \
      experiment=toy/fm_lifted_ae_latent \
      "experiment_name=${experiment_name}" \
      "space.projections.0.ambient_dim=${dim}" \
      "space.projections.0.seed=${SEED}" \
      "task.epochs=${EPOCHS}" \
      "task.steps_per_epoch=${STEPS_PER_EPOCH}" \
      "task.batch_size=${BATCH_SIZE}" \
      "trainer.precision=bf16-mixed"
    run_dir="$(latest_run_dir "$experiment_name")"
  fi

  if [ "$FORCE_PREDICT" != "1" ] && [ -f "$run_dir/predictions/held_out/predictions.h5ad" ]; then
    echo "skipping predict for ${experiment_name} using ${run_dir}"
  else
    "$PYTHON_BIN" -m flatcfm.modelcore.predict "predict.run_dir=${run_dir}"
  fi
done
