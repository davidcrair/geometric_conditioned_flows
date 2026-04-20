#!/usr/bin/env bash
# shared helpers for training and prediction scripts
# source this file from other scripts: source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

latest_run_dir() {
  local experiment_name="$1"
  local run_root="artifacts/runs/${experiment_name}"
  if [ ! -d "$run_root" ]; then
    return 1
  fi
  ls -td "$run_root"/* 2>/dev/null | head -n 1
}

ensure_train() {
  local experiment="$1"
  local experiment_name="$2"
  shift 2

  local run_dir=""
  if [ "$FORCE_RETRAIN" != "1" ] && run_dir="$(latest_run_dir "$experiment_name")" && [ -f "$run_dir/checkpoints/best.ckpt" ]; then
    echo "skipping train for ${experiment_name} using ${run_dir}"
    return
  fi

  "$PYTHON_BIN" -m flatcfm.modelcore.train \
    "experiment=${experiment}" \
    "experiment_name=${experiment_name}" \
    "trainer.precision=bf16-mixed" \
    "$@"
}

ensure_predict() {
  local experiment_name="$1"
  local run_dir="$2"

  if [ "${FORCE_PREDICT:-0}" != "1" ] && [ -f "$run_dir/predictions/held_out/predictions.h5ad" ]; then
    echo "skipping predict for ${experiment_name} using ${run_dir}"
    return
  fi

  "$PYTHON_BIN" -m flatcfm.modelcore.predict "predict.run_dir=${run_dir}"
}
