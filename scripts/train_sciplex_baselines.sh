#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
FORCE_RETRAIN="${FORCE_RETRAIN:-0}"

export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

BASELINES=(
  "sciplex/baseline_no_effect"
  "sciplex/baseline_additive"
  "sciplex/baseline_context_mean"
  "sciplex/baseline_perturb_mean"
)

for baseline in "${BASELINES[@]}"; do
  experiment_name="sciplex_deg_${baseline##*/}"
  ensure_train "$baseline" "$experiment_name"
  run_dir="$(latest_run_dir "$experiment_name")"
  ensure_predict "$experiment_name" "$run_dir"
  echo "trained+predicted ${experiment_name} -> ${run_dir}"
done

# learned baselines train for real (100 epochs) so they go last
ensure_train "sciplex/baseline_decoder" "sciplex_deg_baseline_decoder"
decoder_run_dir="$(latest_run_dir "sciplex_deg_baseline_decoder")"
ensure_predict "sciplex_deg_baseline_decoder" "$decoder_run_dir"
echo "trained+predicted sciplex_deg_baseline_decoder -> ${decoder_run_dir}"

ensure_train "sciplex/baseline_linear" "sciplex_deg_baseline_linear"
linear_run_dir="$(latest_run_dir "sciplex_deg_baseline_linear")"
ensure_predict "sciplex_deg_baseline_linear" "$linear_run_dir"
echo "trained+predicted sciplex_deg_baseline_linear -> ${linear_run_dir}"

echo ""
echo "=== all baselines trained and predicted ==="
echo "run dirs for view_benchmark_results.ipynb:"
for baseline in "${BASELINES[@]}"; do
  experiment_name="sciplex_deg_${baseline##*/}"
  echo "  ${experiment_name}: $(latest_run_dir "$experiment_name")"
done
echo "  sciplex_deg_baseline_decoder: ${decoder_run_dir}"
echo "  sciplex_deg_baseline_linear: ${linear_run_dir}"
