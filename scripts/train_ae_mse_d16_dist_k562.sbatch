#!/usr/bin/env bash
#SBATCH --job-name=ae_mse_d16_dist_k562
#SBATCH --output=slurm_logs/slurm-%x-%A_%a.out
#SBATCH --error=slurm_logs/slurm-%x-%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=12:00:00

# train k562 mse d16 autoencoder with phate diffusion potential distance
# loss mirrors the d128 run hyperparameters so the d16 vs d128 comparison
# is apples to apples
#
# usage:
#   sbatch --partition=gpu_h200 scripts/train_ae_mse_d16_dist_k562.sbatch
#   # or scavenge:
#   sbatch --partition=scavenge_gpu scripts/train_ae_mse_d16_dist_k562.sbatch

set -euo pipefail

ROOT_DIR="${FLATCFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$ROOT_DIR"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

srun --chdir="$ROOT_DIR" "$PYTHON_BIN" -m flatcfm.modelcore.train \
  experiment=sciplex/ae_log1p_mse_d16_dist \
  splitter=strict_k562 \
  experiment_name=sciplex_ae_deg_log1p_mse_d16_dist_k562 \
  task.epochs="${TASK_EPOCHS:-300}" \
  task.batch_size="${BATCH_SIZE:-2048}" \
  task.lr="${LR:-0.003}" \
  loss.distance_alpha_min="${ALPHA_MIN:-0.1}" \
  loss.distance_zeta="${ZETA:-0.007}" \
  trainer.num_workers="${NUM_WORKERS:-10}"

echo "== done =="
