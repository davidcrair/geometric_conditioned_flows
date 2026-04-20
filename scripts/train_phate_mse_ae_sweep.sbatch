#!/usr/bin/env bash
#SBATCH --job-name=phate_mse_ae_sweep
#SBATCH --output=slurm_logs/slurm-%x-%A_%a.out
#SBATCH --error=slurm_logs/slurm-%x-%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --array=0-6%3

# train PHATE AE variants with MSE reconstruction at each latent dim for
# the theoretical limits notebook mirrors train_phate_ae_sweep.sbatch
# but overrides model.family=standard so the decoder is MSE rather
# than negative_binomial this lets the notebook compare PHATE MSE AE
# to PHATE NB AE head to head on reconstruction fidelity
#
# artifact tags are sciplex_ae_deg_phate_mse_d{DIM} matching the
# notebook's phate_mse variant
#
# usage
#   sbatch --partition=gpu_b200 scripts/train_phate_mse_ae_sweep.sbatch
#   (rtx6000 or h200 work too)

set -euo pipefail

ROOT_DIR="${FLATCFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$ROOT_DIR"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

DIMS=(16 32 64 128 256 512 1024)

if [ "${SLURM_ARRAY_TASK_ID:-}" = "" ]; then
  echo "error SLURM_ARRAY_TASK_ID not set" >&2
  exit 1
fi

DIM="${DIMS[$SLURM_ARRAY_TASK_ID]}"
AE_EXP_NAME="sciplex_ae_deg_phate_mse_d${DIM}"

echo "== task $SLURM_ARRAY_TASK_ID dim=$DIM =="
echo "AE_EXP_NAME=$AE_EXP_NAME"

srun --chdir="$ROOT_DIR" "$PYTHON_BIN" -m flatcfm.modelcore.train \
  experiment=sciplex/ae_log1p_phate \
  "experiment_name=$AE_EXP_NAME" \
  "space.ae_export_artifact_tag=$AE_EXP_NAME" \
  "model.family=standard" \
  "model.latent_dim=$DIM" \
  "loss.distance_zeta=${ZETA:-0.007}" \
  "loss.distance_alpha_min=${ALPHA_MIN:-0.1}" \
  "task.epochs=${TASK_EPOCHS:-200}" \
  "task.batch_size=${BATCH_SIZE:-2048}" \
  "trainer.num_workers=${NUM_WORKERS:-10}"

echo "== task $SLURM_ARRAY_TASK_ID done =="
