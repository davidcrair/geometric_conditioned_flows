#!/usr/bin/env bash
#SBATCH --job-name=phate_ae_sweep
#SBATCH --output=slurm_logs/slurm-%x-%A_%a.out
#SBATCH --error=slurm_logs/slurm-%x-%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --array=0-6%3

# train PHATE AE variants at each latent dim for the theoretical limits
# notebook uses the corrected phate diffusion potential distance loss
# (the legacy AEs at d=16..256 were trained against the broken 2D PHATE
# target) artifact tags match what view_theoretical_limits.ipynb expects
# (sciplex_ae_deg_phate_d{DIM}) so EXISTING PKLS ARE OVERWRITTEN
#
# to fill only the missing dims (d=512 d=1024):
#   sbatch --partition=gpu_rtx6000 --array=5-6 scripts/train_phate_ae_sweep.sbatch
# to retrain all 7:
#   sbatch --partition=gpu_rtx6000 scripts/train_phate_ae_sweep.sbatch

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
AE_EXP_NAME="sciplex_ae_deg_phate_d${DIM}"

echo "== task $SLURM_ARRAY_TASK_ID dim=$DIM =="
echo "AE_EXP_NAME=$AE_EXP_NAME"

# use the ae_log1p_phate base experiment (two phase PHATE AE) with
# diffpot scale distance hyperparameters override the model latent_dim
# and tag to match the notebook's expected artifact name
srun --chdir="$ROOT_DIR" "$PYTHON_BIN" -m flatcfm.modelcore.train \
  experiment=sciplex/ae_log1p_phate \
  "experiment_name=$AE_EXP_NAME" \
  "space.ae_export_artifact_tag=$AE_EXP_NAME" \
  "model.latent_dim=$DIM" \
  "loss.distance_zeta=${ZETA:-0.007}" \
  "loss.distance_alpha_min=${ALPHA_MIN:-0.1}" \
  "task.epochs=${TASK_EPOCHS:-200}" \
  "task.batch_size=${BATCH_SIZE:-2048}" \
  "trainer.num_workers=${NUM_WORKERS:-10}"

echo "== task $SLURM_ARRAY_TASK_ID done =="
