#!/usr/bin/env bash
#SBATCH --job-name=geom_reg_sweep
#SBATCH --output=slurm_logs/slurm-%x-%A_%a.out
#SBATCH --error=slurm_logs/slurm-%x-%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --array=0-4

# distance loss weight sweep for the MSE+dist autoencoder run with the
# corrected phate diffusion potential distance loss each array task
# trains the AE at one weight then trains FM on the resulting latent
# and runs predict on the held out split so everything lands in place
# for check_geom_reg_sweep.py and downstream eval
#
# usage:
#   sbatch --partition=scavenge_gpu scripts/train_geom_reg_sweep.sbatch
#   # or non preemptible:
#   sbatch --partition=gpu_b200 scripts/train_geom_reg_sweep.sbatch

set -euo pipefail

ROOT_DIR="${FLATCFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$ROOT_DIR"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

AE_BASE_EXPERIMENT="sciplex/ae_log1p_hybrid"
FM_BASE_EXPERIMENT="sciplex/fm_ae_latent_hybrid"

# distance loss weights paired with file safe tags (dots replaced by p)
# these must match the tags that check_geom_reg_sweep.py looks for so
# the status table can find the trained artifacts
WEIGHTS=(0 0.001 0.01 0.1 1)
TAGS=(w0 w0p001 w0p01 w0p1 w1)

if [ "${SLURM_ARRAY_TASK_ID:-}" = "" ]; then
  echo "error: SLURM_ARRAY_TASK_ID not set submit with sbatch" >&2
  exit 1
fi

WEIGHT="${WEIGHTS[$SLURM_ARRAY_TASK_ID]}"
TAG="${TAGS[$SLURM_ARRAY_TASK_ID]}"
AE_EXP_NAME="sciplex_ae_deg_geom_reg_${TAG}"
FM_EXP_NAME="sciplex_fm_deg_geom_reg_${TAG}"

echo "== task $SLURM_ARRAY_TASK_ID weight=$WEIGHT tag=$TAG =="
echo "AE_EXP_NAME=$AE_EXP_NAME"
echo "FM_EXP_NAME=$FM_EXP_NAME"

# 1 train AE at this distance weight
srun --chdir="$ROOT_DIR" "$PYTHON_BIN" -m flatcfm.modelcore.train \
  "experiment=$AE_BASE_EXPERIMENT" \
  "experiment_name=$AE_EXP_NAME" \
  "space.ae_export_artifact_tag=$AE_EXP_NAME" \
  "loss.weights.distance=$WEIGHT" \
  "task.epochs=${AE_EPOCHS:-300}" \
  "task.batch_size=${AE_BATCH_SIZE:-2048}" \
  "task.lr=${AE_LR:-0.003}" \
  "trainer.num_workers=${NUM_WORKERS:-12}"

# 2 train FM on that AE latent
srun --chdir="$ROOT_DIR" "$PYTHON_BIN" -m flatcfm.modelcore.train \
  "experiment=$FM_BASE_EXPERIMENT" \
  "experiment_name=$FM_EXP_NAME" \
  "space.projections.0.artifact_tag=$AE_EXP_NAME" \
  "task.epochs=${FM_EPOCHS:-500}" \
  "task.batch_size=${FM_BATCH_SIZE:-2048}" \
  "task.use_ot_coupling=true" \
  "trainer.num_workers=${NUM_WORKERS:-12}"

# 3 predict held out using that FM run
FM_RUN_DIR=$(ls -dt "artifacts/runs/${FM_EXP_NAME}/"*/ 2>/dev/null | head -n1)
if [ -z "$FM_RUN_DIR" ]; then
  echo "error: could not locate FM run dir after training" >&2
  exit 3
fi
echo "FM_RUN_DIR=$FM_RUN_DIR"
srun --chdir="$ROOT_DIR" "$PYTHON_BIN" -m flatcfm.modelcore.predict \
  "predict.run_dir=$FM_RUN_DIR"

echo "== task $SLURM_ARRAY_TASK_ID done =="
