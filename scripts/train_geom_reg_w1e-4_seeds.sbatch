#!/usr/bin/env bash
#SBATCH --job-name=geom_reg_w1e-4
#SBATCH --output=slurm_logs/slurm-%x-%A_%a.out
#SBATCH --error=slurm_logs/slurm-%x-%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --array=0-2

# 3 seeds at distance weight = 1e-4 to probe whether the training-
# stability benefit of distance regularization kicks in at very small
# weights or only at w >= 0.001 fills in the sweep gap below the
# existing w=0.001 point
#
# task 0  seed 42 weight 1e-4
# task 1  seed 43 weight 1e-4
# task 2  seed 44 weight 1e-4
#
# usage (pick your partition)
#   sbatch --partition=gpu_rtx6000 scripts/train_geom_reg_w1e-4_seeds.sbatch
#   sbatch --partition=gpu_b200    scripts/train_geom_reg_w1e-4_seeds.sbatch

set -euo pipefail

ROOT_DIR="${FLATCFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$ROOT_DIR"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

AE_BASE_EXPERIMENT="sciplex/ae_log1p_hybrid"
FM_BASE_EXPERIMENT="sciplex/fm_ae_latent_hybrid"

SEEDS=(42 43 44)
WEIGHT=0.0001
TAG=w0p0001

if [ "${SLURM_ARRAY_TASK_ID:-}" = "" ]; then
  echo "error SLURM_ARRAY_TASK_ID not set" >&2
  exit 1
fi

SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
# seed 42 is the default (no suffix) to match the existing convention
if [ "$SEED" = "42" ]; then
  SUFFIX=""
else
  SUFFIX="_s${SEED}"
fi
AE_EXP_NAME="sciplex_ae_deg_geom_reg_${TAG}${SUFFIX}"
FM_EXP_NAME="sciplex_fm_deg_geom_reg_${TAG}${SUFFIX}"

echo "== task $SLURM_ARRAY_TASK_ID seed=$SEED weight=$WEIGHT =="
echo "AE_EXP_NAME=$AE_EXP_NAME"
echo "FM_EXP_NAME=$FM_EXP_NAME"

srun --chdir="$ROOT_DIR" "$PYTHON_BIN" -m flatcfm.modelcore.train \
  "experiment=$AE_BASE_EXPERIMENT" \
  "experiment_name=$AE_EXP_NAME" \
  "space.ae_export_artifact_tag=$AE_EXP_NAME" \
  "splitter.seed=$SEED" \
  "loss.weights.distance=$WEIGHT" \
  "task.epochs=${AE_EPOCHS:-300}" \
  "task.batch_size=${AE_BATCH_SIZE:-2048}" \
  "task.lr=${AE_LR:-0.003}" \
  "trainer.num_workers=${NUM_WORKERS:-12}"

srun --chdir="$ROOT_DIR" "$PYTHON_BIN" -m flatcfm.modelcore.train \
  "experiment=$FM_BASE_EXPERIMENT" \
  "experiment_name=$FM_EXP_NAME" \
  "splitter.seed=$SEED" \
  "space.projections.0.artifact_tag=$AE_EXP_NAME" \
  "task.epochs=${FM_EPOCHS:-500}" \
  "task.batch_size=${FM_BATCH_SIZE:-2048}" \
  "task.use_ot_coupling=true" \
  "trainer.num_workers=${NUM_WORKERS:-12}"

FM_RUN_DIR=$(ls -dt "artifacts/runs/${FM_EXP_NAME}/"*/ 2>/dev/null | head -n1)
if [ -z "$FM_RUN_DIR" ]; then
  echo "error could not locate FM run dir" >&2
  exit 3
fi
echo "FM_RUN_DIR=$FM_RUN_DIR"
srun --chdir="$ROOT_DIR" "$PYTHON_BIN" -m flatcfm.modelcore.predict \
  "predict.run_dir=$FM_RUN_DIR"

echo "== task $SLURM_ARRAY_TASK_ID done =="
