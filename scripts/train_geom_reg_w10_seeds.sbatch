#!/usr/bin/env bash
#SBATCH --job-name=geom_reg_w10
#SBATCH --output=slurm_logs/slurm-%x-%A_%a.out
#SBATCH --error=slurm_logs/slurm-%x-%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --array=0-2

# 3 seeds at w=10 to push above w=1 tests whether heavier distance
# regularization starts to hurt downstream metrics or continues the
# W1-degrades W2-improves trend
#
# usage  sbatch --partition=gpu_b200 scripts/train_geom_reg_w10_seeds.sbatch

set -euo pipefail

ROOT_DIR="${FLATCFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$ROOT_DIR"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

AE_BASE_EXPERIMENT="sciplex/ae_log1p_hybrid"
FM_BASE_EXPERIMENT="sciplex/fm_ae_latent_hybrid"

SEEDS=(42 43 44)
WEIGHT=10
TAG=w10

if [ "${SLURM_ARRAY_TASK_ID:-}" = "" ]; then
  echo "error SLURM_ARRAY_TASK_ID not set" >&2; exit 1
fi

SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
if [ "$SEED" = "42" ]; then SUFFIX=""; else SUFFIX="_s${SEED}"; fi
AE_EXP_NAME="sciplex_ae_deg_geom_reg_${TAG}${SUFFIX}"
FM_EXP_NAME="sciplex_fm_deg_geom_reg_${TAG}${SUFFIX}"

echo "== task $SLURM_ARRAY_TASK_ID seed=$SEED weight=$WEIGHT =="

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
[ -z "$FM_RUN_DIR" ] && { echo "error: no FM run dir" >&2; exit 3; }
srun --chdir="$ROOT_DIR" "$PYTHON_BIN" -m flatcfm.modelcore.predict \
  "predict.run_dir=$FM_RUN_DIR"
echo "== done =="
