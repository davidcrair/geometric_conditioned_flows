#!/usr/bin/env bash
#SBATCH --job-name=disjoint3_fm
#SBATCH --output=slurm_logs/slurm-%x-%A_%a.out
#SBATCH --error=slurm_logs/slurm-%x-%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --array=0-2

# FM only on the existing disjoint3 AE projections trains FM in the AE
# latent then predicts held out skips AE and ODE entirely so this is the
# rescue script when the main sbatch only completed AE for some seeds
# defaults to weight=0 seeds 42 43 44 override WEIGHT and SEEDS env vars
# or array range to target other (weight seed) combos
#
# array layout 3 tasks one per seed
#   task 0 -> seed 42
#   task 1 -> seed 43
#   task 2 -> seed 44
#
# usage
#   sbatch --partition=gpu_rtx6000 scripts/train_disjoint3_fm_only.sbatch
#   # or for a different weight that also already has AEs trained
#   WEIGHT=0.01 TAG=w0p01 sbatch --partition=gpu_rtx6000 scripts/train_disjoint3_fm_only.sbatch

set -euo pipefail

ROOT_DIR="${FLATCFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$ROOT_DIR"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

FM_BASE_EXPERIMENT="sciplex/fm_ae_latent_hybrid"

WEIGHT="${WEIGHT:-0}"
TAG="${TAG:-w0}"
SEEDS=(42 43 44)

if [ "${SLURM_ARRAY_TASK_ID:-}" = "" ]; then
  echo "error SLURM_ARRAY_TASK_ID not set submit with sbatch" >&2
  exit 1
fi

SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

AE_EXP_NAME="sciplex_ae_disjoint3_${TAG}_s${SEED}"
FM_EXP_NAME="sciplex_fm_disjoint3_${TAG}_s${SEED}"

echo "== task $SLURM_ARRAY_TASK_ID weight=$WEIGHT tag=$TAG seed=$SEED =="
echo "AE_EXP_NAME=$AE_EXP_NAME"
echo "FM_EXP_NAME=$FM_EXP_NAME"

# verify AE projection exists before training FM otherwise FM will fail
# late and waste a slurm slot
if ! ls "artifacts/spaces/sciplex_ae_projection_"*"_${AE_EXP_NAME}.pkl" 1>/dev/null 2>&1; then
  echo "error no AE projection pkl for $AE_EXP_NAME train AE first" >&2
  exit 2
fi

# 1 train FM in the AE latent
srun --chdir="$ROOT_DIR" "$PYTHON_BIN" -m flatcfm.modelcore.train \
  "experiment=$FM_BASE_EXPERIMENT" \
  "experiment_name=$FM_EXP_NAME" \
  "splitter=disjoint3" \
  "splitter.seed=$SEED" \
  "space.projections.0.artifact_tag=$AE_EXP_NAME" \
  "task.epochs=${FM_EPOCHS:-500}" \
  "task.batch_size=${FM_BATCH_SIZE:-2048}" \
  "task.use_ot_coupling=true" \
  "predict.control_source.match_cell_types_from=held_out" \
  "trainer.num_workers=${NUM_WORKERS:-12}"

# 2 predict held out for FM (covers all 3 cell type buckets at once)
FM_RUN_DIR=$(ls -dt "artifacts/runs/${FM_EXP_NAME}/"*/ 2>/dev/null | head -n1)
if [ -z "$FM_RUN_DIR" ]; then
  echo "error could not locate FM run dir" >&2
  exit 3
fi
echo "FM_RUN_DIR=$FM_RUN_DIR"
srun --chdir="$ROOT_DIR" "$PYTHON_BIN" -m flatcfm.modelcore.predict \
  "predict.run_dir=$FM_RUN_DIR" \
  "predict.control_source.match_cell_types_from=held_out"

echo "== task $SLURM_ARRAY_TASK_ID done =="
