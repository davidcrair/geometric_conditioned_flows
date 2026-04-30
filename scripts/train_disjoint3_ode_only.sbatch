#!/usr/bin/env bash
#SBATCH --job-name=disjoint3_ode
#SBATCH --output=slurm_logs/slurm-%x-%A_%a.out
#SBATCH --error=slurm_logs/slurm-%x-%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --array=0-2

# ODE only on the existing disjoint3 AE projections trains ODE in the AE
# latent then predicts held out skips AE and FM entirely so this is the
# rescue script when only AE has been trained for some seeds defaults to
# weight=0 seeds 42 43 44 override WEIGHT and TAG env vars or array
# range to target other (weight seed) combos
#
# array layout 3 tasks one per seed
#   task 0 -> seed 42
#   task 1 -> seed 43
#   task 2 -> seed 44
#
# usage
#   sbatch --partition=gpu_rtx6000 scripts/train_disjoint3_ode_only.sbatch
#   # or for a different weight that also already has AEs trained
#   WEIGHT=0.01 TAG=w0p01 sbatch --partition=gpu_rtx6000 scripts/train_disjoint3_ode_only.sbatch

set -euo pipefail

ROOT_DIR="${FLATCFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$ROOT_DIR"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

ODE_BASE_EXPERIMENT="sciplex/ode_ae_latent_hybrid_d1e3"

WEIGHT="${WEIGHT:-0}"
TAG="${TAG:-w0}"
SEEDS=(42 43 44)

if [ "${SLURM_ARRAY_TASK_ID:-}" = "" ]; then
  echo "error SLURM_ARRAY_TASK_ID not set submit with sbatch" >&2
  exit 1
fi

SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

AE_EXP_NAME="sciplex_ae_disjoint3_${TAG}_s${SEED}"
ODE_EXP_NAME="sciplex_ode_disjoint3_${TAG}_s${SEED}"

echo "== task $SLURM_ARRAY_TASK_ID weight=$WEIGHT tag=$TAG seed=$SEED =="
echo "AE_EXP_NAME=$AE_EXP_NAME"
echo "ODE_EXP_NAME=$ODE_EXP_NAME"

# verify AE projection exists before training ODE
if ! ls "artifacts/spaces/sciplex_ae_projection_"*"_${AE_EXP_NAME}.pkl" 1>/dev/null 2>&1; then
  echo "error no AE projection pkl for $AE_EXP_NAME train AE first" >&2
  exit 2
fi

# 1 train ODE in the AE latent
srun --chdir="$ROOT_DIR" "$PYTHON_BIN" -m flatcfm.modelcore.train \
  "experiment=$ODE_BASE_EXPERIMENT" \
  "experiment_name=$ODE_EXP_NAME" \
  "splitter=disjoint3" \
  "splitter.seed=$SEED" \
  "space.projections.0.artifact_tag=$AE_EXP_NAME" \
  "task.epochs=${ODE_EPOCHS:-500}" \
  "task.batch_size=${ODE_BATCH_SIZE:-2048}" \
  "predict.control_source.match_cell_types_from=held_out" \
  "trainer.num_workers=${NUM_WORKERS:-12}"

# 2 predict held out for ODE (covers all 3 cell type buckets at once)
ODE_RUN_DIR=$(ls -dt "artifacts/runs/${ODE_EXP_NAME}/"*/ 2>/dev/null | head -n1)
if [ -z "$ODE_RUN_DIR" ]; then
  echo "error could not locate ODE run dir" >&2
  exit 3
fi
echo "ODE_RUN_DIR=$ODE_RUN_DIR"
srun --chdir="$ROOT_DIR" "$PYTHON_BIN" -m flatcfm.modelcore.predict \
  "predict.run_dir=$ODE_RUN_DIR" \
  "predict.control_source.match_cell_types_from=held_out"

echo "== task $SLURM_ARRAY_TASK_ID done =="
