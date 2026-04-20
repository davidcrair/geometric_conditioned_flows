#!/usr/bin/env bash
#SBATCH --job-name=ode_w10
#SBATCH --output=slurm_logs/slurm-%x-%A_%a.out
#SBATCH --error=slurm_logs/slurm-%x-%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --array=0-2

# neural ODE at w=10 in the geom_reg MSE AE latent 3 seeds reuses the
# already trained AE projections so no AE training needed fills in the
# FM best operating point comparison for ODE
#
# usage  sbatch --partition=gpu_rtx6000 scripts/train_ode_geom_reg_w10_seeds.sbatch

set -euo pipefail

ROOT_DIR="${FLATCFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$ROOT_DIR"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

ODE_BASE_EXPERIMENT="sciplex/ode_ae_latent_hybrid_d1e3"

SEEDS=(42 43 44)
TAG=w10

if [ "${SLURM_ARRAY_TASK_ID:-}" = "" ]; then
  echo "error SLURM_ARRAY_TASK_ID not set" >&2; exit 1
fi

SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
if [ "$SEED" = "42" ]; then SUFFIX=""; else SUFFIX="_s${SEED}"; fi
AE_TAG="sciplex_ae_deg_geom_reg_${TAG}${SUFFIX}"
ODE_EXP_NAME="sciplex_ode_deg_geom_reg_${TAG}${SUFFIX}"

echo "== task $SLURM_ARRAY_TASK_ID seed=$SEED weight_tag=$TAG =="
echo "AE_TAG=$AE_TAG  ODE_EXP_NAME=$ODE_EXP_NAME"

if ! ls "artifacts/spaces/sciplex_ae_projection_"*"_${AE_TAG}.pkl" 1>/dev/null 2>&1; then
  echo "error no AE projection pkl for $AE_TAG" >&2
  exit 2
fi

srun --chdir="$ROOT_DIR" "$PYTHON_BIN" -m flatcfm.modelcore.train \
  "experiment=$ODE_BASE_EXPERIMENT" \
  "experiment_name=$ODE_EXP_NAME" \
  "splitter.seed=$SEED" \
  "space.projections.0.artifact_tag=$AE_TAG" \
  "task.epochs=${ODE_EPOCHS:-500}" \
  "task.batch_size=${ODE_BATCH_SIZE:-2048}" \
  "trainer.num_workers=${NUM_WORKERS:-12}"

RUN_DIR=$(ls -dt "artifacts/runs/${ODE_EXP_NAME}/"*/ 2>/dev/null | head -n1)
if [ -z "$RUN_DIR" ]; then
  echo "error could not locate ODE run dir" >&2; exit 3
fi
echo "RUN_DIR=$RUN_DIR"

srun --chdir="$ROOT_DIR" "$PYTHON_BIN" -m flatcfm.modelcore.predict \
  "predict.run_dir=$RUN_DIR"

echo "== task $SLURM_ARRAY_TASK_ID done =="
