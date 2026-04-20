#!/usr/bin/env bash
#SBATCH --job-name=ode_w0_xtra
#SBATCH --output=slurm_logs/slurm-%x-%A_%a.out
#SBATCH --error=slurm_logs/slurm-%x-%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --array=0-2

# neural ODE at w=0 on additional seeds to increase the n for the
# no distance regularization baseline takes SEEDS as a space separated
# env var submit with --export=ALL,SEEDS="45 46 47" to run 3 seeds
#
# usage
#   sbatch --partition=gpu_rtx6000 --export=ALL,SEEDS="45 46 47" \
#     scripts/train_ode_geom_reg_w0_extra_seeds.sbatch
#   sbatch --partition=gpu_b200 --export=ALL,SEEDS="48 49 50" \
#     scripts/train_ode_geom_reg_w0_extra_seeds.sbatch

set -euo pipefail

ROOT_DIR="${FLATCFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$ROOT_DIR"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

ODE_BASE_EXPERIMENT="sciplex/ode_ae_latent_hybrid_d1e3"

if [ -z "${SEEDS:-}" ]; then
  echo "error SEEDS not set pass via sbatch --export=ALL,SEEDS=\"45 46 47\"" >&2
  exit 1
fi
read -ra SEED_ARR <<<"$SEEDS"
if [ "${#SEED_ARR[@]}" -ne 3 ]; then
  echo "error SEEDS must contain exactly 3 values got ${#SEED_ARR[@]}" >&2
  exit 1
fi

TAG=w0

if [ "${SLURM_ARRAY_TASK_ID:-}" = "" ]; then
  echo "error SLURM_ARRAY_TASK_ID not set" >&2; exit 1
fi

SEED="${SEED_ARR[$SLURM_ARRAY_TASK_ID]}"
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
