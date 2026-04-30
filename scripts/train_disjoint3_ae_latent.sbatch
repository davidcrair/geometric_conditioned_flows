#!/usr/bin/env bash
#SBATCH --job-name=disjoint3
#SBATCH --output=slurm_logs/slurm-%x-%A_%a.out
#SBATCH --error=slurm_logs/slurm-%x-%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=5:00:00
#SBATCH --array=0-14

# disjoint 3 cell type partition split sweep across distance loss weights
# and seeds one task per (weight seed) splitter=disjoint3 sets
# split_policy=disjoint_partition with disjoint_partition_cell_types=[K562
# A549 MCF7] and holdout_fraction=0.3 so for each seed the held out set
# is the union of 3 disjoint per cell type buckets each cell type gets
# 30% of its drug universe held out and a given drug is held out for at
# most one cell type (188/188 sciplex drugs are present in all 3 cell
# types so every drug is seen perturbed in at least 2 cell types in
# train)
#
# each task trains
#   1 hybrid AE on the unified disjoint3 split at this distance weight
#   2 FM in that AE latent (sciplex/fm_ae_latent_hybrid)
#   3 ODE in that AE latent (sciplex/ode_ae_latent_hybrid_d1e3)
#   4 predict held out for FM (covers all 3 cell type buckets at once)
#   5 predict held out for ODE
#
# control matching is overridden to held_out so paired controls span
# all 3 cell types in the held out set rather than restricting to a
# single test_cell_type which would mis-match the 3 way evaluation
#
# array layout 15 tasks
#   task = weight_idx*3 + seed_idx
#   weight_idx 0 1 2 3 4 -> 0  1e-4  1e-2  1  1e2
#   seed_idx   0 1 2     -> 42 43 44
#
# usage
#   sbatch --partition=gpu_rtx6000 scripts/train_disjoint3_ae_latent.sbatch
#   # or split across partitions if queue contention
#   sbatch --partition=gpu_rtx6000 --array=0-7  scripts/train_disjoint3_ae_latent.sbatch
#   sbatch --partition=gpu_b200    --array=8-14 scripts/train_disjoint3_ae_latent.sbatch

set -euo pipefail

ROOT_DIR="${FLATCFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$ROOT_DIR"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

AE_BASE_EXPERIMENT="sciplex/ae_log1p_hybrid"
FM_BASE_EXPERIMENT="sciplex/fm_ae_latent_hybrid"
ODE_BASE_EXPERIMENT="sciplex/ode_ae_latent_hybrid_d1e3"

WEIGHTS=(0 0.0001 0.01 1 100)
TAGS=(w0 w0p0001 w0p01 w1 w100)
SEEDS=(42 43 44)

if [ "${SLURM_ARRAY_TASK_ID:-}" = "" ]; then
  echo "error SLURM_ARRAY_TASK_ID not set submit with sbatch" >&2
  exit 1
fi

WEIGHT_IDX=$(( SLURM_ARRAY_TASK_ID / 3 ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % 3 ))

WEIGHT="${WEIGHTS[$WEIGHT_IDX]}"
TAG="${TAGS[$WEIGHT_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"

AE_EXP_NAME="sciplex_ae_disjoint3_${TAG}_s${SEED}"
FM_EXP_NAME="sciplex_fm_disjoint3_${TAG}_s${SEED}"
ODE_EXP_NAME="sciplex_ode_disjoint3_${TAG}_s${SEED}"

echo "== task $SLURM_ARRAY_TASK_ID weight=$WEIGHT tag=$TAG seed=$SEED =="
echo "AE_EXP_NAME=$AE_EXP_NAME"
echo "FM_EXP_NAME=$FM_EXP_NAME"
echo "ODE_EXP_NAME=$ODE_EXP_NAME"

# 1 train AE on the unified disjoint3 split at this distance weight
srun --chdir="$ROOT_DIR" "$PYTHON_BIN" -m flatcfm.modelcore.train \
  "experiment=$AE_BASE_EXPERIMENT" \
  "experiment_name=$AE_EXP_NAME" \
  "space.ae_export_artifact_tag=$AE_EXP_NAME" \
  "splitter=disjoint3" \
  "splitter.seed=$SEED" \
  "loss.weights.distance=$WEIGHT" \
  "task.epochs=${AE_EPOCHS:-300}" \
  "task.batch_size=${AE_BATCH_SIZE:-2048}" \
  "task.lr=${AE_LR:-0.003}" \
  "trainer.num_workers=${NUM_WORKERS:-12}"

# 2 train FM in the AE latent
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

# 3 train ODE in the same AE latent
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

# 4 predict held out for FM (covers all 3 cell type buckets at once)
FM_RUN_DIR=$(ls -dt "artifacts/runs/${FM_EXP_NAME}/"*/ 2>/dev/null | head -n1)
if [ -z "$FM_RUN_DIR" ]; then
  echo "error could not locate FM run dir" >&2
  exit 3
fi
echo "FM_RUN_DIR=$FM_RUN_DIR"
srun --chdir="$ROOT_DIR" "$PYTHON_BIN" -m flatcfm.modelcore.predict \
  "predict.run_dir=$FM_RUN_DIR" \
  "predict.control_source.match_cell_types_from=held_out"

# 5 predict held out for ODE
ODE_RUN_DIR=$(ls -dt "artifacts/runs/${ODE_EXP_NAME}/"*/ 2>/dev/null | head -n1)
if [ -z "$ODE_RUN_DIR" ]; then
  echo "error could not locate ODE run dir" >&2
  exit 4
fi
echo "ODE_RUN_DIR=$ODE_RUN_DIR"
srun --chdir="$ROOT_DIR" "$PYTHON_BIN" -m flatcfm.modelcore.predict \
  "predict.run_dir=$ODE_RUN_DIR" \
  "predict.control_source.match_cell_types_from=held_out"

echo "== task $SLURM_ARRAY_TASK_ID done =="
