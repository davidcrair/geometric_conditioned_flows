#!/usr/bin/env bash
#SBATCH --job-name=geom_reg_seeds
#SBATCH --output=slurm_logs/slurm-%x-%A_%a.out
#SBATCH --error=slurm_logs/slurm-%x-%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --array=0-9

# replicate the distance loss weight sweep on two additional splitter
# seeds so we have 3 seeds total (42 already trained by
# scripts/train_geom_reg_sweep.sbatch and 43 44 here) this lets the
# thesis plot mean +/- std across seeds for each weight to show the
# null/slightly negative finding is robust to the choice of held out
# drug subset
#
# layout array index -> (seed index 0 or 1, weight index 0..4)
#   task 0..4  seed 43 weights 0 0.001 0.01 0.1 1
#   task 5..9  seed 44 weights 0 0.001 0.01 0.1 1
#
# usage
#   sbatch --partition=gpu_rtx6000 scripts/train_geom_reg_sweep_seeds.sbatch

set -euo pipefail

ROOT_DIR="${FLATCFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$ROOT_DIR"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

AE_BASE_EXPERIMENT="sciplex/ae_log1p_hybrid"
FM_BASE_EXPERIMENT="sciplex/fm_ae_latent_hybrid"

WEIGHTS=(0 0.001 0.01 0.1 1)
TAGS=(w0 w0p001 w0p01 w0p1 w1)
SEEDS=(43 44)

if [ "${SLURM_ARRAY_TASK_ID:-}" = "" ]; then
  echo "error SLURM_ARRAY_TASK_ID not set submit with sbatch" >&2
  exit 1
fi

SEED_IDX=$(( SLURM_ARRAY_TASK_ID / 5 ))
WEIGHT_IDX=$(( SLURM_ARRAY_TASK_ID % 5 ))

SEED="${SEEDS[$SEED_IDX]}"
WEIGHT="${WEIGHTS[$WEIGHT_IDX]}"
TAG="${TAGS[$WEIGHT_IDX]}"

AE_EXP_NAME="sciplex_ae_deg_geom_reg_${TAG}_s${SEED}"
FM_EXP_NAME="sciplex_fm_deg_geom_reg_${TAG}_s${SEED}"

echo "== task $SLURM_ARRAY_TASK_ID seed=$SEED weight=$WEIGHT tag=$TAG =="
echo "AE_EXP_NAME=$AE_EXP_NAME"
echo "FM_EXP_NAME=$FM_EXP_NAME"

# 1 AE training at this distance weight and splitter seed
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

# 2 FM on that AE latent with the same splitter seed so the held out
# drug set is consistent between AE and FM training
srun --chdir="$ROOT_DIR" "$PYTHON_BIN" -m flatcfm.modelcore.train \
  "experiment=$FM_BASE_EXPERIMENT" \
  "experiment_name=$FM_EXP_NAME" \
  "splitter.seed=$SEED" \
  "space.projections.0.artifact_tag=$AE_EXP_NAME" \
  "task.epochs=${FM_EPOCHS:-500}" \
  "task.batch_size=${FM_BATCH_SIZE:-2048}" \
  "task.use_ot_coupling=true" \
  "trainer.num_workers=${NUM_WORKERS:-12}"

# 3 predict held out
FM_RUN_DIR=$(ls -dt "artifacts/runs/${FM_EXP_NAME}/"*/ 2>/dev/null | head -n1)
if [ -z "$FM_RUN_DIR" ]; then
  echo "error could not locate FM run dir after training" >&2
  exit 3
fi
echo "FM_RUN_DIR=$FM_RUN_DIR"
srun --chdir="$ROOT_DIR" "$PYTHON_BIN" -m flatcfm.modelcore.predict \
  "predict.run_dir=$FM_RUN_DIR"

echo "== task $SLURM_ARRAY_TASK_ID done =="
