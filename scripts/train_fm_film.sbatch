#!/usr/bin/env bash
#SBATCH --job-name=fm_film
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --array=0-1

# FM with FiLM conditioning vs concat baseline in narrow MSE AE d=128
#
# usage:
#   sbatch --partition=gpu_rtx6000 scripts/train_fm_film.sbatch

set -euo pipefail

ROOT_DIR="${FLATCFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$ROOT_DIR"
echo "ROOT_DIR=$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

FM_BASE_EXPERIMENT="sciplex/fm_ae_latent_mse_d512"
AE_TAG="sciplex_ae_deg_mse_d128"

# each entry: "experiment_name  conditioning  extra_overrides"
CONFIGS=(
  "sciplex_fm_deg_film_d128    film    "
  "sciplex_fm_deg_concat_d128  concat  "
)

if [ "${SLURM_ARRAY_TASK_ID:-}" = "" ]; then
  echo "error: SLURM_ARRAY_TASK_ID not set -- submit with sbatch" >&2
  exit 1
fi

entry="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
read -r fm_exp_name conditioning <<< "$entry"

echo "== task $SLURM_ARRAY_TASK_ID =="
echo "ae_tag=$AE_TAG"
echo "conditioning=$conditioning"
echo "fm_exp_name=$fm_exp_name"

srun --chdir="$ROOT_DIR" "$PYTHON_BIN" -m flatcfm.modelcore.train \
  "experiment=$FM_BASE_EXPERIMENT" \
  "experiment_name=$fm_exp_name" \
  "space.projections.0.artifact_tag=$AE_TAG" \
  "model.conditioning=$conditioning" \
  "task.epochs=500" \
  "trainer.num_workers=12"

RUN_DIR=$(ls -dt "artifacts/runs/${fm_exp_name}/"*/ 2>/dev/null | head -n1)
if [ -z "$RUN_DIR" ]; then
  echo "error: could not locate run dir after training" >&2
  exit 3
fi
echo "RUN_DIR=$RUN_DIR"

srun --chdir="$ROOT_DIR" "$PYTHON_BIN" -m flatcfm.modelcore.predict \
  "predict.run_dir=$RUN_DIR"

echo "== task $SLURM_ARRAY_TASK_ID done =="
