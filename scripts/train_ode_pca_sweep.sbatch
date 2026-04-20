#!/usr/bin/env bash
#SBATCH --job-name=ode_pca_sweep
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=12:00:00

# ODE PCA dimension sweep (standalone FlatCFM pipeline)
# trains sciplex ODE at a single PCA dim then generates held-out predictions
#
# usage (one submission per missing dim):
#   sbatch --partition=scavenge_gpu --export=ALL,DIM=16 scripts/train_ode_pca_sweep.sbatch
#
# or use scripts/check_ode_pca_sweep.py --sbatch to print all missing submissions

set -euo pipefail

ROOT_DIR="${FLATCFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$ROOT_DIR"
echo "ROOT_DIR=$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

if [ -z "${DIM:-}" ]; then
  echo "error: DIM env var not set (e.g. --export=ALL,DIM=16)" >&2
  exit 1
fi

BASE_EXPERIMENT="sciplex/ode_pca_deg"
EXP_NAME="sciplex_ode_deg_pca_d${DIM}"

echo "DIM=$DIM"
echo "EXP_NAME=$EXP_NAME"

srun --chdir="$ROOT_DIR" "$PYTHON_BIN" -m flatcfm.modelcore.train \
  "experiment=$BASE_EXPERIMENT" \
  "experiment_name=$EXP_NAME" \
  "space.projections.0.n_components=$DIM" \
  "trainer.num_workers=8"

RUN_DIR=$(ls -dt "artifacts/runs/${EXP_NAME}/"*/ 2>/dev/null | head -n1)
if [ -z "$RUN_DIR" ]; then
  echo "error: could not locate run dir after training" >&2
  exit 2
fi
echo "RUN_DIR=$RUN_DIR"

srun --chdir="$ROOT_DIR" "$PYTHON_BIN" -m flatcfm.modelcore.predict \
  "predict.run_dir=$RUN_DIR"

echo "== done DIM=$DIM =="
