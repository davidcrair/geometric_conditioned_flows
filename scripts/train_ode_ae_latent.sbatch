#!/usr/bin/env bash
#SBATCH --job-name=ode_ae_latent
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=12:00:00

# retrain ODE models in AE latent spaces using the current debiased sinkhorn
# loss (prior March runs predate the geomloss swap and produced nan losses)
#
# one submission per experiment by EXPERIMENT env var
#
# usage:
#   sbatch --partition=scavenge_gpu \
#     --export=ALL,EXPERIMENT=sciplex/ode_ae_latent_phate \
#     scripts/train_ode_ae_latent.sbatch
#   sbatch --partition=scavenge_gpu \
#     --export=ALL,EXPERIMENT=sciplex/ode_ae_latent_mse_d512 \
#     scripts/train_ode_ae_latent.sbatch

set -euo pipefail

ROOT_DIR="${FLATCFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$ROOT_DIR"
echo "ROOT_DIR=$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

if [ -z "${EXPERIMENT:-}" ]; then
  echo "error: EXPERIMENT env var not set (e.g. --export=ALL,EXPERIMENT=sciplex/ode_ae_latent_phate)" >&2
  exit 1
fi

# derive experiment_name from hydra (the config sets experiment_name) so we
# can locate the run dir after training. parse once from the config yaml
EXP_YAML="$ROOT_DIR/src/flatcfm/configs/experiment/${EXPERIMENT}.yaml"
if [ ! -f "$EXP_YAML" ]; then
  echo "error: experiment config not found: $EXP_YAML" >&2
  exit 2
fi
EXP_NAME=$(awk -F': ' '/^experiment_name:/ {print $2; exit}' "$EXP_YAML")
if [ -z "$EXP_NAME" ]; then
  echo "error: could not parse experiment_name from $EXP_YAML" >&2
  exit 3
fi

echo "EXPERIMENT=$EXPERIMENT"
echo "EXP_NAME=$EXP_NAME"

srun --chdir="$ROOT_DIR" "$PYTHON_BIN" -m flatcfm.modelcore.train \
  "experiment=$EXPERIMENT" \
  "trainer.num_workers=8"

RUN_DIR=$(ls -dt "artifacts/runs/${EXP_NAME}/"*/ 2>/dev/null | head -n1)
if [ -z "$RUN_DIR" ]; then
  echo "error: could not locate run dir after training under artifacts/runs/$EXP_NAME/" >&2
  exit 4
fi
echo "RUN_DIR=$RUN_DIR"

srun --chdir="$ROOT_DIR" "$PYTHON_BIN" -m flatcfm.modelcore.predict \
  "predict.run_dir=$RUN_DIR"

echo "== done EXPERIMENT=$EXPERIMENT =="
