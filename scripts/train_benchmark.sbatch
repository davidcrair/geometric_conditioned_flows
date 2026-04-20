#!/usr/bin/env bash
#SBATCH --job-name=benchmark
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=12:00:00

# train + predict any sciplex experiment against any splitter for the
# view_benchmark_results thesis comparison (k562 a549 mcf7 splits)
#
# usage:
#   sbatch --partition=scavenge_gpu \
#     --export=ALL,EXPERIMENT=sciplex/fm_pca_deg,SPLITTER=strict_a549,DIM=16 \
#     scripts/train_benchmark.sbatch
#
#   sbatch --partition=scavenge_gpu \
#     --export=ALL,EXPERIMENT=sciplex/fm_ae_latent_phate,SPLITTER=strict_mcf7 \
#     scripts/train_benchmark.sbatch
#
# env vars:
#   EXPERIMENT (required) hydra experiment path e.g. sciplex/fm_pca_deg
#   SPLITTER   (required) splitter name e.g. strict_k562 strict_a549 strict_mcf7
#   DIM        (optional) PCA n_components override for pca-dim sweeps
#   EXP_NAME   (optional) override experiment_name otherwise derived as
#              <base>_<short_splitter>[_d<DIM>]
#   SKIP_PREDICT (optional) set to 1 to skip the predict stage (for AE pretrain)

set -euo pipefail

ROOT_DIR="${FLATCFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$ROOT_DIR"
echo "ROOT_DIR=$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

: "${EXPERIMENT:?error: EXPERIMENT env var not set}"
: "${SPLITTER:?error: SPLITTER env var not set}"

# derive a short splitter suffix for experiment_name
# strict_k562 -> k562, strict_a549 -> a549, strict_mcf7 -> mcf7
case "$SPLITTER" in
  strict_k562) SUFFIX="k562" ;;
  strict_a549) SUFFIX="a549" ;;
  strict_mcf7) SUFFIX="mcf7" ;;
  *)           SUFFIX="$SPLITTER" ;;
esac

# base experiment_name from the yaml
EXP_YAML="$ROOT_DIR/src/flatcfm/configs/experiment/${EXPERIMENT}.yaml"
if [ ! -f "$EXP_YAML" ]; then
  echo "error: experiment config not found: $EXP_YAML" >&2
  exit 2
fi
BASE_NAME=$(awk -F': ' '/^experiment_name:/ {print $2; exit}' "$EXP_YAML")
if [ -z "$BASE_NAME" ]; then
  echo "error: could not parse experiment_name from $EXP_YAML" >&2
  exit 3
fi

# default name pattern: <base>_<suffix>[_d<DIM>]
if [ -z "${EXP_NAME:-}" ]; then
  EXP_NAME="${BASE_NAME}_${SUFFIX}"
  if [ -n "${DIM:-}" ]; then
    EXP_NAME="${EXP_NAME}_d${DIM}"
  fi
fi

EXTRA_ARGS=()
if [ -n "${DIM:-}" ]; then
  EXTRA_ARGS+=("space.projections.0.n_components=$DIM")
fi
if [ -n "${TASK_EPOCHS:-}" ]; then
  EXTRA_ARGS+=("task.epochs=$TASK_EPOCHS")
fi

echo "EXPERIMENT=$EXPERIMENT"
echo "SPLITTER=$SPLITTER"
echo "EXP_NAME=$EXP_NAME"
echo "extra=${EXTRA_ARGS[*]:-<none>}"

srun --chdir="$ROOT_DIR" "$PYTHON_BIN" -m flatcfm.modelcore.train \
  "experiment=$EXPERIMENT" \
  "splitter=$SPLITTER" \
  "experiment_name=$EXP_NAME" \
  "trainer.num_workers=8" \
  "${EXTRA_ARGS[@]}"

if [ "${SKIP_PREDICT:-0}" = "1" ]; then
  echo "SKIP_PREDICT=1 skipping predict stage"
  echo "== done $EXP_NAME (train only) =="
  exit 0
fi

RUN_DIR=$(ls -dt "artifacts/runs/${EXP_NAME}/"*/ 2>/dev/null | head -n1)
if [ -z "$RUN_DIR" ]; then
  echo "error: could not locate run dir after training under artifacts/runs/$EXP_NAME/" >&2
  exit 4
fi
echo "RUN_DIR=$RUN_DIR"

srun --chdir="$ROOT_DIR" "$PYTHON_BIN" -m flatcfm.modelcore.predict \
  "predict.run_dir=$RUN_DIR"

echo "== done $EXP_NAME =="
