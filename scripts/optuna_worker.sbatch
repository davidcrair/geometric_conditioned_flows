#!/usr/bin/env bash
#SBATCH --job-name=flatcfm_optuna
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail

ROOT_DIR="${FLATCFM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
EXPERIMENT="${EXPERIMENT:?set EXPERIMENT}"
HPO_CONFIG="${HPO_CONFIG:?set HPO_CONFIG}"
OPTUNA_STORAGE="${OPTUNA_STORAGE:?set OPTUNA_STORAGE}"
OPTUNA_STUDY_NAME="${OPTUNA_STUDY_NAME:-}"
OPTUNA_TRIALS_PER_WORKER="${OPTUNA_TRIALS_PER_WORKER:?set OPTUNA_TRIALS_PER_WORKER}"
TRAINER_PRECISION="${TRAINER_PRECISION:-bf16-mixed}"
TRAINER_NUM_WORKERS="${TRAINER_NUM_WORKERS:-4}"
TRAINER_PIN_MEMORY="${TRAINER_PIN_MEMORY:-true}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [ -z "$OPTUNA_STUDY_NAME" ]; then
  OPTUNA_STUDY_NAME="$(printf '%s' "$EXPERIMENT" | tr '/' '_')"
fi

export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  echo "worker ${SLURM_ARRAY_TASK_ID} starting"
fi

echo "experiment=$EXPERIMENT"
echo "hpo=$HPO_CONFIG"
echo "study=$OPTUNA_STUDY_NAME"
echo "storage=$OPTUNA_STORAGE"
echo "trials_per_worker=$OPTUNA_TRIALS_PER_WORKER"

read -r -a EXTRA_ARGV <<< "$EXTRA_ARGS"

"$PYTHON_BIN" -m flatcfm.modelcore.train \
  "experiment=${EXPERIMENT}" \
  "hpo=${HPO_CONFIG}" \
  "hydra.sweeper.n_trials=${OPTUNA_TRIALS_PER_WORKER}" \
  "hydra.sweeper.n_jobs=1" \
  "hydra.sweeper.study_name=${OPTUNA_STUDY_NAME}" \
  "hydra.sweeper.storage=${OPTUNA_STORAGE}" \
  "trainer.accelerator=gpu" \
  "trainer.devices=1" \
  "trainer.precision=${TRAINER_PRECISION}" \
  "trainer.num_workers=${TRAINER_NUM_WORKERS}" \
  "trainer.pin_memory=${TRAINER_PIN_MEMORY}" \
  "${EXTRA_ARGV[@]}"
