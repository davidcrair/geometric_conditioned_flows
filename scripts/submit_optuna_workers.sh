#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ "$#" -lt 5 ]; then
  echo "usage: $0 <experiment> <hpo_config> <worker_count> <trials_per_worker> <optuna_storage> [study_name] [extra sbatch args]"
  exit 1
fi

EXPERIMENT="$1"
HPO_CONFIG="$2"
WORKER_COUNT="$3"
TRIALS_PER_WORKER="$4"
OPTUNA_STORAGE="$5"
if [ "$#" -ge 6 ]; then
  STUDY_NAME="$6"
  shift 6
else
  STUDY_NAME="$(printf '%s' "$EXPERIMENT" | tr '/' '_')"
  shift 5
fi

if [ "$WORKER_COUNT" -lt 1 ]; then
  echo "worker_count must be at least 1"
  exit 1
fi

ARRAY_SPEC="0-$((WORKER_COUNT - 1))"

sbatch \
  --array="$ARRAY_SPEC" \
  --export=ALL,FLATCFM_ROOT="$ROOT_DIR",EXPERIMENT="$EXPERIMENT",HPO_CONFIG="$HPO_CONFIG",OPTUNA_STORAGE="$OPTUNA_STORAGE",OPTUNA_STUDY_NAME="$STUDY_NAME",OPTUNA_TRIALS_PER_WORKER="$TRIALS_PER_WORKER" \
  "$@" \
  scripts/optuna_worker.sbatch
