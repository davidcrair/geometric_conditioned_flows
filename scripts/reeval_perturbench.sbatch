#!/usr/bin/env bash
#SBATCH --job-name=pb_reeval
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=4:00:00

# re-run perturbench test/evaluation on a previously trained run without
# retraining reapplies the original hydra overrides loads the best checkpoint
# and lets on_test_end recompute the summary csv with any newly added metrics
#
# usage for a single run
#   sbatch --partition=scavenge_gpu \
#     --export=ALL,RUN_DIR=/home/dac227/scratch_pi_sk2433/dac227/perturbench/logs/train/runs/2026-04-18_11-08-53_8766046 \
#     scripts/reeval_perturbench.sbatch
#
# usage to re-eval many runs matching a glob pattern
#   export ALL_RUN_DIRS=$(ls -d /home/.../logs/train/runs/2026-04-18_11-*)
#   sbatch --array=0-$(( $(echo $ALL_RUN_DIRS | wc -w) - 1 )) --partition=scavenge_gpu \
#     --export=ALL,ALL_RUN_DIRS="$ALL_RUN_DIRS" scripts/reeval_perturbench.sbatch

set -euo pipefail

PERTURBENCH_DIR="/home/dac227/scratch_pi_sk2433/dac227/perturbench"
FLATCFM_DIR="/nfs/roberts/scratch/pi_sk2433/dac227/FlatCFM"
PYTHON_BIN="$FLATCFM_DIR/.venv/bin/python"
WRAPPER="$FLATCFM_DIR/scripts/run_perturbench.py"

if [ -n "${ALL_RUN_DIRS:-}" ]; then
  read -r -a _runs <<< "$ALL_RUN_DIRS"
  idx="${SLURM_ARRAY_TASK_ID:-0}"
  RUN_DIR="${_runs[$idx]}"
fi

: "${RUN_DIR:?error: RUN_DIR env var not set}"
echo "RUN_DIR=$RUN_DIR"

OVERRIDES_FILE="$RUN_DIR/.hydra/overrides.yaml"
if [ ! -f "$OVERRIDES_FILE" ]; then
  echo "error: overrides file not found: $OVERRIDES_FILE" >&2
  exit 2
fi

CKPT=$(ls -1 "$RUN_DIR/checkpoints/"*.ckpt 2>/dev/null | head -n1)
if [ -z "$CKPT" ]; then
  echo "error: no checkpoint found under $RUN_DIR/checkpoints/" >&2
  exit 3
fi
echo "CKPT=$CKPT"

# reapply original hydra overrides verbatim but swap train=true -> train=false
# and inject ckpt_path for the test stage other overrides (experiment seed
# precision) come through unchanged
ORIG_OVERRIDES=()
while IFS= read -r line; do
  line="${line#- }"
  # drop leading hyphen from yaml list entry
  if [[ "$line" == "train="* ]]; then continue; fi
  if [[ "$line" == "test="* ]]; then continue; fi
  if [[ "$line" == "ckpt_path="* ]]; then continue; fi
  ORIG_OVERRIDES+=("$line")
done < <(grep -E '^- ' "$OVERRIDES_FILE")

echo "overrides: ${ORIG_OVERRIDES[*]}"

cd "$PERTURBENCH_DIR"
export TMPDIR="$PERTURBENCH_DIR/tmp"
mkdir -p "$TMPDIR"

srun --chdir="$PERTURBENCH_DIR" "$PYTHON_BIN" "$WRAPPER" \
  "${ORIG_OVERRIDES[@]}" \
  "train=false" \
  "test=true" \
  "ckpt_path='$CKPT'"

echo "== reeval done: $RUN_DIR =="
