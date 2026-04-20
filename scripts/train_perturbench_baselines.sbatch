#!/usr/bin/env bash
#SBATCH --job-name=pb_baselines
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-8

# train all PerturBench published-best-params baselines for sci-Plex3
# uses the perturbench package (installed in the FlatCFM venv) directly so
# the protocol exactly matches the published Table 2 numbers
#
# scgpt-based variants are skipped because they need pretrained scGPT
# embeddings which require additional setup
#
# usage:
#   sbatch --partition=gpu_rtx6000 scripts/train_perturbench_baselines.sbatch
#
# to throttle concurrency (eg queue caps), override at submit:
#   sbatch --partition=gpu_rtx6000 --array=0-8%4 scripts/train_perturbench_baselines.sbatch

set -euo pipefail

# perturbench writes outputs relative to its own paths config so cd into
# the perturbench repo before launching otherwise logs end up in cwd
PERTURBENCH_DIR="/home/dac227/scratch_pi_sk2433/dac227/perturbench"
FLATCFM_DIR="/nfs/roberts/scratch/pi_sk2433/dac227/FlatCFM"
FLATCFM_VENV="$FLATCFM_DIR/.venv"

cd "$PERTURBENCH_DIR"
echo "PERTURBENCH_DIR=$PERTURBENCH_DIR"

# put TMPDIR on the same filesystem as the checkpoint output otherwise
# lightning's _atomic_save writes to /tmp on the compute node then fails to
# os.rename across devices into /nfs (Invalid cross-device link)
export TMPDIR="$PERTURBENCH_DIR/tmp"
mkdir -p "$TMPDIR"
echo "TMPDIR=$TMPDIR"

# wrapper monkey patches lightning checkpoint loader to use weights_only=False
# required because torch >= 2.6 broke checkpoint reload for the perturbench
# data transform pipeline objects pickled inside the best ckpt
PYTHON_BIN="$FLATCFM_VENV/bin/python"
WRAPPER="$FLATCFM_DIR/scripts/run_perturbench.py"

# perturbench published-best-params experiments (Table 2 reproductions)
EXPERIMENTS=(
  "neurips2025/sciplex3/decoder_best_params_sciplex3"
  "neurips2025/sciplex3/decoder_cov_best_params_sciplex3"
  "neurips2025/sciplex3/linear_best_params_sciplex3"
  "neurips2025/sciplex3/latent_best_params_sciplex3"
  "neurips2025/sciplex3/cpa_best_params_sciplex3"
  "neurips2025/sciplex3/cpa_no_adv_best_params_sciplex3"
  "neurips2025/sciplex3/sams_best_params_sciplex3"
  "neurips2025/sciplex3/sams_modified_best_params_sciplex3"
  "neurips2025/sciplex3/biolord_best_params_sciplex3"
)

# experiment selection precedence:
#   1. EXPERIMENT env var (for ad-hoc experiments not in the EXPERIMENTS array
#      like flatcfm ones under sciplex3/flatcfm_*_sciplex3)
#   2. EXPERIMENTS[$SLURM_ARRAY_TASK_ID] (default behavior for the baseline sweep)
if [ -n "${EXPERIMENT:-}" ]; then
  experiment="$EXPERIMENT"
elif [ "${SLURM_ARRAY_TASK_ID:-}" = "" ]; then
  echo "error: neither EXPERIMENT env var nor SLURM_ARRAY_TASK_ID set" >&2
  exit 1
else
  experiment="${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}"
fi

# optional env vars to override config at submit time:
#   SEED=246       -> seed=246
#   PATIENCE=150   -> callbacks.early_stopping.patience=150
#                     (useful for cpa which has an adversarial val_loss that
#                      plateaus early and triggers premature early stopping)
EXTRA_ARGS=()
if [ -n "${SEED:-}" ]; then
  EXTRA_ARGS+=("seed=$SEED")
fi
if [ -n "${PATIENCE:-}" ]; then
  EXTRA_ARGS+=("callbacks.early_stopping.patience=$PATIENCE")
fi
if [ -n "${DEBIAS:-}" ]; then
  EXTRA_ARGS+=("model.debias=$DEBIAS")
fi

echo "== task $SLURM_ARRAY_TASK_ID =="
echo "experiment=$experiment"
echo "wrapper=$WRAPPER"
echo "extra_args=${EXTRA_ARGS[*]:-<none>}"

# trainer.precision=32 forces full precision because some perturbench models
# (eg cpa) emit float16 sparse predictions in mixed precision and scipy.sparse
# rejects float16 dtype when concatenating anndata objects in test_step
srun --chdir="$PERTURBENCH_DIR" "$PYTHON_BIN" "$WRAPPER" \
  "experiment=$experiment" \
  "train=true" \
  "trainer.precision=32" \
  "${EXTRA_ARGS[@]}"

echo "== task $SLURM_ARRAY_TASK_ID done =="
