#!/usr/bin/env bash
#SBATCH --job-name=fm_budget
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --array=0-3

# training-budget FM sweep in narrow MSE AE d=128 latent
# trains FM at epochs in {50, 100, 200, 500} with all other hyperparameters
# identical to the standard fm_ae_latent_mse_d512 config
#
# purpose: diagnose FM overfitting — if held-out cos_lfc peaks before 500
# epochs then the FM model is overfitting to the training perturbations
# (the AE already achieves cos_lfc~0.34 but FM-in-latent gives ~0.14)
#
# prerequisites:
#   artifacts/spaces/sciplex_ae_projection_*_sciplex_ae_deg_mse_d128.pkl
#   must exist before submitting
#
# usage:
#   sbatch --partition=gpu_rtx6000 scripts/train_fm_budget_sweep.sbatch

set -euo pipefail

ROOT_DIR="${FLATCFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$ROOT_DIR"
echo "ROOT_DIR=$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

FM_BASE_EXPERIMENT="sciplex/fm_ae_latent_mse_d512"
AE_TAG="sciplex_ae_deg_mse_d128"

# each entry: "epochs  experiment_name"
CONFIGS=(
  "50   sciplex_fm_deg_budget_e50"
  "100  sciplex_fm_deg_budget_e100"
  "200  sciplex_fm_deg_budget_e200"
  "500  sciplex_fm_deg_budget_e500"
)

if [ "${SLURM_ARRAY_TASK_ID:-}" = "" ]; then
  echo "error: SLURM_ARRAY_TASK_ID not set -- submit with sbatch" >&2
  exit 1
fi

entry="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
read -r epochs fm_exp_name <<< "$entry"

echo "== task $SLURM_ARRAY_TASK_ID =="
echo "ae_tag=$AE_TAG"
echo "epochs=$epochs"
echo "fm_exp_name=$fm_exp_name"

# verify the AE projection exists before training
if ! ls "artifacts/spaces/sciplex_ae_projection_"*"_${AE_TAG}.pkl" 1>/dev/null 2>&1; then
  echo "error: no projection pickle found for ae_tag=$AE_TAG" >&2
  exit 2
fi

# train FM in narrow MSE d=128 latent at this epoch budget
srun --chdir="$ROOT_DIR" "$PYTHON_BIN" -m flatcfm.modelcore.train \
  "experiment=$FM_BASE_EXPERIMENT" \
  "experiment_name=$fm_exp_name" \
  "space.projections.0.artifact_tag=$AE_TAG" \
  "task.epochs=$epochs" \
  "trainer.num_workers=12"

# find the just-created run directory (latest subdirectory under the experiment)
RUN_DIR=$(ls -dt "artifacts/runs/${fm_exp_name}/"*/ 2>/dev/null | head -n1)
if [ -z "$RUN_DIR" ]; then
  echo "error: could not locate run dir after training" >&2
  exit 3
fi
echo "RUN_DIR=$RUN_DIR"

# generate held-out predictions in the same job
srun --chdir="$ROOT_DIR" "$PYTHON_BIN" -m flatcfm.modelcore.predict \
  "predict.run_dir=$RUN_DIR"

echo "== task $SLURM_ARRAY_TASK_ID done =="
