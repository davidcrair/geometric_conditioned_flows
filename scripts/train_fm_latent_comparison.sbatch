#!/usr/bin/env bash
#SBATCH --job-name=fm_latent_cmp
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --array=0-3

# train flow matching head-to-head in four matched-dim latent spaces
# (linear AE, narrow MSE AE, wide MSE AE, PHATE AE) with identical FM
# hyperparameters then run prediction on the held-out perturbations so
# you can compare downstream perturbation prediction metrics across
# latent space architectures
#
# d=128 chosen as a middle-of-the-road dim with fast training and
# enough capacity to be informative all four AE projections at d=128
# must already exist in artifacts/spaces/ before submitting
#
# usage:
#   sbatch --partition=gpu_rtx6000 scripts/train_fm_latent_comparison.sbatch

set -euo pipefail

ROOT_DIR="${FLATCFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$ROOT_DIR"
echo "ROOT_DIR=$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

FM_DIM=128
FM_BASE_EXPERIMENT="sciplex/fm_ae_latent_mse_d512"

# each entry: "ae_variant_tag  fm_experiment_name"
# ae_variant_tag must match the existing AE projection in artifacts/spaces
# fm_experiment_name controls the output run directory
CONFIGS=(
  "sciplex_ae_deg_linear_d128     sciplex_fm_deg_latent_linear_d128"
  "sciplex_ae_deg_mse_d128        sciplex_fm_deg_latent_mse_narrow_d128"
  "sciplex_ae_deg_mse_wide_d128   sciplex_fm_deg_latent_mse_wide_d128"
  "sciplex_ae_deg_phate_d128      sciplex_fm_deg_latent_phate_d128"
)

if [ "${SLURM_ARRAY_TASK_ID:-}" = "" ]; then
  echo "error: SLURM_ARRAY_TASK_ID not set -- submit with sbatch" >&2
  exit 1
fi

entry="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
read -r ae_tag fm_exp_name <<< "$entry"

echo "== task $SLURM_ARRAY_TASK_ID =="
echo "ae_tag=$ae_tag"
echo "fm_exp_name=$fm_exp_name"
echo "fm_dim=$FM_DIM"

# verify the AE projection exists before training
if ! ls "artifacts/spaces/sciplex_ae_projection_"*"_${ae_tag}.pkl" 1>/dev/null 2>&1; then
  echo "error: no projection pickle found for ae_tag=$ae_tag" >&2
  exit 2
fi

# train FM in this specific AE latent
srun --chdir="$ROOT_DIR" "$PYTHON_BIN" -m flatcfm.modelcore.train \
  "experiment=$FM_BASE_EXPERIMENT" \
  "experiment_name=$fm_exp_name" \
  "space.projections.0.artifact_tag=$ae_tag" \
  "task.epochs=500" \
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
