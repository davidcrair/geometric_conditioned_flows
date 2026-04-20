#!/usr/bin/env bash
#SBATCH --job-name=fm_cond_reg
#SBATCH --output=slurm_logs/slurm-%x-%A_%a.out
#SBATCH --error=slurm_logs/slurm-%x-%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --array=0-2

# conditioning encoder regularization ablation
# three new FM runs in the MSE+dist d=128 AE latent one per dropout combo
# baseline (no regularization) is already trained as
#   sciplex_fm_deg_ae_latent_mse_d128_dist_k562
# directly targets the held out cosine_log_fc collapse on K562 without
# touching the backbone or AE
#
# task 0 sample_covariate_dropout=0.2   (cell_type CFG dropout only)
# task 1 perturbation_dropout=0.2       (pert CFG dropout only)
# task 2 all three dropout knobs at 0.2 (combined regularization)
#
# usage
#   sbatch --partition=gpu_rtx6000 scripts/train_fm_cond_reg_sweep.sbatch

set -euo pipefail

ROOT_DIR="${FLATCFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$ROOT_DIR"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

FM_BASE_EXPERIMENT="sciplex/fm_ae_latent_hybrid_d1e3"
AE_TAG="sciplex_ae_deg_log1p_mse_d128_dist"

# each entry holds "experiment_name_suffix  pert_dropout  sample_cov_dropout  embedding_dropout"
CONFIGS=(
  "cell_dropout0p2      0.0  0.2  0.0"
  "pert_dropout0p2      0.2  0.0  0.0"
  "all_dropout0p2       0.2  0.2  0.2"
)

if [ "${SLURM_ARRAY_TASK_ID:-}" = "" ]; then
  echo "error SLURM_ARRAY_TASK_ID not set submit with sbatch" >&2
  exit 1
fi

entry="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
# shellcheck disable=SC2086
read -r suffix pert_p cell_p emb_p <<< $entry

FM_EXP_NAME="sciplex_fm_deg_ae_latent_mse_d128_dist_${suffix}_k562"

echo "== task $SLURM_ARRAY_TASK_ID =="
echo "FM_EXP_NAME=$FM_EXP_NAME"
echo "pert_dropout=$pert_p  sample_cov_dropout=$cell_p  embedding_dropout=$emb_p"

if ! ls "artifacts/spaces/sciplex_ae_projection_"*"_k562_"*"_${AE_TAG}.pkl" 1>/dev/null 2>&1; then
  echo "error no K562 projection pickle found for ae_tag=$AE_TAG" >&2
  exit 2
fi

srun --chdir="$ROOT_DIR" "$PYTHON_BIN" -m flatcfm.modelcore.train \
  "experiment=$FM_BASE_EXPERIMENT" \
  "experiment_name=$FM_EXP_NAME" \
  "splitter=strict_k562" \
  "space.projections.0.artifact_tag=$AE_TAG" \
  "task.epochs=${TASK_EPOCHS:-500}" \
  "task.batch_size=${BATCH_SIZE:-2048}" \
  "task.use_ot_coupling=true" \
  "model.perturbation_dropout=$pert_p" \
  "model.sample_covariate_dropout=$cell_p" \
  "model.embedding_dropout=$emb_p" \
  "trainer.num_workers=${NUM_WORKERS:-12}"

RUN_DIR=$(ls -dt "artifacts/runs/${FM_EXP_NAME}/"*/ 2>/dev/null | head -n1)
if [ -z "$RUN_DIR" ]; then
  echo "error could not locate run dir after training" >&2
  exit 3
fi
echo "RUN_DIR=$RUN_DIR"

srun --chdir="$ROOT_DIR" "$PYTHON_BIN" -m flatcfm.modelcore.predict \
  "predict.run_dir=$RUN_DIR"

echo "== task $SLURM_ARRAY_TASK_ID done =="
