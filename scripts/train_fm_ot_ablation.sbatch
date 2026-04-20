#!/usr/bin/env bash
#SBATCH --job-name=fm_ot_ablation
#SBATCH --output=slurm_logs/slurm-%x-%A_%a.out
#SBATCH --error=slurm_logs/slurm-%x-%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --array=0-3

# fills in the 4 missing cells of the (method dim OT) grid so each method
# x dim pair has both a with OT and without OT variant trained with
# matched hyperparameters within the pair
#
# already trained
#   PCA d16    no OT   100 ep bs 1024
#   PCA d128   no OT   100 ep bs 1024
#   MSE+dist d128 OT   500 ep bs 2048
#   MSE+dist d16  OT   500 ep bs 2048
#
# this script trains the 4 complements
#   task 0  PCA d16      OT   100 ep bs 1024  (matches no-ot sibling)
#   task 1  PCA d128     OT   100 ep bs 1024
#   task 2  MSE+dist d128 no OT 500 ep bs 2048  (matches ot sibling)
#   task 3  MSE+dist d16  no OT 500 ep bs 2048
#
# usage
#   sbatch --partition=gpu_rtx6000 scripts/train_fm_ot_ablation.sbatch

set -euo pipefail

ROOT_DIR="${FLATCFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$ROOT_DIR"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

# each entry holds
#   base_experiment experiment_name extra_overrides epochs batch_size
CONFIGS=(
  "sciplex/fm_pca_deg                 sciplex_fm_deg_pca_d16_ot                                       space.projections.0.n_components=16  true  100 1024"
  "sciplex/fm_pca_deg                 sciplex_fm_deg_pca_d128_ot                                      space.projections.0.n_components=128 true  100 1024"
  "sciplex/fm_ae_latent_hybrid_d1e3   sciplex_fm_deg_ae_latent_mse_d128_dist_no_ot_k562               space.projections.0.artifact_tag=sciplex_ae_deg_log1p_mse_d128_dist false 500 2048"
  "sciplex/fm_ae_latent_hybrid_d1e3   sciplex_fm_deg_ae_latent_mse_d16_dist_no_ot_k562                space.projections.0.artifact_tag=sciplex_ae_deg_log1p_mse_d16_dist  false 500 2048"
)

if [ "${SLURM_ARRAY_TASK_ID:-}" = "" ]; then
  echo "error SLURM_ARRAY_TASK_ID not set" >&2
  exit 1
fi

entry="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
# shellcheck disable=SC2086
read -r base_exp fm_exp space_override ot epochs bs <<< $entry

echo "== task $SLURM_ARRAY_TASK_ID =="
echo "base=$base_exp name=$fm_exp space_override=$space_override ot=$ot epochs=$epochs bs=$bs"

srun --chdir="$ROOT_DIR" "$PYTHON_BIN" -m flatcfm.modelcore.train \
  "experiment=$base_exp" \
  "experiment_name=$fm_exp" \
  "splitter=strict_k562" \
  "$space_override" \
  "task.epochs=$epochs" \
  "task.batch_size=$bs" \
  "task.use_ot_coupling=$ot" \
  "trainer.num_workers=${NUM_WORKERS:-12}"

RUN_DIR=$(ls -dt "artifacts/runs/${fm_exp}/"*/ 2>/dev/null | head -n1)
if [ -z "$RUN_DIR" ]; then
  echo "error could not locate run dir" >&2
  exit 3
fi
echo "RUN_DIR=$RUN_DIR"

srun --chdir="$ROOT_DIR" "$PYTHON_BIN" -m flatcfm.modelcore.predict \
  "predict.run_dir=$RUN_DIR"

echo "== task $SLURM_ARRAY_TASK_ID done =="
