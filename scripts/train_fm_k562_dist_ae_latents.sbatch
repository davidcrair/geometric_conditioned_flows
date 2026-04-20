#!/usr/bin/env bash
#SBATCH --job-name=fm_k562_dist_ae
#SBATCH --output=slurm_logs/slurm-%x-%A_%a.out
#SBATCH --error=slurm_logs/slurm-%x-%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --array=0-1

# train flow matching in two K562 latent spaces that were fit with the
# phate diffusion potential distance loss
#   task 0: hybrid_d1e3 (NB recon + dist)
#   task 1: mse_d128_dist (MSE recon + dist)
#
# follows the same pattern as train_fm_latent_comparison.sbatch and
# runs predict in the same job so you get held-out predictions
# the AE projection pkls for both tags must already exist in
# artifacts/spaces/ for the K562 strict splitter
#
# usage:
#   sbatch --partition=scavenge_gpu scripts/train_fm_k562_dist_ae_latents.sbatch
#   # or on a non preemptible partition
#   sbatch --partition=gpu_h200 scripts/train_fm_k562_dist_ae_latents.sbatch

set -euo pipefail

ROOT_DIR="${FLATCFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$ROOT_DIR"
echo "ROOT_DIR=$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

FM_BASE_EXPERIMENT="sciplex/fm_ae_latent_hybrid_d1e3"

# each entry: "ae_variant_tag  fm_experiment_name"
# ae_variant_tag must match the pkl under artifacts/spaces/
# fm_experiment_name controls the output run directory
CONFIGS=(
  "sciplex_ae_deg_log1p_hybrid_d1e3     sciplex_fm_deg_ae_latent_hybrid_d1e3_diffpot_k562"
  "sciplex_ae_deg_log1p_mse_d128_dist   sciplex_fm_deg_ae_latent_mse_d128_dist_k562"
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

# verify the K562 projection pkl exists before training
if ! ls "artifacts/spaces/sciplex_ae_projection_"*"_k562_"*"_${ae_tag}.pkl" 1>/dev/null 2>&1; then
  echo "error: no K562 projection pickle found for ae_tag=$ae_tag" >&2
  exit 2
fi

# train FM in the AE latent
srun --chdir="$ROOT_DIR" "$PYTHON_BIN" -m flatcfm.modelcore.train \
  "experiment=$FM_BASE_EXPERIMENT" \
  "experiment_name=$fm_exp_name" \
  "splitter=strict_k562" \
  "space.projections.0.artifact_tag=$ae_tag" \
  "task.epochs=${TASK_EPOCHS:-500}" \
  "task.batch_size=${BATCH_SIZE:-2048}" \
  "task.use_ot_coupling=true" \
  "trainer.num_workers=${NUM_WORKERS:-12}"

# find the latest run dir
RUN_DIR=$(ls -dt "artifacts/runs/${fm_exp_name}/"*/ 2>/dev/null | head -n1)
if [ -z "$RUN_DIR" ]; then
  echo "error: could not locate run dir after training" >&2
  exit 3
fi
echo "RUN_DIR=$RUN_DIR"

# held-out predictions in the same job so evaluate can pick them up later
srun --chdir="$ROOT_DIR" "$PYTHON_BIN" -m flatcfm.modelcore.predict \
  "predict.run_dir=$RUN_DIR"

echo "== task $SLURM_ARRAY_TASK_ID done =="
