#!/usr/bin/env bash
#SBATCH --job-name=fm_molformer_d128_k562
#SBATCH --output=slurm_logs/slurm-%x-%A.out
#SBATCH --error=slurm_logs/slurm-%x-%A.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=6:00:00

# train flow matching in k562 mse d128 dist ae latent with frozen
# MolFormer-XL drug embeddings replacing learned perturbation tokens
# matches fm_ae_latent_mse_d128_dist_k562 except for the drug encoder
#
# requires artifacts/drug_embeddings/molformer_xl_sciplex.pt which comes
# from scripts/build_drug_embeddings.py
#
# usage:
#   sbatch --partition=gpu_b200 scripts/train_fm_molformer_mse_d128_k562.sbatch

set -euo pipefail

ROOT_DIR="${FLATCFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$ROOT_DIR"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

DRUG_EMB="artifacts/drug_embeddings/molformer_xl_sciplex.pt"
AE_TAG="sciplex_ae_deg_log1p_mse_d128_dist"
FM_EXP_NAME="sciplex_fm_deg_ae_latent_mse_d128_dist_molformer_k562"

if [ ! -f "$DRUG_EMB" ]; then
  echo "error: drug embedding artifact missing: $DRUG_EMB" >&2
  echo "run: $PYTHON_BIN scripts/build_drug_embeddings.py" >&2
  exit 2
fi

if ! ls "artifacts/spaces/sciplex_ae_projection_"*"_k562_"*"_${AE_TAG}.pkl" 1>/dev/null 2>&1; then
  echo "error: no K562 projection pickle found for ae_tag=$AE_TAG" >&2
  exit 3
fi

srun --chdir="$ROOT_DIR" "$PYTHON_BIN" -m flatcfm.modelcore.train \
  experiment=sciplex/fm_ae_latent_mse_d128_dist_molformer \
  experiment_name="$FM_EXP_NAME" \
  splitter=strict_k562 \
  task.epochs="${TASK_EPOCHS:-500}" \
  task.batch_size="${BATCH_SIZE:-2048}" \
  task.use_ot_coupling=true \
  trainer.num_workers="${NUM_WORKERS:-12}"

RUN_DIR=$(ls -dt "artifacts/runs/${FM_EXP_NAME}/"*/ 2>/dev/null | head -n1)
if [ -z "$RUN_DIR" ]; then
  echo "error: could not locate run dir after training" >&2
  exit 4
fi
echo "RUN_DIR=$RUN_DIR"

srun --chdir="$ROOT_DIR" "$PYTHON_BIN" -m flatcfm.modelcore.predict \
  "predict.run_dir=$RUN_DIR"

echo "== done =="
