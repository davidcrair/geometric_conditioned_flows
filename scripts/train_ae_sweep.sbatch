#!/usr/bin/env bash
#SBATCH --job-name=ae_sweep
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --array=0-28       # 29 entries: 15 original + 7 mse_wide + 7 linear

# train 15 missing AE dim sweep models in parallel as a SLURM job array
# usage:
#   mkdir -p logs
#   sbatch scripts/train_ae_sweep.sbatch
#
# to throttle concurrent tasks (eg queue caps), override at submit time:
#   sbatch --array=0-14%4 scripts/train_ae_sweep.sbatch
#
# add partition / account here or pass with --partition= / --account= on sbatch
# #SBATCH --partition=gpu
# #SBATCH --account=pi_sk2433

set -euo pipefail

# on the compute node BASH_SOURCE points to /var/spool/slurmd/... not the repo
# so use SLURM_SUBMIT_DIR which slurm sets to the directory sbatch was run from
# (expected to be the repo root)
ROOT_DIR="${FLATCFM_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$ROOT_DIR"
echo "ROOT_DIR=$ROOT_DIR"

# absolute path required: srun launches on the compute node with its own CWD
# so a relative .venv/bin/python would resolve against /var/spool/slurmd/ and fail
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

# each entry: "experiment latent_dim tag [extra_hydra_overrides...]"
# keep in sync with scripts/check_dim_sweep.py --generate
CONFIGS=(
  "sciplex/ae_log1p_nb_pullback   32   sciplex_ae_deg_nb_d32     loss.weights.pullback=0.0"
  "sciplex/ae_log1p_nb_pullback   64   sciplex_ae_deg_nb_d64     loss.weights.pullback=0.0"
  "sciplex/ae_log1p_nb_pullback   128  sciplex_ae_deg_nb_d128    loss.weights.pullback=0.0"
  "sciplex/ae_log1p_nb_pullback   256  sciplex_ae_deg_nb_d256    loss.weights.pullback=0.0"
  "sciplex/ae_log1p_nb_pullback   512  sciplex_ae_deg_nb_d512    loss.weights.pullback=0.0"
  "sciplex/ae_log1p_nb_pullback   1024 sciplex_ae_deg_nb_d1024   loss.weights.pullback=0.0"
  "sciplex/ae_log1p_nb_pullback   16   sciplex_ae_deg_flatvi_d16"
  "sciplex/ae_log1p_nb_pullback   32   sciplex_ae_deg_flatvi_d32"
  "sciplex/ae_log1p_nb_pullback   64   sciplex_ae_deg_flatvi_d64"
  "sciplex/ae_log1p_nb_pullback   128  sciplex_ae_deg_flatvi_d128"
  "sciplex/ae_log1p_nb_pullback   256  sciplex_ae_deg_flatvi_d256"
  "sciplex/ae_log1p_nb_pullback   512  sciplex_ae_deg_flatvi_d512"
  "sciplex/ae_log1p_nb_pullback   1024 sciplex_ae_deg_flatvi_d1024"
  "sciplex/ae_log1p_hybrid        512  sciplex_ae_deg_phate_d512"
  "sciplex/ae_log1p_hybrid        1024 sciplex_ae_deg_phate_d1024"
  # wide MSE AE with hidden_dim scaled to max(512, 2 * latent_dim)
  # so the architectural bottleneck is the latent dim not the hidden dim
  "sciplex/ae_log1p_mse_wide      16   sciplex_ae_deg_mse_wide_d16    model.hidden_dim=512"
  "sciplex/ae_log1p_mse_wide      32   sciplex_ae_deg_mse_wide_d32    model.hidden_dim=512"
  "sciplex/ae_log1p_mse_wide      64   sciplex_ae_deg_mse_wide_d64    model.hidden_dim=512"
  "sciplex/ae_log1p_mse_wide      128  sciplex_ae_deg_mse_wide_d128   model.hidden_dim=512"
  "sciplex/ae_log1p_mse_wide      256  sciplex_ae_deg_mse_wide_d256   model.hidden_dim=512"
  "sciplex/ae_log1p_mse_wide      512  sciplex_ae_deg_mse_wide_d512   model.hidden_dim=1024"
  "sciplex/ae_log1p_mse_wide      1024 sciplex_ae_deg_mse_wide_d1024  model.hidden_dim=2048"
  # linear AE (2 nn.Linear layers, no hidden stack) matches PCA theoretical
  # limit and provides a clean baseline for the nonlinear AE plateau claim
  "sciplex/ae_log1p_mse_wide      16   sciplex_ae_deg_linear_d16      model.family=linear"
  "sciplex/ae_log1p_mse_wide      32   sciplex_ae_deg_linear_d32      model.family=linear"
  "sciplex/ae_log1p_mse_wide      64   sciplex_ae_deg_linear_d64      model.family=linear"
  "sciplex/ae_log1p_mse_wide      128  sciplex_ae_deg_linear_d128     model.family=linear"
  "sciplex/ae_log1p_mse_wide      256  sciplex_ae_deg_linear_d256     model.family=linear"
  "sciplex/ae_log1p_mse_wide      512  sciplex_ae_deg_linear_d512     model.family=linear"
  "sciplex/ae_log1p_mse_wide      1024 sciplex_ae_deg_linear_d1024    model.family=linear"
)

if [ "${SLURM_ARRAY_TASK_ID:-}" = "" ]; then
  echo "error: SLURM_ARRAY_TASK_ID not set -- submit with sbatch" >&2
  exit 1
fi

entry="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
read -r -a fields <<< "$entry"
experiment="${fields[0]}"
latent_dim="${fields[1]}"
tag="${fields[2]}"
extra_overrides=("${fields[@]:3}")

echo "== task $SLURM_ARRAY_TASK_ID =="
echo "experiment=$experiment"
echo "latent_dim=$latent_dim"
echo "tag=$tag"
echo "extra_overrides=${extra_overrides[*]:-<none>}"

srun --chdir="$ROOT_DIR" "$PYTHON_BIN" -m flatcfm.modelcore.train \
  "experiment=$experiment" \
  "model.latent_dim=$latent_dim" \
  "experiment_name=$tag" \
  "space.ae_export_artifact_tag=$tag" \
  "trainer.num_workers=12" \
  "${extra_overrides[@]}"
