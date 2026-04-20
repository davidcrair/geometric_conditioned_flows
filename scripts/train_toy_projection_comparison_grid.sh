#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-100}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-50}"
BATCH_SIZE="${BATCH_SIZE:-128}"
AE_BATCH_SIZE="${AE_BATCH_SIZE:-256}"
LATENT_DIM="${LATENT_DIM:-32}"
DISTANCE_WEIGHT="${DISTANCE_WEIGHT:-1.0}"
RFF_FEATURE_SCALE="${RFF_FEATURE_SCALE:-1.0}"
FORCE_RETRAIN="${FORCE_RETRAIN:-0}"
FORCE_PREDICT="${FORCE_PREDICT:-0}"

if [ "$#" -gt 0 ]; then
  DIMS=("$@")
else
  DIMS=(2 8 16 512)
fi

export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

for dim in "${DIMS[@]}"; do
  fm_experiment_name="toy_fm_lifted_d${dim}"
  ensure_train \
    "toy/lifted" \
    "$fm_experiment_name" \
    "space.projections.0.ambient_dim=${dim}" \
    "space.projections.0.seed=${SEED}" \
    "task.epochs=${EPOCHS}" \
    "task.steps_per_epoch=${STEPS_PER_EPOCH}" \
    "task.batch_size=${BATCH_SIZE}"
  fm_run_dir="$(latest_run_dir "$fm_experiment_name")"
  ensure_predict "$fm_experiment_name" "$fm_run_dir"

  ae_experiment_name="toy_ae_lifted_gr_d${dim}"
  ensure_train \
    "toy/ae_lifted_gr" \
    "$ae_experiment_name" \
    "space.projections.0.ambient_dim=${dim}" \
    "space.projections.0.seed=${SEED}" \
    "task.epochs=${EPOCHS}" \
    "task.batch_size=${AE_BATCH_SIZE}" \
    "model.latent_dim=${LATENT_DIM}" \
    "loss.weights.distance=${DISTANCE_WEIGHT}"

  fm_grae_experiment_name="toy_fm_lifted_ae_latent_d${dim}"
  ensure_train \
    "toy/fm_lifted_ae_latent" \
    "$fm_grae_experiment_name" \
    "space.projections.0.ambient_dim=${dim}" \
    "space.projections.0.seed=${SEED}" \
    "task.epochs=${EPOCHS}" \
    "task.steps_per_epoch=${STEPS_PER_EPOCH}" \
    "task.batch_size=${BATCH_SIZE}"
  fm_grae_run_dir="$(latest_run_dir "$fm_grae_experiment_name")"
  ensure_predict "$fm_grae_experiment_name" "$fm_grae_run_dir"

  ae_mse_experiment_name="toy_ae_lifted_mse_d${dim}"
  ensure_train \
    "toy/ae_lifted_mse" \
    "$ae_mse_experiment_name" \
    "space.projections.0.ambient_dim=${dim}" \
    "space.projections.0.seed=${SEED}" \
    "+space.ae_export_artifact_tag=ae_mse_d${dim}" \
    "task.epochs=${EPOCHS}" \
    "task.batch_size=${AE_BATCH_SIZE}" \
    "model.latent_dim=${LATENT_DIM}"

  fm_mse_ae_experiment_name="toy_fm_lifted_ae_latent_mse_d${dim}"
  ensure_train \
    "toy/fm_lifted_ae_latent_mse" \
    "$fm_mse_ae_experiment_name" \
    "space.projections.0.ambient_dim=${dim}" \
    "space.projections.0.seed=${SEED}" \
    "space.projections.1.artifact_tag=ae_mse_d${dim}" \
    "task.epochs=${EPOCHS}" \
    "task.steps_per_epoch=${STEPS_PER_EPOCH}" \
    "task.batch_size=${BATCH_SIZE}"
  fm_mse_ae_run_dir="$(latest_run_dir "$fm_mse_ae_experiment_name")"
  ensure_predict "$fm_mse_ae_experiment_name" "$fm_mse_ae_run_dir"

  fm_ot_experiment_name="toy_fm_lifted_ot_d${dim}"
  ensure_train \
    "toy/fm_lifted_ot" \
    "$fm_ot_experiment_name" \
    "space.projections.0.ambient_dim=${dim}" \
    "space.projections.0.seed=${SEED}" \
    "task.epochs=${EPOCHS}" \
    "task.steps_per_epoch=${STEPS_PER_EPOCH}" \
    "task.batch_size=${BATCH_SIZE}"
  fm_ot_run_dir="$(latest_run_dir "$fm_ot_experiment_name")"
  ensure_predict "$fm_ot_experiment_name" "$fm_ot_run_dir"

  fm_lifted_pca_experiment_name="toy_fm_lifted_pca_d${dim}"
  ensure_train \
    "toy/fm_lifted_pca" \
    "$fm_lifted_pca_experiment_name" \
    "space.projections.0.ambient_dim=${dim}" \
    "space.projections.0.seed=${SEED}" \
    "task.epochs=${EPOCHS}" \
    "task.steps_per_epoch=${STEPS_PER_EPOCH}" \
    "task.batch_size=${BATCH_SIZE}"
  fm_lifted_pca_run_dir="$(latest_run_dir "$fm_lifted_pca_experiment_name")"
  ensure_predict "$fm_lifted_pca_experiment_name" "$fm_lifted_pca_run_dir"

  fm_nonlinear_experiment_name="toy_fm_nonlinear_rff_d${dim}"
  ensure_train \
    "toy/fm_nonlinear_rff" \
    "$fm_nonlinear_experiment_name" \
    "space.projections.0.ambient_dim=${dim}" \
    "space.projections.0.seed=${SEED}" \
    "space.projections.0.feature_scale=${RFF_FEATURE_SCALE}" \
    "task.epochs=${EPOCHS}" \
    "task.steps_per_epoch=${STEPS_PER_EPOCH}" \
    "task.batch_size=${BATCH_SIZE}"
  fm_nonlinear_run_dir="$(latest_run_dir "$fm_nonlinear_experiment_name")"
  ensure_predict "$fm_nonlinear_experiment_name" "$fm_nonlinear_run_dir"

  ae_nonlinear_experiment_name="toy_ae_nonlinear_rff_gr_d${dim}"
  ensure_train \
    "toy/ae_nonlinear_rff_gr" \
    "$ae_nonlinear_experiment_name" \
    "space.projections.0.ambient_dim=${dim}" \
    "space.projections.0.seed=${SEED}" \
    "space.projections.0.feature_scale=${RFF_FEATURE_SCALE}" \
    "task.epochs=${EPOCHS}" \
    "task.batch_size=${AE_BATCH_SIZE}" \
    "model.latent_dim=${LATENT_DIM}" \
    "loss.weights.distance=${DISTANCE_WEIGHT}"

  fm_nonlinear_grae_experiment_name="toy_fm_nonlinear_rff_ae_latent_d${dim}"
  ensure_train \
    "toy/fm_nonlinear_rff_ae_latent" \
    "$fm_nonlinear_grae_experiment_name" \
    "space.projections.0.ambient_dim=${dim}" \
    "space.projections.0.seed=${SEED}" \
    "space.projections.0.feature_scale=${RFF_FEATURE_SCALE}" \
    "task.epochs=${EPOCHS}" \
    "task.steps_per_epoch=${STEPS_PER_EPOCH}" \
    "task.batch_size=${BATCH_SIZE}"
  fm_nonlinear_grae_run_dir="$(latest_run_dir "$fm_nonlinear_grae_experiment_name")"
  ensure_predict "$fm_nonlinear_grae_experiment_name" "$fm_nonlinear_grae_run_dir"

  ae_nonlinear_mse_experiment_name="toy_ae_nonlinear_rff_mse_d${dim}"
  ensure_train \
    "toy/ae_nonlinear_rff_mse" \
    "$ae_nonlinear_mse_experiment_name" \
    "space.projections.0.ambient_dim=${dim}" \
    "space.projections.0.seed=${SEED}" \
    "space.projections.0.feature_scale=${RFF_FEATURE_SCALE}" \
    "+space.ae_export_artifact_tag=ae_rff_mse_d${dim}" \
    "task.epochs=${EPOCHS}" \
    "task.batch_size=${AE_BATCH_SIZE}" \
    "model.latent_dim=${LATENT_DIM}"

  fm_nonlinear_mse_ae_experiment_name="toy_fm_nonlinear_rff_ae_latent_mse_d${dim}"
  ensure_train \
    "toy/fm_nonlinear_rff_ae_latent_mse" \
    "$fm_nonlinear_mse_ae_experiment_name" \
    "space.projections.0.ambient_dim=${dim}" \
    "space.projections.0.seed=${SEED}" \
    "space.projections.0.feature_scale=${RFF_FEATURE_SCALE}" \
    "space.projections.1.artifact_tag=ae_rff_mse_d${dim}" \
    "task.epochs=${EPOCHS}" \
    "task.steps_per_epoch=${STEPS_PER_EPOCH}" \
    "task.batch_size=${BATCH_SIZE}"
  fm_nonlinear_mse_ae_run_dir="$(latest_run_dir "$fm_nonlinear_mse_ae_experiment_name")"
  ensure_predict "$fm_nonlinear_mse_ae_experiment_name" "$fm_nonlinear_mse_ae_run_dir"

  fm_nonlinear_ot_experiment_name="toy_fm_nonlinear_rff_ot_d${dim}"
  ensure_train \
    "toy/fm_nonlinear_rff_ot" \
    "$fm_nonlinear_ot_experiment_name" \
    "space.projections.0.ambient_dim=${dim}" \
    "space.projections.0.seed=${SEED}" \
    "space.projections.0.feature_scale=${RFF_FEATURE_SCALE}" \
    "task.epochs=${EPOCHS}" \
    "task.steps_per_epoch=${STEPS_PER_EPOCH}" \
    "task.batch_size=${BATCH_SIZE}"
  fm_nonlinear_ot_run_dir="$(latest_run_dir "$fm_nonlinear_ot_experiment_name")"
  ensure_predict "$fm_nonlinear_ot_experiment_name" "$fm_nonlinear_ot_run_dir"

  fm_nonlinear_pca_experiment_name="toy_fm_nonlinear_rff_pca_d${dim}"
  ensure_train \
    "toy/fm_nonlinear_rff_pca" \
    "$fm_nonlinear_pca_experiment_name" \
    "space.projections.0.ambient_dim=${dim}" \
    "space.projections.0.seed=${SEED}" \
    "space.projections.0.feature_scale=${RFF_FEATURE_SCALE}" \
    "task.epochs=${EPOCHS}" \
    "task.steps_per_epoch=${STEPS_PER_EPOCH}" \
    "task.batch_size=${BATCH_SIZE}"
  fm_nonlinear_pca_run_dir="$(latest_run_dir "$fm_nonlinear_pca_experiment_name")"
  ensure_predict "$fm_nonlinear_pca_experiment_name" "$fm_nonlinear_pca_run_dir"
done
