#!/usr/bin/env bash
#SBATCH --job-name=ode_time
#SBATCH --output=slurm-%x-%j-bs${BATCH_SIZE}.out
#SBATCH --error=slurm-%x-%j-bs${BATCH_SIZE}.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=02:00:00

# usage:
#   BATCH_SIZE=4000 sbatch --export=ALL,BATCH_SIZE --partition=gpu_rtx6000 scripts/time_ode_batch.sbatch
#   BATCH_SIZE=64   sbatch --export=ALL,BATCH_SIZE --partition=gpu_rtx6000 scripts/time_ode_batch.sbatch
#
# times one short ODE training run with native flatcfm pipeline to compare
# batch sizes (keeping steps_per_epoch and epochs fixed for fair comparison)

set -euo pipefail

cd /nfs/roberts/scratch/pi_sk2433/dac227/FlatCFM

BATCH_SIZE="${BATCH_SIZE:?set BATCH_SIZE via --export}"
N_EPOCHS="${N_EPOCHS:-5}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-50}"

echo "== batch size $BATCH_SIZE, $N_EPOCHS epochs, $STEPS_PER_EPOCH steps/epoch =="
echo "start: $(date '+%F %T')"
t0=$SECONDS

.venv/bin/python -m flatcfm.modelcore.train \
  experiment=sciplex/ode_pca \
  experiment_name="sciplex_ode_pca_timing_bs${BATCH_SIZE}" \
  task.batch_size="${BATCH_SIZE}" \
  task.epochs="${N_EPOCHS}" \
  task.steps_per_epoch="${STEPS_PER_EPOCH}" \
  trainer.precision=bf16-mixed \
  trainer.num_workers=12

t1=$SECONDS
echo "end: $(date '+%F %T')"
echo "elapsed seconds: $((t1 - t0))"
echo "seconds per step: $(python3 -c "print(($t1 - $t0) / ($N_EPOCHS * $STEPS_PER_EPOCH))")"
