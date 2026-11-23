#!/bin/bash -l
#SBATCH --job-name=swe-data
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
source cloud/nersc/env.sh
SIZE="${MODEL_SIZE:-7B}"
CONTEXT="${CONTEXT:-32768}"
srun --ntasks=1 --cpu-bind=none bash cloud/run.sh prepare \
  --size "$SIZE" --max-length "$CONTEXT"
