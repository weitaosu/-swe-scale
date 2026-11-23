#!/bin/bash -l
#SBATCH --job-name=swe-gpu
#SBATCH --constraint=gpu&hbm40g
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
source cloud/nersc/env.sh
STUDY_GPUS="${STUDY_GPUS:-1}"
printf 'Node %s GPUs %s\n' "$(hostname)" "$STUDY_GPUS"
# One Slurm task owns all requested GPUs. cloud.py starts torchrun workers.
srun --ntasks=1 --gpus-per-task="$STUDY_GPUS" \
  --cpu-bind=none --gpu-bind=none bash cloud/run.sh "$@"
