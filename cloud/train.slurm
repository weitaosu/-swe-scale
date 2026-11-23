#!/usr/bin/env bash
#SBATCH --job-name=swe-qlora
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
# Edit GPU count, partition/account and memory to match your cluster.
# Submit FROM the extracted project root. Do setup/prepare first in an allocation.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
# One Slurm task launches all per-GPU torchrun workers; do not use one task per GPU.
bash cloud/run.sh train --gpus "${GPUS:-1}" "$@"
