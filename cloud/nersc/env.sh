#!/usr/bin/env bash
# Source this in a NERSC login shell or one-node Slurm job.
export STUDY_ROOT="${STUDY_ROOT:-${SCRATCH:?SCRATCH must be set}/swe-study-a100}"
cd "$STUDY_ROOT"
module load python
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export CONDA_PKGS_DIRS="$STUDY_ROOT/.cache/conda"
export PIP_CACHE_DIR="$STUDY_ROOT/.cache/pip"
export HF_HOME="$STUDY_ROOT/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_XET_CACHE="$HF_HOME/xet"
export TMPDIR="$STUDY_ROOT/.cache/tmp"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
mkdir -p "$TMPDIR" logs
