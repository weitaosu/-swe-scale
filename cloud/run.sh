#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
if [[ "${1:-}" == setup ]]; then
  exec "${PYTHON:-/usr/bin/python3}" scripts/cloud.py "$@"
fi
if [[ ! -x .envs/train/bin/python ]]; then
  printf '%s\n' 'Run bash cloud/run.sh setup first.' >&2
  exit 1
fi
exec .envs/train/bin/python scripts/cloud.py "$@"
