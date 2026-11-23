#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
export PYTHON="$STUDY_ROOT/.envs/bootstrap/bin/python"
[[ -x "$PYTHON" ]] || { printf '%s\n' 'Run download_models.sh first.' >&2; exit 1; }
bash cloud/run.sh setup --skip-gpu-check --storage-accounting project
.envs/train/bin/python scripts/study.py download --kind dataset
