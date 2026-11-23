#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
exec > >(tee -a "logs/model-download-$(date -u +%Y%m%dT%H%M%S)-$$.log") 2>&1
# Initialize before downloads; never import a workstation storage baseline.
python scripts/cloud.py init --storage-accounting project
BOOTSTRAP="$STUDY_ROOT/.envs/bootstrap/bin/python"
if [[ ! -x "$BOOTSTRAP" ]]; then
  conda create -y --prefix "$STUDY_ROOT/.envs/bootstrap" python=3.12 pip
fi
"$BOOTSTRAP" -m pip install --no-cache-dir 'huggingface-hub==1.12.0'
SIZE="${1:-7B}"
case "$SIZE" in
  all) SIZES=(1.5B 3B 7B) ;;
  1.5B|3B|7B) SIZES=("$SIZE") ;;
  *) printf '%s\n' 'Choose 1.5B, 3B, 7B, or all.' >&2; exit 2 ;;
esac
for size in "${SIZES[@]}"; do
  "$BOOTSTRAP" scripts/study.py download --kind model --only "$size"
done
