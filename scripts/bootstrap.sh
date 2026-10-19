#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$project_root"
python3 scripts/bootstrap.py --name train --install
