#!/usr/bin/env bash
# Explicit end-to-end run; downloads and trains only when invoked by the user.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
SIZE="${SIZE:-7B}"
CONTEXT="${CONTEXT:-8192}"
bash cloud/run.sh setup
bash cloud/run.sh prepare --size "$SIZE" --max-length "$CONTEXT"
bash cloud/run.sh train --size "$SIZE" --max-length "$CONTEXT" "$@"
