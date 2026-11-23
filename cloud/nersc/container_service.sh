#!/usr/bin/env bash
# Keep this foreground process alive on the same CPU node as agent/grader work.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/cloud/nersc/container_env.sh"
command -v podman-hpc
exec 9>"$ROOT/.cache/podman/service.lock"
flock -n 9 || { echo 'An evaluation container service already owns this project store.' >&2; exit 1; }
exec podman-hpc system service --time=0 "$DOCKER_HOST"
