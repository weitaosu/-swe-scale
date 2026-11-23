#!/usr/bin/env bash
# Keep this foreground process alive on the same CPU node as agent/grader work.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/cloud/nersc/container_env.sh"
if [[ "$(hostname -s)" == login* || -z "${SLURM_JOB_ID:-}" ]]; then
  echo 'Run the evaluation API service inside a Slurm compute allocation (CPU or GPU).' >&2
  exit 1
fi
command -v podman-hpc
command -v slirp4netns >/dev/null || { echo 'slirp4netns is unavailable; preserve the log and stop here.' >&2; exit 1; }
exec 9>"$ROOT/.cache/podman/service.lock"
flock -n 9 || { echo 'An evaluation container service already owns this project store.' >&2; exit 1; }
exec podman-hpc system service --time=0 "$DOCKER_HOST"
