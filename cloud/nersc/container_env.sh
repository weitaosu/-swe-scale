#!/usr/bin/env bash
# Source on the CPU evaluation node; no system configuration is changed.
export STUDY_ROOT="${STUDY_ROOT:-${SCRATCH:?SCRATCH must be set}/swe-study-a100}"
source "$STUDY_ROOT/cloud/nersc/env.sh"
export SWE_STUDY_CONTAINER_ENGINE=podman-hpc
export PODMANHPC_RUN_ROOT="/tmp/${UID}_swe_study_podman"
export PODMANHPC_GRAPH_ROOT="$PODMANHPC_RUN_ROOT/graph"
export PODMANHPC_CONFIG_HOME="$STUDY_ROOT/.cache/podman/config"
export PODMANHPC_ADDITIONAL_STORES=""
export SQUASH_DIR="$STUDY_ROOT/.cache/podman/squash"
export DOCKER_HOST="unix://$PODMANHPC_RUN_ROOT/docker.sock"
export SWE_STUDY_CONTAINER_STORAGE_MANIFEST="manifests/container_storage/$(hostname -s).json"
export PATH="$STUDY_ROOT/cloud/nersc/podman-bin:$PATH"
mkdir -p "$PODMANHPC_GRAPH_ROOT" "$PODMANHPC_CONFIG_HOME" "$SQUASH_DIR"
mkdir -p "$PODMANHPC_RUN_ROOT"
chmod 700 "$PODMANHPC_RUN_ROOT"
