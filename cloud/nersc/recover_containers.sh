#!/usr/bin/env bash
# Run on an allocated CPU/GPU node. Restart only this project's API, then test it.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/cloud/nersc/container_env.sh"
if [[ "$(hostname -s)" == login* || -z "${SLURM_JOB_ID:-}" ]]; then
  echo 'Enter an allocated compute node with srun before running this recovery.' >&2
  exit 1
fi
PYTHON="$ROOT/.envs/eval/bin/python"
[[ -x "$PYTHON" ]] || { echo 'Evaluator environment missing; run bootstrap_scaffolds.py --name eval first.' >&2; exit 1; }
command -v podman-hpc >/dev/null
command -v slirp4netns >/dev/null
"$PYTHON" "$ROOT/scripts/stop_container_service.py" --stop
SERVICE_LOG="$ROOT/logs/container-service-${SLURM_JOB_ID}-$(hostname -s)-$(date -u +%Y%m%dT%H%M%SZ).log"
bash "$ROOT/cloud/nersc/container_service.sh" >"$SERVICE_LOG" 2>&1 &
SERVICE_PID=$!
echo "Project service PID: $SERVICE_PID"
echo "Service log: $SERVICE_LOG"
if ! "$PYTHON" - "$SERVICE_PID" "$DOCKER_HOST" <<'PY'
import os,socket,sys,time
pid=int(sys.argv[1]);endpoint=sys.argv[2].removeprefix('unix://')
deadline=time.monotonic()+30
while time.monotonic()<deadline:
    try:os.kill(pid,0)
    except ProcessLookupError:raise SystemExit('API service exited before becoming ready; inspect its log.')
    try:
        with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as client:
            client.settimeout(1);client.connect(endpoint)
            client.sendall(b'GET /_ping HTTP/1.0\r\nHost: localhost\r\n\r\n')
            response=client.recv(1024)
            if b'200' in response.split(b'\r\n',1)[0]:break
    except OSError:pass
    time.sleep(.2)
else:raise SystemExit('API service was not ready within 30 seconds.')
PY
then
  tail -n 80 "$SERVICE_LOG"
  exit 1
fi
if ! "$PYTHON" "$ROOT/scripts/check_container_api.py"; then
  echo "Container check failed. Service log: $SERVICE_LOG" >&2
  tail -n 80 "$SERVICE_LOG"
  exit 1
fi
echo 'Container API check passed. Source cloud/nersc/container_env.sh in the shell used for benchmark commands.'
