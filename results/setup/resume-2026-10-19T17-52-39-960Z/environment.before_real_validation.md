# Environment findings

Updated 2026-10-19. This report distinguishes user-provided Ubuntu terminal results from agent-side Windows probes. The Windows access failures remain historical/current session evidence; they do not describe the working Ubuntu Docker server.

## Default environment for future sessions

User preference: use the existing Ubuntu 24.04.2 WSL2 environment through the standard exec_command runner for future project execution. Persistent workspace instructions are in [AGENTS.md](../AGENTS.md).

Use /mnt/c/Users/weita/Desktop/SE in AI/proj as the working directory. Bootstrap with /usr/bin/python3 (verified 3.12.3), then use .envs/train/bin/python once the project Linux environment exists. Select separate project Linux interpreters for the older agent and official evaluator. The interactive shell's Python 3.11.6 is a separate interpreter choice.

~~~text
wsl.exe --cd "/mnt/c/Users/weita/Desktop/SE in AI/proj" --exec env PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 scripts/study.py preflight
~~~

The standard runner works even though Node REPL child-process attempts and some native Windows helper commands fail. Use the working WSL route before treating a tool-specific failure as an environment blocker. Keep all artifacts and logs under the shared project folder.

## Agent-verified Ubuntu route

The standard command runner now enters Ubuntu successfully. The earlier E_ACCESSDENIED results came from the Node REPL child-process route and do not block the working runner. The agent verified the shared project path, /usr/bin/python3 3.12.3, PyTorch 2.6.0+cu124 with a CUDA tensor operation, BF16 support, Docker 26.1.3, Hugging Face HTTP 200, and a successful GitHub git ls-remote.

The interactive shell's reported Python 3.11.6 differs from the non-interactive /usr/bin/python3 3.12.3. Use an explicit interpreter for the project Linux environment. Transformers, PEFT, bitsandbytes, and the other training dependencies are absent from this system Python and still need a separate project installation.

[Verified route and command logs](../results/access_check/working-wsl-2026-10-19T17-38-28-975Z/summary.json); [Linux hardware/package preflight](../results/preflight/20261019T173829-35e0d010/hardware.json).

## User-confirmed Ubuntu terminal

| Check | User-provided result |
| --- | --- |
| OS | Ubuntu 24.04.2 LTS, WSL2 kernel 5.15.153.1-microsoft-standard-WSL2, x86_64 |
| Python | python3 reports 3.11.6 |
| GPU visibility | NVIDIA GeForce RTX 4090 visible through nvidia-smi |
| GPU software display | NVIDIA-SMI 535.183.01; Driver Version 610.88; CUDA Version 13.3 as reported by the tool |
| GPU memory at observation | 10,966 MiB used / 24,564 MiB total; subtraction leaves 13,598 MiB (about 13.3 GiB) |
| Listed GPU processes | python3.11, Xwayland, ollama |
| Docker client/server | Both 26.1.3; server details returned successfully in the Ubuntu terminal |
| Docker storage | /var/lib/docker; overlay2; backing filesystem extfs |
| Existing Docker state | 3 containers, 2 running; 3 images; these predate project container validation |
| Docker host resources | 24 CPUs, 46.93 GiB reported memory |
| Docker warnings | Missing CLI-plugin targets and unsupported blkio throttling features were reported |
| Linux PyTorch CUDA | Later agent check passed using /usr/bin/python3 3.12.3 and PyTorch 2.6.0+cu124 |
| Project image/runtime controls | Not yet tested |

Full [user-provided terminal output](../results/access_check/ubuntu-readiness-2026-10-19T17-34-45-861Z/user_terminal_output.txt) and [structured findings](../results/access_check/ubuntu-readiness-2026-10-19T17-34-45-861Z/findings.json) are saved locally. The displayed GPU memory is a single observation and may change. No existing GPU process or Docker container was stopped. The plugin warnings did not prevent docker info from returning server details; the selected image/build/runtime paths still need validation.

Python 3.11.6 meets this project package's declared >=3.11 minimum. The existing validated environment is Windows Python 3.12.6. Linux dependencies and the agent/evaluator's own pinned Python requirements must be checked separately.

## Agent-side Windows checks

| Check | Observed result |
| --- | --- |
| GPU | One NVIDIA GeForce RTX 4090, 25,756,696,576 bytes device memory (about 24 GiB) |
| CUDA via PyTorch | Available; tensor operation, NF4 training, and generation passed |
| Compute / BF16 | Capability 8.9; BF16 supported |
| Python | 3.12.6; .envs/validation reuses system site packages |
| CPU | 24 logical processors |
| RAM | 102,823,464,960 bytes total physical memory (about 95.8 GiB) |
| PyTorch | 2.9.1+cu128, compiled for CUDA 12.8 |
| Training stack | Transformers 5.5.3; PEFT 0.18.1; bitsandbytes 0.49.2; Accelerate 1.13.0 |
| Dataset stack | datasets 3.6.0; PyArrow 23.0.1; Hugging Face Hub 1.12.0 |
| Serving stack | FastAPI 0.136.1; Uvicorn 0.46.0 |
| Git | 2.46.0.windows.1; executable works, remote HTTPS fails |
| nvidia-smi | NVML initialization error; driver version not established |
| Agent-to-WSL access | Enumeration denied with E_ACCESSDENIED; a direct wsl.exe --exec probe also failed with Wsl/Service/CreateInstance/E_ACCESSDENIED |
| Docker | Client 27.2.0 exists; engine pipe absent/unavailable; config read denied |
| Hugging Face | Real model and SFT dataset requests fail with WinError 10013 |
| GitHub checkout | git ls-remote fails connecting to port 443 |

A zero Docker-info exit code was observed despite ServerErrors and an empty ServerVersion. Readiness is therefore based on valid server information, not only the exit code. The nvidia-smi error does not negate successful PyTorch CUDA work; no driver changes were made.

The initial project-volume free-space baseline was 102,881,542,144 bytes. Current free space and conservative volume growth are in manifests/storage_usage.json. This accounting includes concurrent use by other applications and OS paging, so volume growth is not the size of this project. Local file sizes are recorded separately in manifests/artifact_inventory.json.

All project-controlled caches, temp files, environments, model/data targets, adapters, and logs are under this folder. Existing system packages were read through a small project venv; they were not duplicated or modified. Package-manager dependency ranges are bootstrap inputs; manifests/validation_packages.freeze.txt is the exact observed Windows inventory, not a clean Linux lockfile.

The physical host-volume/backing-file mapping for Ubuntu Docker's /var/lib/docker remains unverified. This project has not created images or containers; the user's Docker server already has three containers and three images. The Docker gate requires a baseline before grading/rollout and observes shared storage limits during child commands. These checks are conservative preflight/monitoring safeguards, not filesystem quotas. Keep the first image operation limited to one task and inspect its actual expansion before adding tasks.

The user has confirmed a working interactive Ubuntu 24.04.2 WSL2 installation. An earlier Node REPL subprocess route could not enter that installation; its failed probe is recorded in [the Ubuntu access check](../results/access_check/ubuntu-check-2026-10-19T17-29-13-391Z/direct_wsl_probe.json). The user's later terminal output above confirms GPU visibility and Docker daemon connectivity in Ubuntu. The later standard-runner checks above verified Linux PyTorch CUDA and GitHub/Hugging Face access. Physical Docker storage accounting, the project Linux package stack, and project image/runtime controls remain unverified. The scripts preserve failures and stop at these boundaries; no paid resources or global Docker/WSL changes were made.
