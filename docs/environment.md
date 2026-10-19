# Environment record

Updated 2026-10-19. Use the existing Ubuntu 24.04.2 WSL2 installation for all future execution. Persistent instructions are in [AGENTS.md](../AGENTS.md).

## Default execution

Windows project: C:/Users/weita/Desktop/SE in AI/proj. Ubuntu project: /mnt/c/Users/weita/Desktop/SE in AI/proj. Launch wsl.exe through the standard command runner, with an explicit working directory; login=false also avoids some intermittent runner-helper initialization errors. Node is used for file I/O, rather than launching WSL subprocesses.

~~~text
wsl.exe --cd "/mnt/c/Users/weita/Desktop/SE in AI/proj" --exec env PYTHONDONTWRITEBYTECODE=1 .envs/train/bin/python scripts/study.py preflight
~~~

Bootstrap interpreter: /usr/bin/python3, Python 3.12.3. The user's interactive shell separately reports Python 3.11.6. Select interpreters explicitly.

| Environment | Purpose | State / exact packages |
| --- | --- | --- |
| .envs/train/bin/python | Data, NF4 QLoRA, serving | Installed and GPU-validated; [freeze](../manifests/train_packages.freeze.txt) |
| .envs/agent/bin/python | Pinned SWE-Gym OpenHands fork | Installed; imports and all three released tool types replayed; [freeze](../manifests/agent_packages.freeze.txt) |
| .envs/eval/bin/python | Pinned official SWE-bench grader | Installed; gold/blank controls passed; [freeze](../manifests/eval_packages.freeze.txt) |
| .envs/validation/Scripts/python.exe | Historical Windows fixture checks | Obsolete environment removed during cleanup; logs and creation settings archived |

Training stack: PyTorch 2.9.1 with CUDA 12.8, Transformers 5.5.3, PEFT 0.18.1, bitsandbytes 0.49.2, Accelerate 1.13.0. The training environment is isolated from the older agent and newer official evaluator. [Training bootstrap](../results/bootstrap/20261019T175240-bc869e1e/run.json).

## Hardware and GPU measurements

One RTX 4090, approximately 24 GiB VRAM, capability 8.9, BF16 supported. Ubuntu CUDA tensor checks and real Qwen NF4 training passed. GPU usage changes over time; earlier observations included python3.11, Xwayland, and Ollama. Project cancellation targets only identified project processes.

Real 1.5B results: 4K four-example overfit used 9.34 GiB peak allocated / 11.24 GiB reserved. The optimized 8K 32-example pilot used 6.85 GiB allocated / 16.63 GiB reserved and completed 25 steps. The 3B two-step 8K probe used 8.57 GiB allocated / 13.59 GiB reserved. The 7B two-step 8K probe passed with 14.81 GiB allocated / 18.61 GiB reserved; fresh reloads matched exactly for all three models. Allocated tensor memory, allocator reservations, and nvidia-smi device usage measure different things; record all available observations.

The initial full-logit 8K attempt saturated device memory and slowed sharply, so it was stopped before an optimizer step. The verified target-only-logit implementation skips outputs for masked labels while preserving causal loss and gradients. Inference/reload uses matching nonquantized FP32 layers and BF16 autocast; fresh reload now matches the saved reference exactly. [Pilot evidence](pilot_report.md).

## Local services and Docker

Model API: 127.0.0.1:8615, base URL http://127.0.0.1:8615/v1. Port 8000 was already occupied; the existing service was preserved.

Docker 26.1.3 is reachable through unix:///var/run/docker.sock, with /var/lib/docker on the active Ubuntu VHDX. Its Windows backing file is C:/Users/weita/AppData/Local/Packages/CanonicalGroupLimited.Ubuntu_79rhkp1fndgsc/LocalState/ext4.vhdx, on the project volume. The mapping was matched against the active distribution's Windows registry entry. [Physical storage baseline](../manifests/docker_storage.json).

A checksum-verified Buildx v0.17.1 executable is installed under .tools/docker/cli-plugins, with project-local Docker client configuration. Global Docker configuration was not migrated. [Buildx manifest](../manifests/buildx.json).

Both cached pvlib task images needed dependency-only NumPy 1.26.4 derivatives. Their gold controls pass; blank predictions are handled correctly but skip test execution. The OpenHands runtime requires Chromium during startup even when browsing tools are disabled. Its scripted view/edit/test/patch replay passed with no host mounts and no privileged mode; the loopback-only port replay also passed. All runtime builds now use full command capture and storage monitoring.

## Storage and historical access findings

The approved cap remains 70 GB incremental volume growth and at least 25 GB free. [Live storage state](../manifests/storage_usage.json). Whole-volume changes include other applications and paging; these checks are monitoring safeguards, rather than filesystem quotas. Models/data/environments/checkpoints/logs are under the project. Docker growth is accounted through its VHDX on the same volume.

Current full five-task gates are the approved storage allowance and SQLFluff image access. [Measured Docker constraints and next steps](development_environment_constraints.md). Cached first-task smoke validation remains distinct from full five-task acceptance.

Earlier Windows/Node-route Hugging Face WinError 10013, GitHub HTTPS, WSL E_ACCESSDENIED, and unavailable Windows Docker endpoint findings are historical. The standard WSL runner successfully accessed Hugging Face, GitHub, CUDA, and the Docker daemon. Original [access evidence](../results/access_check/working-wsl-2026-10-19T17-38-28-975Z/summary.json) and [previous environment report](../results/setup/resume-2026-10-19T17-52-39-960Z/environment.before_real_validation.md) are retained. No such old access finding should be reused as a current Ubuntu blocker without rechecking the working route.
