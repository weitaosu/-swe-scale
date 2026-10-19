# CS6158: local SWE agent and QLoRA setup

Implementation approved on 2026-10-19 for one RTX 4090. Scope is the first two proposal milestones, with a 70 GB incremental storage cap and a 25 GB free-space floor. [Current progress](docs/implementation_status.md) records completed checks and remaining work.

**Default environment:** use the existing Ubuntu WSL2 installation through the standard command runner. Follow [workspace instructions](AGENTS.md) and the [environment record](docs/environment.md).

**Current result:** the pipeline code is implemented and tested locally, and real CUDA/NF4 adapter training, checkpoint resume/reload, and the HTTP response interface passed using a random tiny Qwen2 fixture. Real Qwen2.5 snapshots, SWE-Gym payloads, source checkouts, Docker grading, and agent rollouts remain pending. Direct Ubuntu access now works through the standard command runner. Agent checks confirmed the shared project folder, a Linux PyTorch CUDA tensor operation, Docker 26.1.3 connectivity, Hugging Face HTTP 200, and GitHub remote access. The Linux training dependencies and project container integration still need setup. Both milestones are incomplete.

## Models and download links

Use the original Qwen2.5-Coder-Instruct Transformers/Safetensors repositories, including model weights, tokenizer files, and configuration files. Start with 1.5B; download 3B and 7B after the real 1.5B pilot succeeds. NF4 quantization is applied when loading the original snapshots.

| Order | Official model repository | Local destination | Planned snapshot allowance |
| --- | --- | --- | ---: |
| 1 — initial setup and LoRA pilot | [Qwen/Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct) | models/Qwen2.5-Coder-1.5B-Instruct/ | About 3.10 GB |
| 2 — size comparison and memory probe | [Qwen/Qwen2.5-Coder-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct) | models/Qwen2.5-Coder-3B-Instruct/ | About 6.18 GB |
| 3 — largest size comparison and memory probe | [Qwen/Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct) | models/Qwen2.5-Coder-7B-Instruct/ | About 15.20 GB |

These are the existing planning allowances, not measured transfer totals. The complete model inventory and eventual pinned revisions are recorded in manifests/planned_downloads_4090.json. The audited downloader writes file hashes and revisions to manifests/model_files.json.

After the Linux training environment is installed, download the first model from the project folder with:

~~~bash
cd "/mnt/c/Users/weita/Desktop/SE in AI/proj"
.envs/train/bin/python scripts/study.py download --kind model --only 1.5B
~~~

For a download from the existing Windows validation environment in a network-enabled PowerShell session:

~~~powershell
.\.envs\validation\Scripts\python.exe -X utf8 scripts/study.py download --kind model --only 1.5B
~~~

## Files and evidence

| Location | Contents |
| --- | --- |
| src/swe_study/ | Download, auditing, data, tokenization, training, serving, and evaluation code |
| configs/ | Active agent settings, 1.5B overfit/pilot, and 3B/7B memory-probe settings |
| manifests/ | Approved inventory, storage baseline, versions, asset failures, and implementation status |
| results/setup/setup-2026-10-19T16-35-12-041Z/ | Session events and numbered commands with full stdout/stderr, arguments, times, and exit codes |
| results/<stage>/<run-id>/ | Per-run config, events, stdout/stderr, metrics, source snapshots, and failure tracebacks |
| checkpoints/pilot/synthetic-tiny-qwen/ | Synthetic adapter and the two most recent Trainer checkpoints; never a project model |
| data/, models/, vendor/ | Approved real-asset destinations; real payloads are not present yet |
| docs/environment.md | Hardware, software, network/WSL/Docker findings |
| docs/data_audit.md | Data processing rules and real-data checks still required |
| docs/pilot_report.md | Measured synthetic GPU results and their limits |

Thirty regression tests passed in the final validation. Refresh the point-in-time file inventory with study.py inventory.

Every study command creates a new run ID. It preserves failures and records configuration hashes; newer runs also save a source/config/test/manifest ZIP with a SHA256. Checkpoints include optimizer, scheduler, RNG, and Trainer state; adapters are saved without duplicate full base weights. Resume is explicit, and nonempty training outputs are preserved. Local logs and large artifacts are ignored by Git but remain on disk. Recognized credential keys are redacted from structured events; raw command logs are local research artifacts.

## Continue the approved setup

Use the verified Ubuntu route for Linux setup and integration. The Windows commands below remain available for the existing validation environment. The existing validation environment reuses installed system packages; it is not an independent dependency installation. Its exact package inventory is manifests/validation_packages.freeze.txt.

~~~powershell
$py = '.\.envs\validation\Scripts\python.exe'
& $py -X utf8 scripts/study.py preflight
& $py -X utf8 scripts/continue_setup.py --stage assets
& $py -X utf8 scripts/continue_setup.py --stage data
& $py -X utf8 scripts/continue_setup.py --stage overfit
& $py -X utf8 scripts/continue_setup.py --stage pilot
& $py -X utf8 scripts/continue_setup.py --stage larger-probes
~~~

Run stages in this order and stop on a failure. The assets stage fetches pinned source checkouts, the 1.5B model, and all five metadata/trajectory datasets. The larger-probes stage downloads 3B and 7B sequentially only after a real 1.5B pilot artifact exists. Each command checks storage; verified files are reused and interrupted Hugging Face transfers use the SDK's resumable local download behavior. Original snapshots total about 24.48 GB and raw datasets about 0.40 GB; 4-bit GPU loading does not reduce those snapshot downloads.

The data stage freezes five development IDs disjoint from all Verified tasks and 100 repository-proportional Verified IDs with seed 6158. It prepares task-disjoint successful SWE-Gym histories and reports context coverage and assistant target masks. It refuses empty retained splits and does not silently truncate long histories. Review real schema, action format, mask samples, and retained coverage before treating the data as training-ready.

Overfit and pilot stages refuse to overwrite existing output directories. For an interrupted run, pass the actual retained Trainer checkpoint with --resume. This resumes optimizer/scheduler/RNG state; restarting a new schedule is a separate experiment. Inspect losses and exclusions rather than treating a successful exit as proof of useful learning.

The user's interactive Ubuntu terminal reports Python 3.11.6, while the agent's non-interactive Ubuntu runner resolves /usr/bin/python3 to Python 3.12.3. The existing system PyTorch 2.6.0+cu124 passed a CUDA tensor check. Use /usr/bin/python3 explicitly to create the separate project environment; the full Linux training dependencies still need installation and validation:

~~~bash
cd "/mnt/c/Users/weita/Desktop/SE in AI/proj"
/usr/bin/python3 scripts/bootstrap.py --name train --install
.envs/train/bin/python scripts/study.py preflight
.envs/train/bin/python scripts/continue_setup.py --stage assets
~~~

Bootstrap writes its installation logs, package freeze, and preflight report locally. Linux compatibility and the older OpenHands environment are still unvalidated. Install the pinned official evaluator and OpenHands fork in separate project environments after inspecting their lockfiles and Python requirements. The Windows validation freeze must not be presented as their Linux lockfile. The implemented SFT path uses Transformers Trainer plus PEFT and explicit labels; it does not require TRL.

## Ubuntu readiness and remaining Docker/agent checks

User-provided terminal output confirms Ubuntu 24.04.2 WSL2, Python 3.11.6, RTX 4090 visibility, and a reachable Docker 26.1.3 server using /var/lib/docker. The GPU reported 10,966 MiB used out of 24,564 MiB, leaving about 13.3 GiB at that measurement; python3.11 and ollama were among the listed processes. Existing processes and containers were left untouched. The agent subsequently verified Linux PyTorch CUDA execution and Docker connectivity. The physical Docker storage baseline and project image/runtime controls remain unverified. [Environment evidence](docs/environment.md).

The OpenHands scaffold requires Linux/WSL2 and Docker. Keep its environment separate from training and the official evaluator. Run the model server and scaffold on the same Linux host; Windows-to-WSL localhost forwarding has not been verified.

Before any image operation, record physical Docker storage with study.py docker-storage. For Docker Desktop/WSL, supply --backing-file followed by the actual existing Docker VM backing-file path, expressed for the shell in use. The script records this operator-supplied mapping and host-volume free space; native local Linux Docker can use its DockerRootDir. It refuses remote daemons. It does not relocate storage or prune existing images. Docker growth on a different volume is added to the cap, and each affected volume keeps the free-space floor.

After dependency and source compatibility checks, use the CLI entries below. The --python arguments select the separate agent/evaluator environments. These integration steps are not yet validated:

~~~bash
.envs/train/bin/python scripts/study.py grade --control gold --python .envs/eval/bin/python
.envs/train/bin/python scripts/study.py grade --control blank --python .envs/eval/bin/python
.envs/train/bin/python scripts/study.py serve --model models/Qwen2.5-Coder-1.5B-Instruct
# In another Linux terminal:
.envs/train/bin/python scripts/study.py rollout --model-name cs6158-Qwen2.5-Coder-1.5B-Instruct --python .envs/agent/bin/python
~~~

The server binds 127.0.0.1:8000, uses NF4 for both baseline and adapter conditions, reserves output within an 8,192-token context, and enforces a 32,768 completion-token budget per frozen task. Use one fresh server run per model/adapter condition; its persistent ledger is scoped to that run. Requests that cannot identify one frozen issue fail rather than resetting a task budget. The scaffold limits tasks to 30 steps, 30 minutes, and one worker. Exact action compatibility still needs real-data replay.

The official grading wrapper verifies pinned checkouts and task metadata identity, keeps gold material in grader-only files, fills missing predictions with empty patches, and validates report denominators and control outcomes. Metadata sanitation and fixture tests do not establish complete container/image isolation. Audit actual mounts, runtime images, hidden-test exposure, and the action-to-patch-to-grader path before reporting benchmark results.

## Verification

~~~powershell
& $py -X utf8 -m pytest -q --basetemp .cache/tmp/pytest
& $py -X utf8 scripts/study.py validate-checkpoint --adapter checkpoints/pilot/synthetic-tiny-qwen
& $py -X utf8 scripts/validate_local_server.py
~~~

The last two commands use only the synthetic fixture. Do not use its timing, memory, or loss as a production estimate. [Pilot evidence](docs/pilot_report.md) lists measured results and the remaining real-model validation.

The [file cleanup audit](results/cleanup/cleanup-2026-10-19T17-16-59-166Z/run.json) records the full file review, removals, preserved evidence, and verification. The 4090 implementation plan is the current planning document.
