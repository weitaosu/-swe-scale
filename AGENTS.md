# Project workspace instructions

## Default execution environment

The user requested that future work use the existing Ubuntu 24.04.2 WSL2 environment. Use it by default for setup, package installation, Git/network operations, data preparation, training, inference, Docker, and evaluation.

- Launch wsl.exe through the standard exec_command runner. This route has been verified to work. Node REPL child-process launches of wsl.exe returned E_ACCESSDENIED; use Node only for file I/O when needed, not as the Ubuntu command launcher.
- Shared Linux project root: /mnt/c/Users/weita/Desktop/SE in AI/proj.
- Shared Windows project root: C:/Users/weita/Desktop/SE in AI/proj. Keep model/data/cache/checkpoint/log files in this project.
- Bootstrap with /usr/bin/python3, verified as Python 3.12.3. The interactive shell reports a different Python 3.11.6, so choose the interpreter explicitly.
- Use the installed .envs/train/bin/python for project training/data/server commands. Keep agent and evaluator dependencies in their separate project Linux environments. The obsolete Windows .envs/validation environment was removed during cleanup; its historical logs and archived creation settings remain available.
- A Windows helper failure or Node child-process access error does not establish that Ubuntu is unavailable. Test the standard wsl.exe route before declaring an access blocker.

Known working preflight command, issued through the standard command runner:

~~~text
wsl.exe --cd "/mnt/c/Users/weita/Desktop/SE in AI/proj" --exec env PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 scripts/study.py preflight
~~~

As verified on 2026-10-19, this route can access the shared project, the RTX 4090 through PyTorch/CUDA, Docker 26.1.3, Hugging Face HTTPS, and GitHub. Recheck current package availability before using an environment; the full Linux training stack was not installed by the access check. Evidence and current state are in docs/environment.md and manifests/implementation_status.json.

## Tracking and scope

- Track commands, stdout/stderr, failures, versions, configurations, dataset/model revisions, and metrics in the project logs/manifests. Use the audited scripts/study.py and stage helpers where available.
- Preserve the approved 70 GB incremental storage cap and 25 GB free-space floor. Record the physical Docker storage baseline before project image operations.
- Preserve existing logs, source snapshots, the proposal, and required checkpoints. Avoid broad Docker pruning, unrelated-file removal, and stopping unrelated workloads.
- Keep synthetic compatibility checks distinct from real-model/data validation. Both first milestones require their documented acceptance checks before being marked complete. The full nine-run training matrix and full benchmark evaluations are outside the current staged setup scope.

## Current validated runtime details

- Use project model API port 8615 on 127.0.0.1; port 8000 was already occupied.
- The 8K pilot uses target-only output logits with a verified equivalent assistant causal loss. Preserve this setting for memory comparisons.
- NF4 inference and reload preserve FP32 unquantized layers and use BF16 autocast to match training.
- Project runtime builds/pulls use the audited compatibility overlay and local Buildx config; browsing tools are disabled but the old runtime requires Chromium for startup.
- Baseline/adapter pipeline validation remains separate from scripted tool replay. Check current progress before claiming either milestone complete.

## WSL validation and bounded development checks

- Run pytest with --capture=sys -p no:cacheprovider and a project-local --basetemp. File-descriptor capture uses unlinked temporary files that fail on this Windows-mounted workspace.
- Sequential API port probes must set SO_REUSEADDR so a stopped server's TIME_WAIT connections do not block the next model. A live listener must still be preserved.
- --smoke always selects the first frozen development task, with five steps and 300 seconds. It is an integration diagnostic and does not satisfy or replace the frozen five-task comparison. Keep its denominator and reports separate.
- Task image dependency repairs require a passing gold control and recorded image identity. Blank predictions are short-circuited by the official grader and do not execute a negative test control.

- Run full file inventory after GPU jobs finish. During this validation, a simultaneous NTFS inventory scan coincided with much slower 7B loading; pause inventory during model loading and resume it after GPU work finishes to avoid competing I/O.

- For a full index of this Windows-backed project, prefer native Windows filesystem reads for metadata. The completed native inventory scanned 111,781 entries in 3.4 seconds; the WSL walk was cancelled because NTFS metadata traversal was slow. Exact static indexer source and audited driver evidence are in results/native_inventory/20261019T224817-43e7e724/. Compute, model, data processing, network, and Docker work still use Ubuntu.
