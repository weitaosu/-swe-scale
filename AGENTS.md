# Project workspace instructions

## Default execution environment

The user requested that future work use the existing Ubuntu 24.04.2 WSL2 environment. Use it by default for setup, package installation, Git/network operations, data preparation, training, inference, Docker, and evaluation.

- Launch wsl.exe through the standard exec_command runner. This route has been verified to work. Node REPL child-process launches of wsl.exe returned E_ACCESSDENIED; use Node only for file I/O when needed, not as the Ubuntu command launcher.
- Shared Linux project root: /mnt/c/Users/weita/Desktop/SE in AI/proj.
- Shared Windows project root: C:/Users/weita/Desktop/SE in AI/proj. Keep model/data/cache/checkpoint/log files in this project.
- Bootstrap with /usr/bin/python3, verified as Python 3.12.3. The interactive shell reports a different Python 3.11.6, so choose the interpreter explicitly.
- Once created, use .envs/train/bin/python for project training/data/server commands. Keep agent and evaluator dependencies in their separate project Linux environments. The Windows .envs/validation environment is historical validation evidence.
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
