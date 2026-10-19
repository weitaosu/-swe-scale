# CS6158: local SWE agent and QLoRA setup

Approved scope: the first two proposal milestones on one RTX 4090, with a 70 GB incremental storage cap and at least 25 GB free. Both milestones remain incomplete until their real-model, agent, and grader acceptance checks pass. See [current progress](docs/implementation_status.md), [implementation plan](docs/rtx_4090_implementation_plan.md), and [4090 constraints](docs/rtx_4090_constraints.md).

Use Ubuntu 24.04.2 WSL2 through the standard command runner, following [AGENTS.md](AGENTS.md) and [environment.md](docs/environment.md). Shared project root: /mnt/c/Users/weita/Desktop/SE in AI/proj. Select interpreters explicitly: /usr/bin/python3 bootstraps Python 3.12.3 environments; .envs/train/bin/python handles training/data/serving, with separate .envs/agent and .envs/eval interpreters.

## Models and download links

All three user-downloaded snapshots passed pinned remote file-size and LFS SHA-256 checks on 2026-10-19. Existing weights were reused. These are original Transformers/Safetensors snapshots; NF4 quantization happens at GPU load time.

| Model | Official repository | Local directory | Planning allowance |
| --- | --- | --- | ---: |
| 1.5B: first pilot | [Qwen/Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct) | models/Qwen2.5-Coder-1.5B-Instruct/ | 3.10 GB |
| 3B: size comparison | [Qwen/Qwen2.5-Coder-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct) | models/Qwen2.5-Coder-3B-Instruct/ | 6.18 GB |
| 7B: largest comparison | [Qwen/Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct) | models/Qwen2.5-Coder-7B-Instruct/ | 15.20 GB |

[Exact sizes, revisions, and hashes](manifests/model_files.json); [verification run](results/model_integrity/20261019T180039-a32aa4d0/run.json). The downloader verifies and reuses matching local files; a mismatched existing file is preserved and reported.

## Real assets and data

The five SWE-Gym/SWE-bench datasets are local and checksum-verified. [Dataset manifest](manifests/dataset_files.json). All three external repositories are checked out at [pinned commits](manifests/source_repositories.json).

The initial real-data audit retained 489 of 491 successful trajectories: 439 training trajectories from 263 tasks and 50 validation trajectories from 29 tasks. Two ambiguous records were quarantined. Splitting is by task, and the overlap screen includes all Verified tasks and the frozen development tasks. [Data audit](docs/data_audit.md). The five development IDs and 100 final evaluation IDs are frozen with seed 6158; this setup does not launch full benchmark evaluation.

38 Linux regression tests passed. Earlier synthetic NF4 training/resume/reload/server results remain historical mechanics checks. They do not establish real-model feasibility. Consult [pilot_report.md](docs/pilot_report.md) for measured GPU results and outstanding checks.

## Run the approved stages in Ubuntu

~~~bash
cd "/mnt/c/Users/weita/Desktop/SE in AI/proj"
/usr/bin/python3 scripts/bootstrap.py --name train --install
/usr/bin/python3 scripts/bootstrap_scaffolds.py --name agent
/usr/bin/python3 scripts/bootstrap_scaffolds.py --name eval
.envs/train/bin/python scripts/study.py preflight
.envs/train/bin/python scripts/continue_setup.py --stage data
.envs/train/bin/python scripts/continue_setup.py --stage overfit
.envs/train/bin/python scripts/continue_setup.py --stage pilot
.envs/train/bin/python scripts/continue_setup.py --stage larger-probes
~~~

Run stages sequentially and inspect their logs. The current installation/validation state is recorded in the progress document. Bootstrap preserves existing environments and writes installation logs and exact package freezes. Agent dependencies derive from the fork's lock; LiteLLM 1.52.5 was unavailable on PyPI, so the installer records its 1.53.1 fallback within the fork's declared range. Compatibility remains subject to integration checks.

To verify or fetch assets again:

~~~bash
.envs/train/bin/python scripts/study.py checkout
.envs/train/bin/python scripts/study.py download --kind model
.envs/train/bin/python scripts/study.py download --kind dataset
~~~

Data preparation preserves task/tool provenance and quarantines ambiguity. Tokenization uses assistant-only labels, reports full-history context exclusions, and refuses empty retained splits. It does not silently truncate. Inspect actual 4K coverage before selecting training data or a production context policy.

Training uses Transformers Trainer and PEFT, frozen NF4 base weights, BF16 computation, LoRA rank 16, gradient checkpointing, microbatch 1, and no CPU offload. Overfit/pilot stages preserve nonempty outputs. Resume an interrupted Trainer run with --resume and a complete retained checkpoint; optimizer, scheduler, RNG, and step state must be present. Save/reload validation uses a fresh model process and fixed-logit tolerances.

## Agent and grading checks

Physical Docker storage is recorded in [docker_storage.json](manifests/docker_storage.json). Its WSL VHDX resides on the project volume and is included in the shared storage budget. The first official development image was downloaded and [pinned by digest](manifests/development_images.json). Other existing containers and GPU workloads are left running.

The first two official pvlib images required dependency-only NumPy repairs and now pass gold controls. Real runtime view/edit/test/patch replay and isolation checks passed. Full five-task execution is gated by storage and SQLFluff image access. See [development environment constraints](docs/development_environment_constraints.md). [Completed single-task diagnostics](docs/agent_smoke_report.md) use --smoke and remain separate from five-task validation; no benchmark score is established.

~~~bash
.envs/train/bin/python scripts/study.py grade --control gold --python .envs/eval/bin/python
.envs/train/bin/python scripts/study.py grade --control blank --python .envs/eval/bin/python
.envs/train/bin/python scripts/study.py serve --model models/Qwen2.5-Coder-1.5B-Instruct
# Full five-task rollout requires the remaining image/storage gates to be resolved:
.envs/train/bin/python scripts/study.py rollout --model-name cs6158-Qwen2.5-Coder-1.5B-Instruct --python .envs/agent/bin/python
~~~

After existing GPU jobs have finished, a bounded first-task diagnostic can be reproduced with:

~~~bash
.envs/train/bin/python scripts/run_development_condition.py --model models/Qwen2.5-Coder-1.5B-Instruct --adapter checkpoints/pilot/qwen-1.5b --label pilot-1.5b-smoke --rollout --smoke
~~~

This uses the verified cached runtime, five steps, and a 300-second task limit. It produces separate smoke reports and keeps the five-task manifest unchanged.

The server uses 127.0.0.1:8615, NF4 for matching baseline/adapter conditions, an 8192-token total context, and a persistent 32768-completion-token budget per frozen task. Start a fresh server for each condition. The scaffold caps execution at one worker, 30 steps, and 30 minutes per task. Emulated XML tool calls stay in text for OpenHands to parse; native tool calls are emitted only when the request supplied native tools.

## Files and logs

| Location | Contents |
| --- | --- |
| src/swe_study/, scripts/, configs/, tests/ | Pipeline implementation, settings, and regression checks |
| models/, data/, vendor/ | Verified snapshots, raw/processed datasets, and pinned sources |
| manifests/ | Revisions, checksums, task lists, package freezes, storage, current status |
| results/<stage>/<run-id>/ | Commands, stdout/stderr, errors, events, metrics, source/config snapshots |
| checkpoints/pilot/ | Adapters and bounded Trainer resume checkpoints; synthetic outputs are marked separately |
| docs/ | Environment, progress, constraints, data audit, and pilot evidence |

Every study command creates a fresh audited run. Logs and large artifacts stay local even when ignored by Git. Structured events redact recognized credential keys. All historical failures and synthetic evidence are preserved. The [file cleanup audit](results/cleanup/cleanup-2026-10-19T17-16-59-166Z/run.json) records earlier removals and retained evidence.

~~~bash
.envs/train/bin/python -B -m pytest -q --capture=sys -p no:cacheprovider --basetemp .cache/tmp/pytest
.envs/train/bin/python scripts/study.py inventory
~~~

Python-level test capture avoids the observed WSL/NTFS temporary-file failure in descriptor capture. The full nine-adapter matrix, full benchmark runs, paid cloud GPUs, global Docker/WSL migration, and unrelated cleanup remain outside this staged setup.
