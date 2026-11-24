# CS6158 Qwen QLoRA and SWE agent study

## Current NERSC results

Status at the **2026-11-24 03:16 UTC** NERSC export: the real 1.5B, 3B and 7B
QLoRA runs completed three epochs and 150 optimizer steps. Each used 400 retained
training histories, 43 retained validation histories and effective batch size 8.
All three adapters improve held-out assistant-response prediction over their
matching base models. The frozen agent benchmark remains incomplete.

| Model | Base validation loss | LoRA validation loss | Relative loss reduction | Validation status |
| --- | ---: | ---: | ---: | --- |
| 1.5B | 0.683364 | 0.446510 | 34.66% | Passed; matches training validation |
| 3B | 0.585693 | 0.408810 | 30.20% | Passed; matches training validation |
| 7B | 0.547683 | 0.372860 | 31.92% | Passed after reproducing the two-GPU training aggregation |

These are cross-entropy losses on **43 histories covering 25 distinct issues**,
selected from 50 validation histories using the same 32,768-token context rule.
Seven over-context histories were excluded. The processed dataset hashes match
the imported NERSC records; training, validation and frozen benchmark issue IDs
are disjoint. The validation set also selected the best checkpoints, so these
results do not establish SWE-bench issue-resolution performance.

The 7B training loss `0.384927124` and canonical single-GPU validation loss
`0.372860312` use different reductions. The later validation run reconstructed
the training reduction as `0.384927164` and passed. Keep the earlier failed
comparison as historical evidence; use the later successful report.

| Agent evaluation in the imported snapshot | Recorded result |
| --- | --- |
| Frozen development suite | 5 tasks, base and LoRA: 10 episodes per model size |
| 1.5B environments prepared | 2/5: pvlib 807 and pydicom 1048 |
| 1.5B episodes graded | 2/10: the first pvlib task under both conditions |
| First task outcomes | Both unresolved; base looped, LoRA reached its 30-step limit |
| Pydicom | Gold and runtime controls passed; no saved agent prediction yet |
| 3B and 7B agent comparisons | No results included in this export |
| Frozen final suite | 100 tasks; final evaluation has not been established by this export |

The LoRA patch on the first task was a placeholder implementation, while the
base patch was empty. Lower validation loss has therefore not yet translated
into a demonstrated agent-resolution gain. Keep the three-epoch checkpoints
and complete the frozen development comparisons before adding training epochs.

**Reporting corrections:** the 3B run resumed at step 13, and the final 7B run
resumed at step 147. Their runtime, GPU-hour and token-counter fields cover those
last execution segments. In particular, 7B's `0.145` GPU-hours and `train_loss`
`0.00925` are not complete-training measurements. Aggregate earlier attempts
for total compute cost, and use held-out losses or the full per-step history
when comparing learning. Memory peaks also describe the measured segment/rank.

Evidence and detailed findings:

- [NERSC results and challenges resolved](docs/nersc_results_and_challenges.md): detailed symptoms, causes, fixes, acceptance evidence and remaining limits.
- [Imported NERSC records](results/nersc_uploaded/20261124T031613Z/): training metadata, validation reports, prepared environments, task outcomes and token ledgers.
- [Numerical and coverage audit](results/nersc_results_review/20261124T032614-36127ffd/review.json): 38 JSON inputs, their SHA-256 hashes and 63 passing consistency checks.
- [Current NERSC machine status](manifests/nersc_status.json) and [implementation progress](docs/implementation_status.md).
- [NERSC evaluation operations](cloud/EVALUATION_NERSC.md): CPU preparation, paired GPU serving, tunnels, prediction and official grading.

## Execution environments and current workflow

Local development uses Ubuntu WSL2. NERSC execution uses native Linux on
Perlmutter. Select the project interpreters explicitly: `.envs/train/bin/python`
for training and serving, `.envs/agent/bin/python` for OpenHands, and
`.envs/eval/bin/python` for grading and monitoring. A shell's default `python3`
may be too old for these scripts.

On NERSC, `$HOME/swe-study-source` is the Git clone and
`$SCRATCH/swe-study-a100` is the extracted execution project. Pull and push from
the clone. Install source updates into scratch with the checked
`deliverables/nersc_evaluation_patch.py` installer; it backs up recognized source
versions and preserves unfamiliar edits. Copy selected result records from
scratch into the clone before committing them. A source-bundle manifest alone
does not upload training or benchmark results.

The current sequence is **CPU prepare -> GPU-backed predict -> CPU grade ->
summary**. A paired server exposes base and LoRA aliases on loopback port 8615,
with NF4 weights, matching precision, an 8K total context and persistent token
ledgers. Training used 32K complete histories; the 8K agent protocol is a separate,
explicit evaluation setting. Reuse the existing ledger directory after a GPU
allocation changes. Keep the CPU node and its task images available.

To monitor the current 1.5B study on its CPU node:

```bash
cd "$SCRATCH/swe-study-a100"
.envs/eval/bin/python scripts/benchmark_progress.py \
  --state results/benchmark_15b_window_v1 --watch
```

The monitor refreshes saved records; a recorded `running` status is not proof
of process liveness. Inspect the recorded PID, child process and inner build
logs when a preparation stage is quiet. The 30-step bar is an episode limit;
the development denominator remains five tasks and ten episodes per model.

The approved 70 GB incremental storage cap and 25 GB free-space floor still
apply. They are project safeguards, separate from NERSC quota. The scratch
project already resides on `/pscratch`; changing its path does not create more
budget. Preserve models, checkpoints, logs and token ledgers. Verify durable
archive copies before removing staging archives; this export does not confirm
completion of an HPSS transfer.

## Historical workstation setup

For a cloud A100 node (including multiple GPUs), use the portable
[cloud training and inference guide](cloud/README.md). Package a clean upload with
`/usr/bin/python3 scripts/package_cloud.py` in Ubuntu; it excludes workstation
environments, machine-specific storage accounting, credentials, weights, and logs.

The original setup covered the first two proposal milestones on one RTX 4090.
The user subsequently authorized the NERSC training and evaluation work recorded
above. Full five-task agent acceptance remains incomplete. The sections below
retain the workstation setup and pilot procedures; dated hardware measurements
describe that host. See [current progress](docs/implementation_status.md),
[implementation plan](docs/rtx_4090_implementation_plan.md), and
[4090 constraints](docs/rtx_4090_constraints.md).

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

The historical workstation regression suite passed 38 tests. The later cloud
evaluation regression suite passed **91 tests** on 2026-11-24 UTC, and the checked
installer validation passed. Those CPU tests establish compatibility and
reporting behavior; real NERSC training and evaluation have their own records.
Earlier synthetic NF4 checks remain mechanics checks. Consult
[pilot_report.md](docs/pilot_report.md) for the historical workstation GPU results.

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

The historical single-condition procedure started a fresh server for each
condition. Current NERSC evaluation uses one paired server with separate base
and LoRA aliases and persistent ledgers for each condition. Both use
127.0.0.1:8615, an 8192-token total context and a 32768-completion-token budget
per task. The scaffold caps an episode at one worker, 30 steps and 30 minutes.
Emulated XML tool calls stay in text for OpenHands to parse; native tool calls
are emitted only when the request supplied native tools.

## Files and logs

| Location | Contents |
| --- | --- |
| src/swe_study/, scripts/, configs/, tests/ | Pipeline implementation, settings, and regression checks |
| models/, data/, vendor/ | Verified snapshots, raw/processed datasets, and pinned sources |
| manifests/ | Revisions, checksums, task lists, package freezes, storage, current status |
| results/<stage>/<run-id>/ | Commands, stdout/stderr, errors, events, metrics, source/config snapshots |
| checkpoints/pilot/, checkpoints/cloud/ | Historical pilot adapters and NERSC adapters/checkpoints; synthetic outputs are marked separately |
| results/nersc_uploaded/, results/nersc_results_review/ | Imported NERSC result snapshots and independent numerical reviews |
| docs/ | Environment, progress, constraints, data audit, and pilot evidence |

Every study command creates a fresh audited run. Logs and large artifacts stay local even when ignored by Git. Structured events redact recognized credential keys. All historical failures and synthetic evidence are preserved. The [file cleanup audit](results/cleanup/cleanup-2026-10-19T17-16-59-166Z/run.json) records earlier removals and retained evidence.

~~~bash
.envs/train/bin/python -B -m pytest -q --capture=sys -p no:cacheprovider --basetemp .cache/tmp/pytest
.envs/train/bin/python scripts/study.py inventory
~~~

Python-level test capture avoids the observed WSL/NTFS temporary-file failure in descriptor capture.
The full nine-adapter matrix and completed final benchmark results have not been
established. Historical workstation setup limits do not override later explicit
user authorization for NERSC work. Global Docker/WSL migration and unrelated
cleanup remain outside the requested work.
