# Current project progress

Recorded 2026-10-19. We have completed the local code foundation and synthetic GPU checks. **Both project milestones remain incomplete.**

## Milestone progress

| Milestone | Completed locally | Still pending |
| --- | --- | --- |
| Sep 14–27: SWE-bench, Qwen models, agent scaffold | Project structure, resumable download scripts, local model server, OpenHands/grader wrappers, task-selection code, storage guards, and logging | Download all three Qwen models and pinned source repositories; install/validate separate Linux agent/evaluator environments; validate Docker controls and real agent runs |
| Sep 28–Oct 11: SWE-Gym preparation and LoRA validation | Provenance recovery and quarantine code, overlap screening, task-disjoint splitting, assistant-only loss masking, NF4 QLoRA training, best-adapter selection, and explicit checkpoint resume | Download and audit real datasets; inspect real Qwen templates and context coverage; run the real 1.5B overfit/pilot and 3B/7B memory probes; validate the adapter through the agent and grader |

## Verified results

- 30 regression tests passed, and all project Python modules compiled during implementation.
- The local RTX 4090 supports CUDA tensor operations and BF16 in the installed Windows PyTorch environment.
- Agent checks now confirm direct Ubuntu access, /usr/bin/python3 3.12.3, a working PyTorch 2.6.0+cu124 CUDA tensor operation, Docker 26.1.3, Hugging Face HTTP 200, and GitHub remote access. The interactive shell separately reported Python 3.11.6.
- NF4 LoRA training, finite nonzero adapter gradients, adapter updates, optimizer/Trainer resume, and fresh-process reload passed using a random tiny Qwen2 model and synthetic text.
- The model API passed in-process HTTP tests with real GPU generation, including streaming, context limits, task identification, and cumulative output-token limits.
- Commands, failures, versions, configurations, metrics, source snapshots, and a file inventory are saved locally.

These checks establish local implementation/runtime mechanics. They do not establish real Qwen2.5-Coder model fit, SWE-Gym training quality, production runtimes, or benchmark performance. Actual networked OpenHands integration is untested.

## Download and execution status

| Item | Current state |
| --- | --- |
| Qwen2.5-Coder 1.5B / 3B / 7B | 0 of 3 real model snapshots downloaded |
| SWE-Gym and SWE-bench payloads | No real dataset payloads downloaded |
| Pinned external source checkouts | Not downloaded |
| Project Docker images / benchmark rollouts | No project images pulled or built; no real agent/grading runs. The user's existing Docker server has 3 containers and 3 images |
| Real-data LoRA pilot | Not started |

The agent can now enter Ubuntu through the standard command runner and access the shared project folder, GPU via PyTorch, Docker, Hugging Face, and GitHub. Earlier access failures were observed through another execution route. Linux project dependencies, real asset downloads, physical Docker storage accounting, and the project agent/grader workflow remain pending. See [environment.md](environment.md) for the verified route and evidence.

## Next steps

1. Create the project Linux training environment using /usr/bin/python3 (3.12.3), validate its package stack, and resume the approved assets stage through the working Ubuntu runner.
2. Download 1.5B and the datasets, then freeze task lists and inspect real data, provenance, labels, and 4K context coverage.
3. Run the 1.5B overfit and bounded pilot; verify reload/resume and measure actual tokens, time, and memory.
4. Download and probe 3B/7B sequentially after the 1.5B pilot succeeds.
5. Use the existing Ubuntu 24.04.2 WSL2 installation, use the verified command runner, install the separate agent/evaluator environments, record physical Docker storage, and validate project image/runtime controls and the full development workflow.

~~~powershell
.\.envs\validation\Scripts\python.exe -X utf8 scripts/continue_setup.py --stage assets
~~~

Run from the project folder. The approved storage limit remains 70 GB incremental growth while keeping at least 25 GB free; original model snapshots are estimated at 24.48 GB and raw datasets at 0.40 GB. Follow [README.md](../README.md) for stage commands and [rtx_4090_implementation_plan.md](rtx_4090_implementation_plan.md) for the retained plan.

## Evidence and file review

- [Synthetic GPU pilot evidence](pilot_report.md)
- [Real-data audit status](data_audit.md)
- [Machine-readable implementation status](../manifests/implementation_status.json)
- [Initial setup session](../results/setup/setup-2026-10-19T16-35-12-041Z/session.json)
- [Artifact inventory](../manifests/artifact_inventory.json)
- [File cleanup audit](../results/cleanup/cleanup-2026-10-19T17-16-59-166Z/run.json)

The cleanup preserves the proposal, working source/tests/configurations, all historical logs and source snapshots, and the synthetic base model plus adapters/checkpoints needed to reproduce the completed checks. It removes regenerable caches/test scratch files and superseded or unused documents/configurations. Cleanup does not advance either milestone's real-data acceptance criteria.
