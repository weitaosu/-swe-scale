# Current project progress

Updated 2026-10-19. Asset setup, real-data preparation, and GPU training/checkpoint validation have passed. **Both milestones remain incomplete. Full five-task development execution is gated by Docker image access and the approved storage allowance; cached single-task integration diagnostics have completed.**


| Area                | Completed                                                                                                                                                                       | Remaining                                                               |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| Models/source       | Three Qwen snapshots verified; five datasets local; three source repositories pinned                                                                                            | No additional model download needed                                     |
| Ubuntu environments | Isolated training, agent, and evaluator installations; exact package freezes saved                                                                                              | Continue using the explicit WSL interpreters                            |
| Data                | 489/491 successful histories retained; 439 training / 50 validation; task-disjoint split; two ambiguous records quarantined                                                     | Production context policy for histories longer than 8K                  |
| 1.5B LoRA           | Four-example overfit; 32-example 8K pilot for 25 steps; optimizer continuation from 25 to 27; exact fresh reload                                                                | Full frozen-five-task adapter comparison                                |
| 3B / 7B             | Two training steps each on long 8K histories; finite loss, nonzero gradients, updated adapters, exact fresh reload                                                              | Full-run throughput/quality measurements are outside these short probes |
| Agent runtime       | All three released tool types converted; real container view/edit/test/patch replay passed; no host mounts, no privileged mode, loopback-only API ports                         | Full frozen-five-task model comparison                                  |
| Grader              | Two pvlib task images repaired by pinning NumPy only; both gold controls passed; executed negative control produced the expected one failure/four passes; blank handling passed | Remaining three task environments and full five-task grading            |
| API/tests           | All three model APIs and the pilot-adapter API passed; 38 Linux regression tests passed; four bounded smoke conditions graded                                                   | Full five-task comparison                                               |

## Measured 8K training feasibility on the RTX 4090


| Model | Scope                             | Peak allocated | Peak reserved | Training/eval/checkpoint time |
| ----- | --------------------------------- | -------------: | ------------: | ----------------------------: |
| 1.5B  | 25 steps, 32 train / 9 validation |       6.85 GiB |     16.63 GiB |                       279.9 s |
| 3B    | 2 steps, 8 train / 2 validation   |       8.57 GiB |     13.59 GiB |                        19.5 s |
| 7B    | 2 steps, 8 train / 2 validation   |      14.81 GiB |     18.61 GiB |                        26.3 s |

Times exclude model loading and fresh reload. All use NF4, BF16 computation, LoRA rank 16, gradient checkpointing, and the tested assistant-target-only output-logit path. The two-step probes do not establish steady-state runtime or training quality. [Detailed pilot evidence](pilot_report.md).

The initial full-logit 8K attempt saturated VRAM and was stopped before an optimizer step. The optimized path preserves the causal loss and parameter gradients in an equivalence test. Fresh reload initially exposed precision/autocast differences; these were corrected, and all real reload comparisons now have maximum logit difference 0.0 without widening tolerance.

## Remaining development workflow

The initial five-task sequence was stopped after a second task image failed its gold control under NumPy 2. That task is now repaired and gold-validated. A registry audit found pydicom and astroid images available but uncached, and SQLFluff returned HTTP 401. The next uncached-image gold check was stopped by the 70 GB storage guard. About 45.19 GB was free at the image audit, leaving 12.31 GB under the approved growth cap. [Exact findings and storage plan](development_environment_constraints.md).

Real API checks and a bounded smoke sequence completed using the already cached, gold-validated first frozen task, pvlib__pvlib-python-807. The smoke settings are five steps and 300 seconds, explicitly separate from the full five-task comparison. Baseline 1.5B, its pilot adapter, 3B, and 7B ran sequentially with fresh servers/ledgers, full logs, patch export, and official grading. The frozen five-task list is unchanged, and smoke results do not satisfy its acceptance check. Failures, empty patches, and missing predictions remain explicit. The full nine-adapter study and final 100-task runs are outside this setup.

Only 46/489 histories fit 8K (9.4%): 37 train / 9 validation. The user selected complete 8K histories for this bounded pilot if memory permits. This short-history subset is not a representative replacement for the full training corpus. [Data audit](data_audit.md).

Use the existing Ubuntu WSL2 environment and project model API 127.0.0.1:8615. Port 8000 was already occupied. The storage cap remains 70 GB incremental growth with at least 25 GB free; Docker's WSL VHDX is included. [Environment](environment.md), [live storage](../manifests/storage_usage.json).

Evidence: [model manifest](../manifests/model_files.json), [datasets](../manifests/dataset_files.json), [source pins](../manifests/source_repositories.json), [25-step pilot](../results/train/20261019T212429-33c80784/metrics.json), [resume validation](../results/resume_validation/20261019T213256-197348df/resume_validation.json), [3B probe](../checkpoints/pilot/qwen-3b-probe/experiment.json), [7B probe](../checkpoints/pilot/qwen-7b-probe/experiment.json), [loopback runtime replay](../results/runtime_replay/20261019T215124-1fa2e208/replay.json), [machine status](../manifests/implementation_status.json).

Every execution creates results/<stage>/<run-id>/ with configuration/source snapshots, events, stdout/stderr, failures, and metrics. The interrupted full sequence is results/development_validation_sequence/20261019T215948-1f760da2/. Completed interface checks: results/model_interface_sequence/20261019T221636-836b7753/. Completed smoke sequence: results/agent_smoke_sequence/20261019T221705-5a995848/. Historical synthetic checks and failed setup attempts are preserved.

The four bounded diagnostics completed through official grading. Their individual episode errors, patch sizes, token counts, timing, and GPU peaks are in the [agent smoke report](agent_smoke_report.md). These results do not complete the frozen-five-task acceptance checks.

All four smoke episodes reached the five-step limit. The 7B model exported a 1,508-byte patch; the other three conditions exported empty patches. None resolved the task. The refreshed native inventory contains 111,781 entries, about 38.40 GB of logical files, with zero scan errors.

Cleanup removed 31,954 disposable files (590 MB) and the obsolete runtime image/cache chain (about 1.89 GB inside Docker). Active environments and all research artifacts/logs were retained. The latest volume check shows 45.03 GB free and 12.15 GB cap headroom, so full development execution remains gated. [Cleanup report](cleanup_report.md).
