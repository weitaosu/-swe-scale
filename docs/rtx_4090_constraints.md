# RTX 4090 constraints: measured update

Updated 2026-10-19. The validated setup runs Qwen2.5-Coder 1.5B, 3B, and 7B NF4 QLoRA at an 8192-token context limit on the local 24 GiB RTX 4090. Larger GPUs are not a prerequisite for these measured configurations.

| Model / test | Peak allocated | Peak reserved | Result |
| --- | ---: | ---: | --- |
| 1.5B, 25-step pilot | 6.85 GiB | 16.63 GiB | Passed, including save/resume/reload |
| 3B, two-step long-history probe | 8.57 GiB | 13.59 GiB | Passed, including fresh reload |
| 7B, two-step long-history probe | 14.81 GiB | 18.61 GiB | Passed, including fresh reload |

These measurements depend on NF4 frozen base weights, BF16 computation, LoRA rank 16, microbatch 1, gradient checkpointing, and computing output logits only for unmasked assistant targets. The loss/gradient equivalence regression passed. Unquantized layers and autocast are kept consistent between training and inference. The original full-logit 8K implementation hit VRAM pressure and slowed severely; that implementation limit was addressed before the successful pilot.

## Constraints that remain

- **Training data coverage:** only 46/489 complete histories fit 8K (9.4%). The approved pilot uses 32 of the 37 retained training histories and nine validation histories. The full study needs an explicit common context/windowing policy and retention audit.
- **Longer contexts:** 16K/32K training has not been measured. The 7B 8K probe already reserves 18.61 GiB; longer-context feasibility must be tested with the chosen implementation and available VRAM.
- **Full dense 7B training:** standard all-GPU BF16 weights plus gradients alone exceed 24 GiB, before optimizer state and activations. This setup validates parameter-efficient training.
- **Throughput:** the 1.5B pilot measured about 4488 processed tokens/s after warmup. Two-step 3B/7B probes establish memory feasibility, with insufficient samples for steady-state or full-study time estimates.
- **Resource scheduling:** this project trains or serves one model condition at a time. Concurrent GPU usage varies; existing unrelated workloads are not stopped by project scripts.
- **Storage:** original model snapshots remain about 24.5 GB locally despite 4-bit GPU loading. Environments, checkpoints, logs, and Docker add space. The approved 70 GB growth cap and 25 GB free-space floor apply to the shared volume and Docker VHDX.
- **Integration:** scripted runtime replay and one-task grader controls passed. Real model-driven development episodes are still being validated; benchmark quality is not established by memory tests.

[Measured pilot report](pilot_report.md), [data audit](data_audit.md), [current progress](implementation_status.md), [previous planning constraints](../results/setup/resume-2026-10-19T17-52-39-960Z/rtx_4090_constraints.before_measurements.md).

The length-only context audit finds that 16K retains 259/489 histories (53.0%), 32K retains 443/489 (90.6%), and 64K retains 484/489 (99.0%). These are corpus coverage figures, not evidence that those settings fit the 4090. [Length evidence](../results/context_coverage_analysis/20261019T222313-80fa1995/coverage.json).
