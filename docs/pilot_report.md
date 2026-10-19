# Real-model pilot validation

Updated 2026-10-19. These results use downloaded Qwen2.5-Coder models and audited SWE-Gym histories on the local RTX 4090. They are setup/feasibility checks, rather than the full nine-adapter study.

| Run | Context limit | Training / validation examples | Steps | Peak allocated / reserved VRAM | Training/eval/checkpoint wall time |
| --- | ---: | ---: | ---: | --- | ---: |
| 1.5B overfit, full logits | 4096 | 4 / 2 | 16 | 9.34 / 11.24 GiB | 67.9 s |
| 1.5B pilot, target-only logits | 8192 | 32 / 9 | 25 | 6.85 / 16.63 GiB | 279.9 s |
| 1.5B resume check | 8192 | 32 / 9 | +2, from 25 to 27 | See immutable resume metrics | 32.7 s |
| 3B memory probe, target-only logits | 8192 | 8 / 2 | 2 | 8.57 / 13.59 GiB | 19.5 s |
| 7B memory probe, target-only logits | 8192 | 8 / 2 | 2 | 14.81 / 18.61 GiB | 26.3 s |

Times exclude model initialization and fresh-process reload. The 1.5B pilot processed actual tokens at about 4488 tokens/s after the first two warmup microbatches. The 3B and 7B probes each used only two microbatches, so it does not provide a steady-state throughput estimate. Each probe’s two training histories totaled 15,698 tokens. Memory feasibility applies to the measured examples and configuration.

The overfit loss fell from about 0.673 to 0.00176, while validation loss later rose. The larger 1.5B pilot's best validation loss was 0.5175. These observations validate the optimizer path; they do not establish benchmark improvement or full-corpus generalization.

## Checkpoint and loss correctness

Nonzero adapter gradients and updated adapter tensors were verified. The fresh-process reload comparison matches exactly (maximum logit difference 0.0) after aligning unquantized layer precision and BF16 autocast. Earlier failed comparisons are retained; the tolerance was not widened.

Resume validation preserved the complete step-25 checkpoint, loaded optimizer/scheduler/RNG/Trainer state, and extended the schedule to 27 for this state check. All 392 optimizer entries advanced by two steps, and 392 adapter tensors changed. The resumed adapter passed fresh reload. This schedule extension is not an exact uninterrupted-schedule equivalence experiment.

Target-only logits compute predictions only at positions whose next-token labels are unmasked assistant targets. A regression test compares full causal loss and all parameter gradients on a small Qwen2 model. User/system/observation/padding labels remain masked. The initial full-logit 8K attempt was stopped under VRAM pressure before an optimizer step and has no valid adapter result.

## Data and integration limits

Only 46 of 489 complete histories fit 8K: 37 train and 9 validation. The user selected complete 8K histories for the bounded pilot if memory permits. The 32 training histories are a short-history subset; final training still needs a deliberate common context/windowing policy. Larger-model probes use long complete histories near the 8K limit and are resource checks.

The source-matched action converter replayed shell execution, editing, and finish. A real OpenHands container passed scripted edit/test/patch export with no host mounts or privileged mode. The first repaired grader image passed gold/blank controls. Real model-to-agent rollouts and adapter-to-grader development validation remain pending.

Evidence: [overfit](../results/train/20261019T181357-425bca26/metrics.json), [25-step pilot](../results/train/20261019T212429-33c80784/metrics.json), [resume state check](../results/resume_validation/20261019T213256-197348df/resume_validation.json), [resumed reload](../results/validate-checkpoint/20261019T213541-77186790/reload_validation.json), [3B probe](../checkpoints/pilot/qwen-3b-probe/experiment.json), [3B reload](../results/validate-checkpoint/20261019T214250-395117df/reload_validation.json), [7B probe](../checkpoints/pilot/qwen-7b-probe/experiment.json), [7B reload](../results/validate-checkpoint/20261019T215338-e30763a2/reload_validation.json), [runtime replay](../results/runtime_replay/20261019T212637-a6467397/replay.json), [historical synthetic report](../results/setup/resume-2026-10-19T17-52-39-960Z/pilot_report.before_real_validation.md).
