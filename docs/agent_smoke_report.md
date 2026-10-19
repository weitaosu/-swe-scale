# Bounded agent integration diagnostic

Updated 2026-10-19. These runs exercise the real local model, OpenHands, patch export, and official grader on the first frozen development task only. Limits are five steps and 300 seconds per episode. They do not replace the five-task comparison or establish research performance.

| Condition | Agent calls / output tokens | Peak allocated / reserved GiB | Patch bytes | Episode error | Resolved |
| --- | ---: | ---: | ---: | --- | --- |
| baseline-1.5b-smoke | 5 / 247 | 2.27 / 2.95 | 0 | Agent reached maximum iteration in headless mode. Current iteration: 5.00, max iteration: 5.00 | False |
| swegym-pilot-1.5b-smoke | 5 / 169 | 2.37 / 3.06 | 0 | Agent reached maximum iteration in headless mode. Current iteration: 5.00, max iteration: 5.00 | False |
| baseline-3b-smoke | 5 / 186 | 3.43 / 4.20 | 0 | Agent reached maximum iteration in headless mode. Current iteration: 5.00, max iteration: 5.00 | False |
| baseline-7b-smoke | 5 / 988 | 9.01 / 9.78 | 1508 | Agent reached maximum iteration in headless mode. Current iteration: 5.00, max iteration: 5.00 | False |

Each condition uses a fresh server/ledger. The one-token API preflight is excluded from the agent token totals but remains charged to the task budget. Memory peaks cover HTTP completions, not model-loading transients. Agent errors and empty patches remain failures of the attempted episode even when orchestration and report generation complete.

[Detailed metrics](../results/smoke_reporting/20261019T224904-9ce37df0/metrics.json), [sequence](../results/agent_smoke_sequence/20261019T221705-5a995848/smoke_sequence.json), [remaining environment gates](development_environment_constraints.md).
