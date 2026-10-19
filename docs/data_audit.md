# SWE-Gym real-data audit

Updated 2026-10-19. Five public dataset payloads are downloaded locally at immutable revisions and verified by file size and LFS SHA-256. See [dataset manifest](../manifests/dataset_files.json) and [download log](../results/download/20261019T180246-8946f115/run.json).

The successful SFT release contains 491 trajectories. Task/tool provenance matching retained 489; two ambiguous records were quarantined. The deterministic task-level split has 439 training trajectories from 263 tasks and 50 validation trajectories from 29 tasks. Training and validation task IDs are disjoint. Screening uses all Verified IDs and issue/patch fingerprints plus the five frozen development IDs. See [audit results](../results/prepare/20261019T180453-a1051bad/data_audit.json).

SFT messages use text/XML actions and user-role tool observations, with tool descriptions already embedded in the system prompt. Raw sampled records supply task identities and native tool schemas; these are matched at task level, without claiming an exact teacher run. All source records, exclusions, and split provenance are preserved under data/metadata.

The real Qwen tokenizer retains only 5 training and 3 validation histories at 4K (8/489 = 1.64%). At 8K it retains 37 training and 9 validation histories (46/489 = 9.41%). Overlength histories are rejected explicitly; no silent truncation is used. Both policies select short histories and are unsuitable as an unqualified proxy for the full corpus. A 4K four-example overfit test is running; an 8K bounded pilot is possible if memory permits. Decoded assistant-label samples and complete exclusion reasons are stored in the tokenizer runs.

Coverage evidence: [4K](../results/tokenize/20261019T181216-4fcfbca5/token_stats.json), [8K](../results/tokenize/20261019T181511-7bdbaf6d/token_stats.json). Production training still needs a deliberate common context/windowing policy and retention analysis. Public overlap screening cannot establish absence of benchmark exposure in pretrained weights.

## Full-history context coverage

Reconstructed from the current normalized 8K tokenization: all retained input lengths plus exact over-context exclusion lengths, with source hashes checked. This is a length analysis; it changes no training policy and performs no GPU probe.

| Context | Train histories | Validation histories | Total / 489 | Coverage |
| --- | ---: | ---: | ---: | ---: |
| 4,096 | 5 | 3 | 8 | 1.6% |
| 8,192 | 37 | 9 | 46 | 9.4% |
| 16,384 | 229 | 30 | 259 | 53.0% |
| 32,768 | 400 | 43 | 443 | 90.6% |
| 65,536 | 435 | 49 | 484 | 99.0% |

Median full-history length is 15,989 tokens; the 95th percentile is 38,967. Only 8K has the all-size real GPU pilot/probe evidence. Memory feasibility and throughput above 8K remain unmeasured. The approved pilot still uses complete histories at 8K. [Exact analysis](../results/context_coverage_analysis/20261019T222313-80fa1995/coverage.json).
