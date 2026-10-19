# SWE-Gym preparation status

The processing and token-labeling code is implemented and covered by synthetic tests. **The real SWE-Gym audit has not run:** dataset access is blocked. Retained counts, task overlap results, length distributions, and usable 4K coverage remain unknown. No synthetic histories are placed in the real processed-data directories.

The approved inventory is manifests/planned_downloads_4090.json. It specifies the 491-row successful SFT split, the 6,055-row sampled pool for provenance/tools, SWE-Gym task metadata, all 500 Verified task identities, and SWE-bench dev metadata. Remote revisions and file checksums are filled only after a successful download; current failure evidence is in manifests/dataset_files.json.

The preparation code:

1. Reads local Parquet in batches and preserves relative file paths and row indices.
2. Matches SFT histories to successful sampled histories or a unique task issue in the first user request; quarantines ambiguous or untraceable matches.
3. Recovers a unique tool schema, validates JSON arguments and call/result pairing, and preserves intermediate mistakes in successful histories.
4. Excludes all Verified IDs and frozen development IDs, plus available normalized issue/patch fingerprint matches.
5. Deduplicates canonical trajectories and partitions about 90/10 by task using seed 6158, so one task cannot cross training and validation.
6. Renders the local model chat template, constructs explicit assistant-only spans with token offsets, includes assistant end markers, and masks system/user/tool observations and padding with -100.
7. Rejects over-context histories, unstable template boundaries, empty targets, and empty retained splits. It records each rejection instead of truncating the successful ending.

After downloads, run the data continuation stage and inspect results/data_audit/, results/tokenize/, data/metadata/, and data/tokenized/. The exact Qwen template and raw tool protocol must be reviewed on real examples. Tool schemas already embedded in the system prompt are detected to avoid duplicate injection; this detection still needs real-corpus review.

Task matching establishes task-level provenance, not necessarily an exact teacher rollout when multiple raw successes share an issue. The published successful SFT split supplies the success claim. Public identity/fingerprint screening cannot prove absence of pretrained benchmark exposure. Record these limits in later experimental reporting.

If too few complete histories fit 4K, stop and measure the rejected lengths. A revised policy can use longer contexts after memory measurement or explicitly constructed complete interaction windows with preserved provenance and target accounting. Such a change must be documented before freezing the pilot; silent truncation is not an acceptable substitute.
