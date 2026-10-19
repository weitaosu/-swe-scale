# RTX 4090 implementation plan for the first two milestones

## Approval status and scope

Approved by the user on 2026-10-19 and implementation started. The estimates and milestone acceptance criteria below remain the agreed scope. See [implementation_status.md](implementation_status.md), [pilot_report.md](pilot_report.md), and [../README.md](../README.md) for actual artifacts, measured checks, blockers, and continuation commands. Real asset downloads and Docker execution remain pending; offline source, a local validation venv, and synthetic GPU evidence have been created.

The target is one desktop RTX 4090 with 24 GB VRAM. We will implement the Sep 14-27 setup milestone and the Sep 28-Oct 11 SWE-Gym preparation and LoRA validation milestone. The date ranges identify the proposal milestones; execution will follow the dependency order below.

The requested project root is C:/Users/weita/Desktop/SE in AI/proj. All project source files, model weights, raw and processed datasets, adapters, logs, and manifests will be placed under this folder. The inventory is [planned_downloads_4090.json](../manifests/planned_downloads_4090.json). Its revisions are intentionally unresolved until the download preflight pins them.

## Study design retained from the original draft

The full study compares Qwen2.5-Coder Instruct 1.5B, 3B, and 7B trained on SWE-Gym, SWE-smith, and SWE-Master-derived data: nine adapters plus three matching unmodified baselines. All twelve conditions should use the same scaffold, tools, inference policy, and approximately 100 frozen Verified tasks. Record resolution, matching-baseline gains, training tokens/GPU hours/memory, and inference tokens/steps/tool calls/runtime/timeouts. These first two milestones cover development checks and SWE-Gym preparation; production training and full evaluation remain later work.

Before later cross-source training, verify downloadable SWE-Master artifacts and map representative SWE-agent/SWE-smith and R2E-Gym/SWE-Master actions to the shared interface. Renaming message roles alone does not preserve tool semantics. Unsupported actions require explicit exclusions or new compatible trajectories. Keep teachers, curation, repositories, licensing, and tool-framework differences in the final interpretation: the comparison measures released-data conditions rather than isolating a causal task-source effect.

Freeze common training/inference settings only after size-specific profiling. Equal token budgets need not imply equal GPU hours. Keep paired per-task outcomes, a fixed evaluation denominator and infrastructure-failure policy, and report uncertainty for the small evaluation subset. Do not tune on the final 100-task scores or interpret development pilots as scaling evidence. Hardware-driven policy changes must be versioned and applied consistently across compared conditions.

The original planning responsibility split assigns Weitao data/training/efficiency work and Qingyao scaffold/task selection/grading/inference work; both review shared schemas, context policy, versions, and acceptance evidence. The earlier draft is preserved in the cleanup audit archive; this document is the current 4090 plan.

## Disk estimate before implementation

GB denotes 1,000,000,000 bytes and MB denotes 1,000,000 bytes. Published repository sizes below are rounded, not exact transfer quotas. Download and extraction overhead are budgeted separately.

### Model files

| Model | Estimated local snapshot size | Destination |
| --- | ---: | --- |
| [Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct/tree/main) | 3.10 GB | models/Qwen2.5-Coder-1.5B-Instruct/ |
| [Qwen2.5-Coder-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct/tree/main) | 6.18 GB | models/Qwen2.5-Coder-3B-Instruct/ |
| [Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct/tree/main) | 15.20 GB | models/Qwen2.5-Coder-7B-Instruct/ |
| Total | 24.48 GB | One canonical snapshot per model |

These are the original model snapshots, including weights and small tokenizer/configuration files. Loading the frozen model in 4-bit form for QLoRA reduces GPU memory; it does not reduce these original downloads. We will avoid saving duplicate full or merged model copies during the pilot.

### Raw dataset files

| Dataset | Records or use | Published size or budget | Destination |
| --- | --- | ---: | --- |
| [SWE-Gym/OpenHands-SFT-Trajectories](https://huggingface.co/datasets/SWE-Gym/OpenHands-SFT-Trajectories) | 491 successful SFT trajectories | 10.4 MB | data/raw/swegym/sft/ |
| [SWE-Gym/OpenHands-Sampled-Trajectories](https://huggingface.co/datasets/SWE-Gym/OpenHands-Sampled-Trajectories) | 6,055 sampled trajectories for provenance and tool schemas | 301 MB | data/raw/swegym/sampled/ |
| [SWE-Gym/SWE-Gym](https://huggingface.co/datasets/SWE-Gym/SWE-Gym) | 2,438 task metadata records | 43.6 MB | data/raw/swegym/tasks/ |
| [SWE-bench Verified](https://huggingface.co/datasets/SWE-bench/SWE-bench_Verified) | 500 evaluation metadata records | 6.31 MB | data/raw/swebench/verified/ |
| [SWE-bench development metadata](https://huggingface.co/datasets/SWE-bench/SWE-bench) | Only the dev split is needed; budget uses the full repository size | Up to 36.6 MB | data/raw/swebench/dev/ |
| Total raw-data allowance | Excludes Docker environments and processed copies | Approximately 0.40 GB | data/raw/ |

The 491-trajectory SFT release is the training starting point. The sampled pool supplies task IDs, tools, and outcomes for provenance; it will not be treated as an unfiltered training dataset. The task metadata is distinct from both the training trajectories and the executable Docker images.

### Working storage

| Component | Estimated additional disk use | Basis |
| --- | ---: | --- |
| Model snapshots | 24.48 GB | Published repository listings |
| Raw datasets | 0.40 GB | Published listings and conservative dev allowance |
| Source checkouts and project code | 1-3 GB | Planning allowance |
| Linux environments and dependencies | 15-30 GB | Planning allowance for pinned training, serving, agent, and evaluation dependencies; share compatible packages where practical |
| Processed data and token caches | 1-3 GB | Planning allowance; avoid unbounded duplication of conversation prefixes |
| Pilot adapters and resume checkpoints | 2-6 GB | Adapter-only artifacts with limited retained checkpoints |
| Logs, reports, and trajectories | 1-5 GB | Small development runs and training pilots |
| Selected Docker images and writable layers | 30-100 GB | Unverified planning allowance for a few task/runtime images; actual use may be higher |
| Temporary downloads and builds | 10-20 GB peak | Partial downloads, wheel caches, and extraction/build overhead |
| Combined working peak | Approximately 85-192 GB | Estimate, not a guaranteed bound |

For comfortable completion of both milestones, aim for approximately 200-250 GB free on the relevant storage volumes before starting. This is our combined project allowance. The [official SWE-bench setup](https://github.com/SWE-bench/SWE-bench) independently recommends at least 120 GB free for evaluation. The [OpenHands evaluation guide](https://github.com/SWE-Gym/OpenHands/blob/main/evaluation/swe_bench/README.md) describes larger storage requirements for broader instance-image sets; we will pull only a small development set in these milestones.

The project drive had approximately 102.9 GB free, or 95.8 GiB, at the planning check. That is enough for an initial staged download and development setup, but it does not cover the upper combined estimate with comfortable headroom. The check does not establish the location or free capacity of Docker's managed storage.

## Local storage layout

The following was the planned layout. The implementation consolidates source modules in src/swe_study/ and uses scripts/study.py for the individual stages. README.md lists actual paths and commands.

    proj/
      CS6158 Project Proposal.pdf
      README.md
      .gitignore
      .env.example
      .envs/
      .cache/
        huggingface/
        datasets/
        pip/
        uv/
        tmp/
      configs/
        agent.yaml
        qlora_overfit.yaml
        qlora_pilot.yaml
        qlora_probe_3b.yaml
        qlora_probe_7b.yaml
      manifests/
        planned_downloads_4090.json
        versions.json
        model_files.json
        dataset_files.json
        dev_tasks.json
        eval_100.json
        storage_usage.json
      vendor/
        SWE-bench/
        SWE-Gym-OpenHands/
        swe-bench-tasks/
      models/
        Qwen2.5-Coder-1.5B-Instruct/
        Qwen2.5-Coder-3B-Instruct/
        Qwen2.5-Coder-7B-Instruct/
      data/
        raw/swegym/{sft,sampled,tasks}/
        raw/swebench/{verified,dev}/
        processed/swegym/{train.jsonl,validation.jsonl}
        metadata/
        tokenized/
      src/swe_study/
        assets.py
        audit.py
        data.py
        tokenization.py
        training.py
        serving.py
        evaluation.py
      scripts/
      checkpoints/pilot/
      results/{preflight,baseline_smoke,data_audit,training_pilot}/
      docs/
        rtx_4090_implementation_plan.md
        environment.md
        data_audit.md
        pilot_report.md

Use the same physical root from WSL2 through /mnt/c/Users/weita/Desktop/SE in AI/proj, with correctly quoted paths. Respect the requested folder rather than maintaining a second dataset or model tree in the Linux home directory. NTFS access may reduce throughput relative to a Linux filesystem; measurements will reflect this layout.

Set Hugging Face, Datasets, package-manager, and temporary cache locations beneath .cache for commands we control. Download snapshots directly to their model directories and load them by local path. Record file sizes, checksums, and immutable revisions. Use local raw Parquet files for processing. [Hugging Face download behavior](https://huggingface.co/docs/huggingface_hub/guides/download).

Docker images and its internal filesystem are managed by the Docker daemon, potentially outside this project folder. Record that physical storage location and include its growth in the budget. Do not move the existing Docker installation or change global Docker/WSL storage settings as part of routine project setup. We will not export duplicate image archives into the project.

## Milestone one Sep 14 to 27

### Step 1 Verify the machine and establish a disk budget

Implement scripts/preflight.ps1 and scripts/preflight.sh. Record the actual GPU, usable VRAM, NVIDIA driver, CUDA availability inside WSL2, CPU/RAM, Docker connectivity, and free space for both the project and Docker storage. Run a small CUDA operation and a container check before installing heavy dependencies or downloading models.

The planning session could not successfully enumerate WSL, so Linux and Docker readiness are not yet verified. The preflight must establish those capabilities and report any host setup requirement explicitly.

For the staged approval option, permit at most 70 GB of incremental project-related disk growth and preserve at least 25 GB free on each affected volume. Include Docker growth and temporary files in the accounting. Recheck before large transfers; stop additional transfers when the limits would be exceeded. This can leave Docker validation pending until more space is available. Do not delete unrelated files or run broad Docker pruning.

Outputs: results/preflight/hardware.json, manifests/storage_usage.json, and docs/environment.md.

### Step 2 Create isolated and reproducible environments

Create a project package and separate Linux environments for the older OpenHands fork and official evaluator. Use a compatible training/serving environment based on PyTorch, Transformers, Datasets, PEFT, and bitsandbytes, and a small local HTTP serving wrapper. Prefer one compatible training/serving stack over installing an additional large serving engine for the pilot.

Pin dependency versions after a successful import, GPU quantization, and generation check. Record Python/CUDA/library versions, repository commits, and environment commands in manifests/versions.json. Put large local artifacts and environments in .gitignore.

### Step 3 Download the specified assets locally

Use src/swe_study/assets.py through scripts/study.py download with inventory, size preflight, revision pinning, resumable transfer, checksum reporting, and local-path loading checks. Download 1.5B first, then 3B and 7B after confirming the preceding setup works. Download the listed raw datasets and their source metadata into data/raw.

Outputs: the three model directories, raw dataset directories, manifests/model_files.json, and manifests/dataset_files.json. No SWE-smith/SWE-Master bulk data, 32B weights, or full benchmark image collection is included.

### Step 4 Establish the benchmark protocol and grader

Create a deterministic development list of five tasks disjoint from Verified, using development metadata or held-out SWE-Gym tasks. Exclude any held-out SWE-Gym task from training. Expand to ten only if the initial pipeline works and storage permits.

Select and freeze 100 Verified task IDs with a recorded seed and repository coverage rule. Retain all 500 Verified IDs as a contamination denylist. This prepares the later evaluation; it does not launch the full 100-task runs now.

Install the official SWE-bench evaluator at a pinned revision. Pull/build only the first necessary reference/development image, measure disk growth, and then decide whether further selected images fit the approved budget. Use a supplied gold patch and a blank patch to validate expected positive and negative grading outcomes. Give each distinct prediction artifact a fresh run ID.

Outputs: manifests/dev_tasks.json, manifests/eval_100.json, scripts/study.py grade, and reference/negative-control reports. Keep gold patches and hidden test information outside the agent-accessible workspace.

### Step 5 Connect the local models and common scaffold

Use the pinned SWE-Gym/OpenHands CodeActAgent fork to match the released training trajectories. Use src/swe_study/serving.py and src/swe_study/evaluation.py with scripts/openhands_entry.py to serve one local model, render its chat/tool template once, parse tool calls, inspect/edit files, run tests, observe feedback, and export a patch. Check the downloaded trajectory action protocol against the configured tools before freezing the interface.

Start with one active task, an 8K total context limit with output space reserved, temperature 0, a 30-step ceiling, and a 30-minute task timeout. These are pilot settings, not final production commitments. Use one consistent inference precision across matching baseline and adapter conditions. Prefer an NF4-compatible Transformers/PEFT path for the QLoRA pilot and verify that adapter loading actually works through the server.

Replay a known valid action sequence, then run 1.5B, 3B, and 7B baselines sequentially on the same development list. Log tokens, steps, tool calls, parsing errors, runtime, peak memory, timeouts, and final patches. Grade the predictions using the official harness.

Milestone one is complete only when all three models traverse the agent-to-grader pipeline and the reference controls behave as expected. A completed download alone is insufficient. If disk or host access prevents Docker execution, report this milestone as incomplete with the remaining checks identified.

## Milestone two Sep 28 to October 11

### Step 6 Prepare and audit the SWE Gym training data

Use src/swe_study/data.py and src/swe_study/tokenization.py. Recover task identities by validated matching between SFT histories, sampled records, and task metadata; quarantine ambiguous matches. Preserve source references, tools, assistant actions, observations, and success provenance.

Validate tool-call/result pairing and supported actions. Deduplicate tasks and normalized trajectories. Check task IDs and available issue/patch fingerprints against the full Verified denylist and development tasks. Split approximately 90/10 by task before generating any derived examples. Preserve intermediate mistakes within ultimately successful histories.

Save processed train/validation JSONL and an audit manifest. Measure raw/retained task counts, total tokens, assistant target tokens, trajectory lengths, and exclusion reasons. Freeze a deterministic context policy that preserves complete tool interactions. Do not silently trim successful endings to fit a memory limit.

Outputs: data/processed/swegym/train.jsonl, validation.jsonl, data/metadata/provenance.jsonl, exclusion records, results/data_audit/, and docs/data_audit.md.

### Step 7 Implement and verify QLoRA SFT

Use src/swe_study/training.py and an explicit, verified assistant-only label mask. System/user text, tool observations, and padding are conditioning context rather than training targets. Check several decoded labeled histories, tool actions, end markers, and nonzero target-token counts. The exact Qwen2.5 template must support the chosen masking method. [TRL SFT requirements](https://huggingface.co/docs/trl/sft_trainer).

Start with 4-bit NF4 frozen weights, BF16 computation where supported, rank 16, alpha 32, dropout 0.05, learning rate 1e-4, microbatch 1, gradient accumulation 8, gradient checkpointing, and packing disabled. Target the attention and MLP projection layers after verifying their module names. These are pilot choices. [PEFT quantization guide](https://huggingface.co/docs/peft/developer_guides/quantization).

Run a finite-loss forward/backward check and a tiny 4-8-example overfit check on 1.5B. Verify that adapter weights change and the base remains frozen. Then use 32-64 task-disjoint training trajectories for a bounded pilot of at most 25 optimizer steps initially. Save the best pilot adapter plus at most two resume checkpoints; report actual processed tokens and repetitions.

Outputs: configs/qlora_pilot.yaml, checkpoint artifacts, loss curves, validation loss, measured token throughput, peak memory, and GPU-hour accounting. Do not require a short pilot to improve benchmark resolution.

### Step 8 Validate checkpoint reuse and larger models

Save optimizer/scheduler and RNG state, briefly resume, then reload the base plus adapter in a fresh process. Check fixed-prompt behavior before and after reload within numerical tolerance. Run the adapter through the same local server, development tasks, patch exporter, and official grader as its baseline.

Perform a few 3B and 7B training steps at 4K/8K to establish memory feasibility. Probe 16K and 32K only when measured headroom and the selected implementation justify it. Record unsupported settings without treating them as completed configurations. Select a common feasible production context after the data-retention audit; larger hardware remains an option if a shorter context is unsuitable.

Training and serving run sequentially on the 4090. Capture size-specific throughput rather than projecting all runtimes from 1.5B. Outputs: results/train/, scripts/study.py validate-checkpoint, and docs/pilot_report.md.

Milestone two is complete when provenance, task-level splitting, label masking, adapter updates, save/resume/reload, and the adapter-to-grader workflow have all been verified. Full nine-model training and twelve-condition evaluation remain outside this approval.

## Current implementation files

| Files | Purpose |
| --- | --- |
| README.md, pyproject.toml, .gitignore, .env.example | Commands, dependencies, and local path policy |
| scripts/study.py, bootstrap.py, continue_setup.py, preflight.ps1, preflight.sh | Audited stage entry points and environment setup |
| configs/agent.yaml, qlora_overfit.yaml, qlora_pilot.yaml, qlora_probe_3b.yaml, qlora_probe_7b.yaml | Active scaffold and training settings |
| src/swe_study/assets.py, audit.py, common.py, storage.py, docker_storage.py, inventory.py | Downloads, run logs, storage accounting, and inventory |
| src/swe_study/data.py, tokenization.py | Data provenance, overlap checks, task selection, and label masks |
| src/swe_study/training.py, serving.py, evaluation.py, preflight.py, cli.py | Adapter training, model API, official grading, and orchestration |
| scripts/openhands_entry.py, swebench_entry.py | Pinned source adapters for separate Linux environments |
| scripts/tiny_gpu_fixture.py, validate_local_server.py, tests/test_pipeline.py | Offline fixtures and regression/integration checks |
| manifests/*.json and data/metadata/*.jsonl | Versions, file hashes, task lists, provenance, and storage state |
| docs/implementation_status.md, environment.md, data_audit.md, pilot_report.md | Current progress and evidence |

Model targets and download policy are recorded in manifests/planned_downloads_4090.json; training model/precision choices are in the active QLoRA YAMLs, and development sampling/grader defaults are implemented in the corresponding source modules. The unused duplicate models.yaml and evaluation.yaml files were removed during file review.

## Approved staged execution and limits

**Staged implementation within the current drive capacity:** implement the files and download assets in order, subject to the 70 GB incremental limit and 25 GB free-space floor. Attempt selected Docker checks only when measured capacity permits. Complete independent data/training work if Docker storage is the limiting factor, and report any pending end-to-end validation explicitly.

**Full implementation after additional space is available:** target approximately 200-250 GB free, then carry out both milestone workflows including Docker grading, while still pulling only the selected development images and monitoring actual growth.

Approval covers the named local downloads, project environments, implementation, and bounded pilots. It does not include paid GPU services, global storage migration, unrelated-file deletion, the optional 32B experiment, or the nine-run production matrix.
