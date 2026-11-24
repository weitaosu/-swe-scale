# NERSC results and challenges resolved

The three Qwen2.5-Coder QLoRA runs completed on NERSC, and matched validation
shows lower assistant-response loss for every adapter. Getting to that result
required fixes across model memory, distributed training, checkpoint recovery,
container compatibility, historical dependencies, agent context handling and
result accounting. Agent issue-resolution evaluation is still in progress.

The current quantitative status is the **2026-11-24 03:16:13 UTC** export,
committed as `2b5d05b`. The associated
[audit](../results/nersc_results_review/20261124T032614-36127ffd/review.json)
checks 38 imported JSON files and records 63 passing consistency checks.
The [raw export](../results/nersc_uploaded/20261124T031613Z/) is preserved.

This record distinguishes imported results, local regression checks and remote
operator-console observations. Remote run paths below identify records on
Perlmutter; they do not imply that every log, image or model binary was copied
into Git. Compatibility fixtures are identified separately from real model and
task runs. Open items remain open even when an earlier stage passed.

## Results and interpretation

### Training completion and matched validation

| Model | Training GPUs | Epochs | Optimizer steps | Effective batch | Base loss | Adapter loss | Relative reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.5B | 1 | 3 | 150 | 8 | 0.683363915 | 0.446509570 | 34.6601% |
| 3B | 1 | 3 | 150 | 8 | 0.585692883 | 0.408809662 | 30.2007% |
| 7B | 2 | 3 | 150 | 8 | 0.547683299 | 0.372860312 | 31.9205% |

All runs used frozen NF4 base weights, BF16 computation, LoRA rank 16 and alpha
32, dropout 0.05, learning rate 0.0001, seed 6158 and microbatch size one.
Gradient accumulation was eight on one GPU and four per rank on two GPUs.
The same seven attention/MLP projection types were adapted. Each final adapter
is associated with checkpoint 150 and recorded nonzero gradients and parameter
changes. The per-step histories contain steps 1 through 150.

| Model | Epoch 1 validation | Epoch 2 validation | Epoch 3 validation | Relative improvement from epoch 2 to 3 |
| --- | ---: | ---: | ---: | ---: |
| 1.5B | 0.476138264 | 0.452354878 | 0.446509570 | 1.29% |
| 3B | 0.432747453 | 0.413946241 | 0.408809662 | 1.24% |
| 7B | 0.405668527 | 0.388594598 | 0.384927124 | 0.94% |

The 7B epoch table uses its historical two-GPU training aggregation. The matched
comparison table uses the canonical single-GPU history mean for every size.
Comparing those different 7B reductions caused an earlier validation rejection;
the reconstruction described below resolves that difference.

Loss decreases over all three evaluated epochs, with smaller final-epoch gains.
Further training is therefore not the immediate priority. This validation set
selected the best checkpoints and is not an untouched final test. Better
response prediction does not establish better repository repair.

The successful comparison reports are
[1.5B](../results/nersc_uploaded/20261124T031613Z/cloud_validation/20261123T232136-88a293aa/comparison.json),
[3B](../results/nersc_uploaded/20261124T031613Z/cloud_validation/20261123T231347-efe31c72/comparison.json)
and [7B](../results/nersc_uploaded/20261124T031613Z/cloud_validation/20261123T234356-8986d78a/comparison.json).
Training configurations and saved step histories are in the three
[imported experiment records](../results/nersc_uploaded/20261124T031613Z/training/).

### Data coverage and independence

The original preparation retained 489 of 491 successful trajectories after
quarantining two ambiguous records. The task-disjoint raw split contains 439
training histories from 263 issues and 50 validation histories from 29 issues.
The NERSC runs retained 400 training and 43 validation histories under the 32K
complete-history policy. The imported validation exclusions explicitly identify
seven histories over the context limit.

The retained validation set has 43 unique trajectory IDs but only **25 distinct
issue IDs**. Its history counts by repository are moto 16, pandas 11, MONAI 9,
Hydra 3, DVC 2, mypy 1 and dask 1. One MONAI issue contributes seven histories.
The mean weights issues with multiple trajectories more heavily; it must not be
described as 43 independent issue-resolution tests.

All sizes used validation selection hash
`a0e4b56a761399750ffea62da861cb7d7194f4ba5c34bce1d79a90f164b9a81d`.
Processed training and validation hashes also match the local files. The audit
checked that their issue IDs do not overlap each other or the frozen five
development and 100 final benchmark IDs. Matching sources, selection, seed and
maximum length make the held-out loss comparisons consistent.

### Agent evaluation at the export time

Five development tasks with base and LoRA conditions require ten episodes per
model, or 30 across the three sizes. The controller's maximum of 30 steps is a
different quantity: steps inside one episode. The frozen final suite has 100
tasks, requiring 200 episodes per model or 600 across all three sizes if executed.

The imported study is `results/benchmark_15b_window_v1`:

| Task | Preparation | Base episode | LoRA episode |
| --- | --- | --- | --- |
| pvlib 807 | Passed gold and runtime checks | Graded, unresolved; loop detected; empty patch | Graded, unresolved; 30-step limit; placeholder patch |
| pydicom 1048 | Passed gold and runtime checks after pytest repair | No saved prediction | No saved prediction |
| SQLFluff 2907 | No prepared record in this export | Not run in this export | Not run in this export |
| astroid 1741 | No prepared record in this export | Not run in this export | Not run in this export |
| pvlib 1048 | No prepared record in this NERSC study export | Not run in this export | Not run in this export |

The adapter's patch added a root-level `scaling.py` with an explicitly labeled
placeholder and `irradiance * level`. It was not a complete Wavelet Variability
Model implementation. Producing a nonempty patch did not resolve the issue.

Only two of ten episodes were graded, with one observed task failure in each
condition. The saved `complete: false` and `valid_comparison: false` are correct.
Zero percent over the frozen denominator is a partial-run field, not a completed
five-task score. No 3B or 7B agent outcomes were included.

The [summary](../results/nersc_uploaded/20261124T031613Z/benchmark_15b_window_v1/development/summary.json)
matches the task records. Paired ledgers contain 375 base tokens and 2,567 adapter
tokens for pvlib 807, and zero for the remaining four tasks. Keep these ledgers
when moving the server to another GPU allocation.

## Memory and distributed training

### Long histories exhausted memory despite NF4 weights

32K training required much more memory than a short loading or two-step probe.
Observed failures requested another 6.16 GiB during cross-entropy, 8.33 GiB during
backward, and later 4.17 GiB after an initial optimization. An advertised 40 GB
A100 exposed about 39.49 GiB to the allocator. Successful early examples did not
guarantee that every later, longer history would fit.

NF4 reduced frozen weight storage but left long-sequence activations, vocabulary
projection and logits gradients. The implementation first avoided logits for
masked prompt positions, then chunked the loss, and finally moved projection
itself inside checkpointed chunks. Chunking cross-entropy after constructing
full logits still leaves a large logits gradient, so the final change mattered.

The final labeled Qwen path projects and computes exact token-summed causal
cross-entropy in 128-token chunks, recomputing small projections during backward.
It returns loss without a full labeled-logit tensor. MLP chunking at 1,024 tokens
for 7B bounds another activation allocation. Assistant targets, causal alignment,
weights and checkpoint keys retain the original objective/interface; generation
uses the ordinary unlabeled forward path.

Mechanics evidence includes
[head validation](../results/head_memory_validation/20261123T204742-189893f2/validation.json),
[MLP validation](../results/mlp_memory_validation/20261123T202652-af9500a6/validation.json)
and [loss regression](../results/memory_loss_regression/20261123T224236-29428373/validation.json).
These fixtures check loss/gradient equivalence and memory behavior. Completion
of the actual retained corpus is established separately by the real NERSC runs.

### Two GPUs did not create one combined memory pool

The distributed setup uses one process and one NF4 replica per GPU on one node.
Two 40 GB GPUs do not give one rank an 80 GB allocation. Each rank must fit its
own weights, activations and microbatch. A single 80 GB GPU has a different
capacity tradeoff from two replicated 40 GB GPUs.

The launcher binds a local device per rank, keeps microbatch one, and adjusts
accumulation to preserve effective batch eight. The orchestrator launches once
and starts its workers; it is not independently launched by both Slurm and
torchrun. The final 7B record confirms two GPUs and four accumulation steps per
rank. Changing GPU count is not an exact resume under the same configuration.

### NCCL timeouts obscured the first failure

Several runs ended with an ALLREDUCE timeout after 600,000 ms and rank-zero
SIGABRT. In the diagnosed memory failures, the earlier rank-one traceback was
CUDA out of memory. Rank zero waited for a collective its partner could not
complete. The final watchdog stack alone did not establish a network fault.

Debugging switched to the first rank exception in the captured command stderr.
Worker error reporting now prints and flushes the original exception before
successful-path teardown. The repair addressed memory pressure rather than
merely increasing NCCL timeouts. Other NCCL failures can have other causes and
still need their first exception and rank logs inspected.

### Allocator tuning did not supply missing capacity

`expandable_segments:True` was enabled to reduce fragmentation from variable sequence lengths.
The environment setting changed from deprecated `PYTORCH_CUDA_ALLOC_CONF` to
`PYTORCH_ALLOC_CONF` after the installed PyTorch emitted that warning. An idle
`nvidia-smi` check confirmed failed workers had released memory before a retry.

These settings did not remove the original tensor requirements. Completion
required the target-only path, bounded head/loss computation, MLP chunking and
context selection. Allocated tensor bytes, allocator reservations and device
usage are different measurements. Experiment peaks cover the measured segment
and rank zero, not every rank and earlier failed allocation.

## Checkpoints and trustworthy measurements

### Interruptions required complete Trainer state

The four-hour allocation limit interrupted 7B training; earlier memory errors
also left partial progress. Adapter weights alone could initialize another run,
but could not reproduce optimizer, scheduler, RNG and step state. Resume
selection therefore requires the expected complete files and chooses the latest
complete checkpoint, not simply the newest numbered directory.

The 3B/7B configurations save every optimizer step with bounded retention. A
completed checkpoint whose optimization is already finished is also handled:
final export is distinguished from an optimization segment that incorrectly
made no parameter updates. A remote tiny-model two-GPU check interrupted at step
two and resumed to four. It was explicitly synthetic and made no 7B capacity claim.

Real exported metadata records 3B resumption from checkpoint 13 and final 7B
resumption from checkpoint 147. Both finish at 150 and select checkpoint 150.
Earlier failures remain part of the history rather than being erased to present
an apparently uninterrupted run.

### Reload precision had to match training

Workstation reload checks exposed different precision/autocast treatment for
unquantized layers. Reload and serving were aligned to NF4 frozen weights,
FP32 unquantized layers and BF16 autocast. Existing tolerances were retained;
they were not widened to force acceptance. Historical real reload checks then
matched reference logits exactly; see the [pilot report](pilot_report.md).

Paired validation loads one adapter model and disables LoRA for the base
condition, keeping precision and base weights aligned. Real comparison records
remain separate from tiny random-model resume or loss-equivalence fixtures.

### Distributed and single GPU validation used different reductions

The initial 7B comparison reported `0.372860312` against training's `0.384927124`,
and the old direct-match guard rejected it. Investigation reproduced Trainer
5.5.3's token normalization across ranks. With 43 histories on two ranks, padding
the final group affects the historical denominator even when duplicate gathered
loss entries are dropped.

The validator now saves per-history losses and target counts, computes the
canonical mean, and reconstructs the training aggregation. Reconstruction gives
`0.384927164`, about `4.0e-8` from the training record. The later real run passed.
The old `adapter_matches_recorded_loss: false` remains compatible with the new
`training_metric_reproduction.passed: true` field.

A [two-process CPU regression](../results/distributed_validation_regression/20261123T233953-65fa436d/validation.json)
tested the aggregation before the real GPU rerun. The earlier failed report
remains historical evidence and must not replace the newer successful status.

### Resumed summaries are not complete training totals

This is a **diagnosed reporting limitation; full-run aggregation remains open**.
Timers and token counters restart per process invocation. The 3B value of 1.643
GPU-hours covers steps 14 through 150; final 7B's 0.145 GPU-hours covers 148
through 150. Neither includes all earlier training, failed attempts, model
loading or idle allocation time.

The final 7B Trainer summary has `train_loss: 0.009253668`. The last three logged
step losses average about `0.462683`; multiplying by 3/150 produces that small
summary. Its full saved step-history mean is about `0.449873`. Thus the small
final summary does not indicate a nearly perfect training loss. The resumed 3B
summary has the same class of denominator problem.

Compare complete step histories or matched held-out losses for learning. Sum
identified segments and failures for training compute, and use Slurm accounting
for allocation charges. Chunking settings and rank/segment memory scope must
accompany resource comparisons; the final process metrics are not total costs.

## Container compatibility and historical tasks

### A responding daemon did not prove task containers would run

Training worked without Docker, but agent execution and official grading needed
containers. The NERSC backend is Podman-HPC through a project-owned Unix socket
and its Docker-compatible Python API. The graph store is node-local under
`/tmp/<uid>_swe_study_podman/graph`.

An explicit API probe tested container create/start, exec, archive transfer and
loopback HTTP access. That probe is a container-mechanics check, separate from
the GPU model API and benchmark scores. The accepted CPU-node console result was
`results/container_api_check/20261124T000500-2cfbf95e`, with Podman 5.8.2 and
Docker API 1.44. The model remained a separate GPU service.

### Bridge networking failed and Pasta was not a working fallback

The first attempt invoked Netavark bridge rules; legacy iptables rejected
`--to-destination`. Trying the rootless default exposed another failure: Pasta
reported an unexpected netlink sequence number. A successful API version
response therefore did not establish working container startup or published ports.

Project configuration now selects `slirp4netns` for both container networking
and the rootless network command. Runtime ports bind to `127.0.0.1`. The old
project service was stopped by identified PIDs, restarted on the CPU compute
node, and the mechanics check passed. Host networking, privileged containers
and site-wide networking changes were unnecessary.

### Tar uploads requested an unmapped host ownership

The official grader initially failed copying `/tmp/patch.diff`: tar metadata
requested UID/GID `115408:115408`, which was invalid in the container's user
namespace. The gold patch never reached the test command.

The archive wrapper sets uploaded tar members to container UID/GID zero while
preserving bytes and mode. It runs only inside the selected grader process;
vendor source and patch content remain intact. Gold execution then advanced
beyond archive upload. Such copy failures remain infrastructure errors rather
than being scored as unsuccessful model repairs.

### Buildah could not use the project scratch temp directory

A dependency build failed creating its rootfs mount under Lustre `.cache/tmp`
with `permission denied`. Ordinary Python temporary files could use that path,
but rootless Buildah mount setup needed a suitable node-local filesystem.

The project `docker` shim overrides `TMPDIR`, `TEMP` and `TMP` only in build
children, using a private node-local directory. General Python work stays under
the project. The NumPy repair and real runtime builds subsequently completed.
No broad permission changes, global temp relocation or unrelated-store deletion
was needed.

### Podman versions were compared against Docker release numbers

The pinned OpenHands builder rejected Podman 5.8.2 with “Docker server version
must be >= 18.09 to use BuildKit.” Those engine version schemes are different;
the comparison did not demonstrate an obsolete Podman engine.

The compatibility initializer explicitly requires a Podman component and
initializes state needed by the audited Podman build path. It does not fake a
Docker version. Native Docker keeps its original check. A local pinned-import
fixture validated initialization, and the later real NERSC build separately
validated the backend path.

### Unqualified images went to the wrong registry

Pulling `swebench/...` used the site's unqualified search list and reached
`registry.suse.com`, which denied access. The preceding cache 404 only meant the
image was not present locally; it did not prove the correct remote was unavailable.

Selected public references are now qualified to `docker.io`; explicit registries
and local images retain their identity. Pinned task metadata is preserved.
Pydicom was then pulled and prepared. SQLFluff's historical HTTP 401 remains an
open access/build item until its own preparation passes. The frozen task list
is not changed to replace inaccessible or failing environments.

### Task YAML omitted required grading sidecars

Preparation first rejected a gold-patch mismatch because YAML alone omitted
metadata stored in separate patch, test and evaluation-script files. Comparing
the incomplete load with the dataset created an identity failure before grading.

The integration uses the pinned official loader and validates instance ID,
repository, base commit and normalized metadata against the dataset. A remote
audit checked 105 frozen tasks and six consumed metadata files per task with
zero failures. Agents receive public task information; references and grading
test identities stay in the grader path.

### Historical pvlib needed a NumPy repair

The first gold control reached collection and failed because NumPy 2 removed
`np.Inf`, which the historical code expected. The fix was a task-image derivative
pinning testbed NumPy to 1.26.4. Original image identity and failed-control evidence
were retained.

Gold passed on the derivative, followed by empty-prediction handling and runtime
checks. The completed NERSC pair used the same prepared image for both model
conditions. Host training dependencies, task test source and scoring were not
edited to obtain acceptance.

### Historical pydicom needed pytest fixture compatibility

The gold control produced 85 failures: missing `file_out`, `fp`, `fp_ex` and other
attributes, plus unexpected warnings. Its image used pytest 8.3.5 on Python
3.8.20. Test classes used plain `setup()`/`teardown()` methods, whose nose
compatibility was removed in pytest 8. Missing setup and cleanup also explained
leaked configuration between tests.

The helper gained a pytest-specific pin rather than applying the unrelated
NumPy fix. The derivative used pytest 7.4.4 and preserved NumPy 1.24.3. An isolated
local fixture reproduced missing `file_out` with pytest 9 and passed with 7.4.4.
The real NERSC gold control then passed, and the imported preparation records
gold and runtime acceptance.

The console recorded versions and identities in
`results/task_environment_repair/20261124T024137-6b1525a9`. See the accepted
[pydicom record](../results/nersc_uploaded/20261124T031613Z/benchmark_15b_window_v1/development/tasks/pydicom__pydicom-1048/prepared.json)
and [pytest's removal notice](https://docs.pytest.org/en/stable/deprecations.html#support-for-tests-written-for-nose).

### Duplicate run banners broke repair bookkeeping

After the repaired image and gold control succeeded, the helper raised “Control
did not record exactly one run.” The CLI prints a run path at startup and after
successful completion. Both lines named `results/grade/20261124T023239-14a20751`;
they did not represent two control executions.

The parser now counts distinct paths. Recovery for this specific failure checks
original arguments, candidate provenance, exact image identity, pinned versions,
command status and control report. It reuses the successful gold control and
runs only the missing blank control, without rebuilding or repeating model
episodes. The console confirms successful recovery.

Regressions cover the actual CLI banners, invalid reports, failed commands,
changed pins and missing-control execution. The latest
[cloud regression](../results/cloud_evaluation_validation/20261124T023913-fa98fc57/validation.json)
passed 91 tests. That count concerns compatibility/reporting and does not count
real benchmark tasks resolved.

### Blank predictions are not an executed negative test

The official grader short-circuits blank predictions. A passing blank control
checks empty-patch handling and denominator integrity, not execution and rejection
of a nonempty incorrect patch. Repair reports now state this distinction.

An earlier workstation
[executed negative control](../results/executed_grader_negative/20261019T222237-60baffd4/negative_validation.json)
used a deliberately wrong numerical derivative and observed the expected failing
test. It remains separate historical evidence, not an executed negative control
for every new NERSC image.

### Setup commits differed while task source trees matched

Startup initially demanded literal HEAD equality with the frozen base. The
published pvlib image had empty setup commit
`96d12dbde601d0d90c6b745fb0f374364ddfc333`, whose parent was frozen base
`e326fa53038f616d949e4f981dab6187d2ca9470`. Their source-tree diff was empty.

The corrected check requires the base commit and compares HEAD's tree, index
and tracked working tree against it. Modified task source still fails. Subsequent
view/edit/test/patch checks execute through the same runtime tools as the agent.
Both imported prepared tasks record passing checkout and tool execution checks.

### CLI and API image IDs used different spellings

Prediction rejected “Prepared runtime image identity changed” although the API
returned `sha256:95d37f...` and the CLI returned the same complete digest without
the prefix. A string comparison had confused spelling with content identity.

Comparisons now normalize only the optional prefix and require all 64 hex digits.
Short or genuinely different hashes still fail. The earlier pre-start attempt
was preserved and reviewed before continuing under the corrected protocol.
This did not authorize resetting an episode that had already generated tokens.

## Agent protocol and model serving

### The nominal context setting did not bound the actual prompt

The LoRA agent first received successful completions and then repeated HTTP 413
responses: prompt plus requested output exceeded the frozen context. The pinned
CodeAct path did not enforce a 6144-token input ceiling merely because
`max_input_tokens` was configured. The server correctly rejected the oversized
request, while the full conversation continued growing.

The explicit `history-window-v1` policy counts the exact chat-template tokens.
It preserves the system/task prefix and latest action/observation exchange,
removes oldest whole exchanges when needed, and clips one oversized recent
observation only with an explicit marker and audit record. Tool-call IDs stay
aligned with observations. It does not silently truncate the issue or assistant
actions; protected content that cannot fit still fails explicitly.

Health reports the policy, and prediction requires the frozen study to match.
Original requests and effective messages are logged. Moving from the old
rejection-only protocol required a fresh study instead of rewriting old outcomes.
The later real LoRA episode reached 30 steps without the old context-error
outcome; acceptance on all five tasks is still incomplete.

The [local context check](../results/context_window_validation/20261124T014727-34734d9b/validation.json)
used the real tokenizers and pinned prompt with synthetic long histories. It
validated bounds and protected-content mechanics, not model quality.

### Nested retry defaults amplified terminal API failures

The agent retried API calls with backoff, and its underlying OpenAI SDK had a
separate retry setting. Terminal context-limit errors generated repeated 413
requests and long waits instead of useful agent steps.

The frozen agent now uses one API attempt and disables nested SDK retries in
its process. A mock confirmed one request on a terminal failure. Original failed
episodes and ledgers remain preserved; retries were not replaced with automatic
resampling under a new label.

### Bounded model outcomes were treated as infrastructure blockers

Loop detection and reaching exactly the configured iteration cap are completed
agent outcomes. They may produce patches that still need official grading.
Initially the reporting layer treated the headless step-limit message as a
blocking infrastructure error even after export and grading succeeded.

The classifier now recognizes the exact loop message and a validated
maximum-iteration message, preserving every error while separating bounded
model outcomes from API/runtime/protocol blockers. It does not mask arbitrary
messages mentioning limits or HTTP 413 failures. Final admission still needs
all frozen development tasks and no blocking errors.

### CPU and GPU services needed separate lifetimes and a tunnel

GPU allocation hosts the model; the CPU allocation hosts task images, OpenHands
tools and official tests. The model binds loopback port 8615, and an SSH tunnel
connects CPU loopback to GPU loopback without a public listener.

Health checks match paired mode, adapter hash, aliases, context policy, task-set
identity, generation limits and deadline. Both aliases use one loaded base and
adapter, with LoRA disabled for the base condition and separate persistent
ledgers. Moving GPUs must reuse the original ledger directory. Losing the CPU
allocation can lose node-local images, a different recovery problem from moving
the model service.

### Preparation reuse works but shared daemon agent concurrency remains open

Task images, gold controls and tool mechanics are independent of model size.
The reuse helper checks compatible protocols, passing source controls, exact
live image IDs and shared runtime mapping. It writes a destination preparation
record without copying predictions/ledgers or taking cleanup ownership of the
source study's images. Keep those images until every borrowing study is finished.

Preparation reuse can avoid rebuilding while another study runs. However, the
pinned runtime still has a broad `close()` default that can remove other runtime
containers. Agent prediction controllers therefore remain sequential when
sharing a daemon. Arbitrary concurrent controllers have not been validated and
are not recorded as a solved capability.

## Operations and delivery

### Existing allocations exhausted the Slurm submission limit

`salloc` failed with `QOSMaxSubmitJobPerUserLimit`. QOS inspection and `squeue`
showed two still-running interactive GPU jobs. Leaving a node prompt was not
enough evidence that its allocation had been released.

The workflow now checks active and pending jobs with their actual QOS names,
identifies the intended GPU job before releasing it, and preserves the separate
CPU container allocation. The observed interactive limit was two submissions
and four hours; current jobs and site policy must be rechecked rather than
treating that observation as a permanent entitlement. The
[NERSC interactive guide](https://docs.nersc.gov/jobs/interactive/) describes the
allocation workflow.

### The deadline guard stopped before either pydicom episode began

After preparation completed, prediction rejected the old GPU allocation because
too little time remained. Two unstarted conditions require a worst-case bound
of two 1800-second episodes plus 300 seconds: 65 minutes. This is admission
headroom, not an estimate that the task will take exactly that long.

The check happens before the first attempt record is written. Pydicom therefore
has a prepared image but no episode or token usage in the export. Continue with
a fresh GPU allocation, the same frozen protocol and existing ledgers. Lowering
task limits, spoofing the deadline or clearing budgets would change the study.

### The project storage guard was mistaken for a NERSC quota failure

Pydicom download was once blocked before pulling its image. Diagnostics showed
50.30 GB project growth, 8.04 GB estimated container-volume growth and a 15 GB
reservation against a 70 GB cap: about 3.35 GB of headroom was missing. Both
filesystems still had substantial physical free space.

The error came from the approved project safeguard. The execution project
already resides on `/pscratch`; moving its name or deleting unrelated scratch
files would not fix project-growth accounting. The container-volume estimate
is conservative and can include other activity on its node filesystem. Preserve
the baseline and cap instead of resetting them to make a check pass.

Pip/uv caches were negligible, and the squashed-image cache alone was too small.
A roughly 7.47 GB preservation tar was identified for possible durable offload.
The supplied procedure uploads it with checksums, verifies the HPSS copy, and
only then removes the staging tar, following
[NERSC HSI guidance](https://docs.nersc.gov/filesystems/hsi/).
**Completed verified HPSS preservation is not confirmed by this export.** Later
pydicom preparation shows its operations passed the storage guard, but does not
identify which storage action supplied the headroom.

### Source delivery had to preserve the extracted runtime

The Git clone is `$HOME/swe-study-source`; execution is under
`$SCRATCH/swe-study-a100`. The latter was extracted from a portable bundle and
had no `.git`. Git commands there failed at the filesystem boundary. With SCP
unavailable, the clone and a generated checked installer became the delivery path.

The installer validates known previous hashes, compiles Python payloads, backs
up changed source and replaces it atomically. Unknown edits stop installation
before writes. An operator-specific `scripts/package_cloud.py` initially blocked
an update and was excluded from runtime overwrite. Shell modes, idempotence,
backups and unknown-edit preservation are covered by
[installer validation](../results/evaluation_installer_validation/20261124T023944-c2a2b9d5/validation.json).

Windows LF-to-CRLF staging warnings were informational; Linux shell contents
and executable modes still needed explicit validation. A separate packaging
problem involved tar entries dated 1970, which normal ZIP source snapshots
could not store. Bundles now use a fixed ZIP-safe date, and snapshots clamp
unsupported filesystem timestamps without changing file content.

### A successful Git upload initially contained no results

Commit `519c6a5`, labeled “upload data,” added only
`results/cloud_package/cloud.tar.gz.json`. That manifest described a 91,566-byte
source archive with 74 files, containing no training losses or benchmark scores.
Git push had succeeded but copied the clone's source metadata rather than the
scratch project's execution records.

The corrected export explicitly copied validation JSON, study JSON and all
three `experiment.json` files into `results/nersc_uploaded`. Commit `2b5d05b`
provided the 38 JSON inputs reviewed here. Weights, optimizer state, environments
and caches were not required for arithmetic review and were not added to Git.
Full agent event logs and raw official grader reports remain useful additional
evidence for a later publication audit.

### Quiet builds and fixed counters looked like a stuck run

The monitor stayed at one prepared task and two graded episodes during the
pydicom runtime build. Preparation writes its accepted record only after image
construction and tool checks, so those counters correctly stayed unchanged.
The outer log was quiet because a nested audited `runtime_build` captured the
actual dependency installation output.

Reading the recorded PID and its `prepare_task_runtime.py` child confirmed
liveness. Following the inner logs exposed downloads, environment creation and
cache cleanup. A cleanup prompt without an explicit automatic-confirmation flag
was investigated as a possible wait, but the build subsequently finished. No
confirmed prompt deadlock or prompt-related repair is claimed.

The monitor itself first failed under old system Python because
`Path.is_relative_to` was unavailable. Launching it with `.envs/eval/bin/python`
resolved that error. Advancing timestamps mean refresh; old 6/30 and 30/30 bars
are the completed pvlib episodes, not a new task's progress or an ETA. Recorded
`running` metadata alone does not prove liveness or forward progress.

### Local WSL and dependency setup had separate challenges

Windows/Node subprocess access failures did not establish that Ubuntu was
unavailable. The standard command runner through `wsl.exe` reached the shared
project, CUDA, GitHub and Docker. Shell-default Python differed from the explicit
bootstrap interpreter, making executable selection part of reproducibility.

Training, agent and evaluator dependencies were isolated because they have
different requirements. An unavailable pinned LiteLLM release required a
recorded compatible 1.53.1 fallback. The old runtime also initializes Chromium
when browsing tools are disabled, so removing that dependency broke startup.

On the Windows-mounted workspace, pytest descriptor capture encountered failing
temporary-file behavior. Checks use `--capture=sys`, no cache provider and a
project-local base temp directory. Sequential API probes use address reuse
while preserving active listeners. A full WSL metadata walk also competed with
model-loading I/O; native Windows inventory reads replaced that large traversal,
with scans paused during GPU loading. These are workstation findings, separate
from native NERSC container and GPU problems. See the
[environment record](environment.md).

## Remaining work and reporting rules

1. Complete the other three NERSC task environments, including gold and runtime
   controls. SQLFluff remains unresolved until its own passing record exists;
   keep the frozen task list intact.
2. Obtain fresh GPU time when a full pair cannot fit, preserving the existing
   per-condition ledgers and CPU task images.
3. Finish the remaining eight 1.5B development episodes, grade saved patches and
   inspect the complete five-task comparison.
4. Reuse validated environments for separate 3B/7B studies with identical task,
   agent, context and generation settings. Keep shared-daemon agent controllers
   sequential until isolation of concurrent cleanup is validated.
5. Aggregate successful/failed training attempts and Slurm allocation charges
   before reporting total compute efficiency.
6. Complete each frozen-five development gate before its frozen 100-task run.
   Report resolution rates with complete denominators and explicit outcomes.
7. Confirm durable preservation independently of scratch staging. Keep adapters,
   matching base snapshots, configurations, results and failures. Archive
   creation alone is not an HPSS backup.

Current evidence establishes three-size training and improved held-out response
prediction, plus a NERSC agent/grader path through two prepared environments and
one completed paired task. Completed development and final SWE-bench quality
comparisons remain pending.
