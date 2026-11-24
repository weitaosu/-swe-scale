# NERSC 3B baseline and LoRA evaluation

Run these stages in order. All compute and containers stay on NERSC. Run the
paired validation now if a one-GPU allocation is still available; CPU archive
work can run separately. The local CPU tests and random NF4/LoRA validation pass.
The real 43-history comparison, NERSC Podman API, task images and agent runs still
require the checks below on the cluster. A compatibility failure stops that stage.

| Stage | Where | Output |
| --- | --- | --- |
| 1. Preserve | CPU allocation | Complete archive plus SHA-256 manifest |
| 2. Paired validation | One allocated A100 40 GB | Base/LoRA losses on the same 43 histories |
| 3. Containers and reference controls | CPU allocation | API, reference patch and tool execution checks |
| 4. Five development tasks | CPU worker plus one GPU model server | Five paired issue-resolution results |
| 5. Frozen final tasks | Same staged workflow | 100-task paired results, resumable between task episodes |

Do not change training epochs while establishing this comparison. Preserve the
100-task list and use the five development tasks for pipeline development.

## 0. Deliver the new scripts using Git

On the local computer, in the project PowerShell terminal:

```powershell
git add -- @(Get-Content deliverables/nersc_evaluation_files.txt)
git commit -m "Add paired validation and staged NERSC benchmark workflow"
git push
```

On a NERSC login terminal:

```bash
module load python
git -C "$HOME/swe-study-source" pull --ff-only &&
python3 "$HOME/swe-study-source/deliverables/nersc_evaluation_patch.py" \
  --root "$SCRATCH/swe-study-a100"
```

The installer checks all existing sources before writing, backs up changed
sources, and preserves unfamiliar edits. The source Git checkout and extracted
scratch runtime are separate directories. Pulling alone does not update scratch.
The installer changes no model weights, adapter weights or training configuration.

## 1. Preserve the completed run — CPU

Use a CPU allocation for hashing and archiving large weights. If you already have
one, reuse it. Otherwise request one from a login terminal outside other jobs:

```bash
salloc --account=m4999 --constraint=cpu --qos=shared_interactive \
  --nodes=1 --ntasks=1 --cpus-per-task=64 --time=04:00:00
srun --ntasks=1 --cpus-per-task=64 --cpu-bind=none --pty bash -l
```

Inside the CPU shell:

```bash
cd "$SCRATCH/swe-study-a100"
source cloud/nersc/env.sh
.envs/train/bin/python scripts/preserve_cloud_results.py \
  --adapter checkpoints/cloud/qwen-3b-32k-1gpu-step-resume-v1 \
  --archive
```

The unique `results/cloud_preservation/<run-id>/` directory contains:

- `preservation.tar`: matching base model, complete adapter/checkpoints, recorded
  data, configuration, code, provenance manifests and available result files.
- `preservation_manifest.json`: SHA-256 for each archived file.
- `preservation.tar.sha256`: checksum of the complete archive.

The archive is uncompressed and needs roughly the size of those files again.
Storage checks preserve the 70 GB growth cap and 25 GB free-space floor. Original
files are retained. After the command succeeds, verify the archive:

```bash
PRESERVE_RUN=$(ls -dt results/cloud_preservation/* | head -n 1)
(cd "$PRESERVE_RUN" && sha256sum -c preservation.tar.sha256)
tar -tf "$PRESERVE_RUN/preservation.tar" | head -n 15
```

Scratch is purgeable. For a durable copy in your HPSS home, run from the login
node after the archive is complete (keep the local archive and originals):

```bash
cd "$SCRATCH/swe-study-a100"
PRESERVE_RUN=$(ls -dt results/cloud_preservation/* | head -n 1)
HPSS_DIR="swe-study-3b-$(date -u +%Y%m%dT%H%M%SZ)"
hsi mkdir "$HPSS_DIR"
hsi put "$PRESERVE_RUN/preservation.tar" : "$HPSS_DIR/preservation.tar"
hsi put "$PRESERVE_RUN/preservation.tar.sha256" : "$HPSS_DIR/preservation.tar.sha256"
hsi put "$PRESERVE_RUN/preservation_manifest.json" : "$HPSS_DIR/preservation_manifest.json"
hsi ls -l "$HPSS_DIR"
```

Require successful transfer exit codes and check the remote sizes. Store the
printed HPSS directory in your project notes. After restoring, verify the tar's
SHA-256 before extracting into a fresh directory. Archived host storage baselines
are historical evidence; initialize new host accounting in a fresh runtime.

## 2. Evaluate base and LoRA on the exact validation split — GPU

Reuse the completed 3B one-GPU allocation if it is still active. Do not start this
inside the two-GPU 7B training allocation. If needed, request a new one-GPU job
from a login terminal, then enter its node:

```bash
salloc --account=m4999 --constraint='gpu&hbm40g' \
  --qos=shared_interactive --nodes=1 --ntasks=1 \
  --cpus-per-task=32 --gpus=1 --time=01:00:00
srun --ntasks=1 --gpus=1 --cpu-bind=none --gpu-bind=none --pty bash -l
```

Run in that GPU shell:

```bash
cd "$SCRATCH/swe-study-a100"
source cloud/nersc/env.sh
unset SWE_STUDY_CONTAINER_ENGINE SWE_STUDY_CONTAINER_STORAGE_MANIFEST DOCKER_HOST
unset PYTORCH_CUDA_ALLOC_CONF
export PYTORCH_ALLOC_CONF=expandable_segments:True

.envs/train/bin/python scripts/cloud_validation.py \
  --adapter checkpoints/cloud/qwen-3b-32k-1gpu-step-resume-v1
```

This is evaluation only: no optimizer updates, saves to the adapter or training
resumes. It loads one NF4 base plus the LoRA, evaluates with the adapter disabled,
then evaluates with it enabled. It reproduces the original Trainer's microbatch-1
history mean, causal assistant labels, complete-history exclusions and seed.
It verifies the validation data hash and retained count against `experiment.json`.

Expected real-run count: **43**. The LoRA loss should reproduce approximately
**0.40881**; a larger discrepancy stops the comparison for inspection. The saved
`results/cloud_validation/<run-id>/comparison.json` records both losses, their
difference, selection hash, precision, versions and the reproduction check.

For adapters trained on multiple GPUs, the original Trainer 5.5.3 evaluation
uses cross-rank token normalization within each step. This can differ from the
canonical single-GPU mean over the same histories. The evaluator saves each
history's loss/count and reconstructs the original multi-GPU aggregation,
including padded final rank batches. Require
`training_metric_reproduction.passed: true`. The direct
`adapter_matches_recorded_loss` can be false for these runs; that fact remains
visible in the report. A failed reconstructed check still stops evaluation.
Never change the saved training loss or loosen the tolerance to force a pass.
Compare model sizes using their canonical base/adapter history means.

Interpret `relative_loss_reduction_percent`: positive means the adapter predicts
held-out assistant responses better. It is not issue-resolution accuracy. The
original validation pass took about 47 seconds; this command does two passes plus
model loading and preprocessing. Release a dedicated evaluation GPU allocation
once this finishes if the container stages below are not ready.

## 3. Prepare containers, pinned dependencies and one task — CPU

Keep all commands in this section on the same CPU allocation. Reuse stage 1's CPU
job if it has enough time. NERSC supports Podman-HPC; the project's Python Docker
clients connect to its compatibility API. This path must pass the API check on
your node before any benchmark claims are possible.

Inside the CPU allocation:

```bash
cd "$SCRATCH/swe-study-a100"
source cloud/nersc/env.sh

.envs/train/bin/python scripts/study.py checkout &&
.envs/train/bin/python scripts/bootstrap_scaffolds.py --name eval &&
.envs/train/bin/python scripts/bootstrap_scaffolds.py --name agent
```

The pinned source revisions come from `manifests/source_repositories.json`.
Agent and official-grader dependencies stay separate from the training environment.
Do not proceed if installation or `pip check` fails.

Start a project-scoped Podman API service inside the compute allocation. A GPU
allocation's CPUs can also run these preparation and grading steps; its GPU can
later serve the model on the same node. Do not run the evaluation service on a
login node. The launcher checks the allocation and rejects login-node startup.

```bash
source cloud/nersc/container_env.sh
command -v podman-hpc
command -v slirp4netns
podman-hpc --version

bash cloud/nersc/recover_containers.sh
```

The API check must print `passed: true`. It checks container creation, command
execution, file export and a loopback-published port, then removes its own test
container. If it fails, inspect `results/container_api_check/<run-id>/` and
`logs/container-service-<job-id>-<host>-<timestamp>.log`. Stop here and resolve that failure before
allocating a GPU for agent work. Podman-HPC's availability alone does not prove
Docker API compatibility for OpenHands.

The project uses `CONTAINERS_CONF_OVERRIDE` pointing to
`cloud/nersc/containers-network.toml`. Its `[containers] netns = "slirp4netns"`
setting preserves other site settings and avoids the Docker-compatible API's
bridge default. That default can fail in Netavark/iptables while publishing a
port, even when `podman-hpc` itself is installed and image pulls succeed. The
`[network] default_rootless_network_cmd = "slirp4netns"` setting also selects
Slirp for the shared rootless network instead of Pasta. The check saves the
API-created container's network mode before starting it; a mismatch points to an
existing service that has not loaded the intended configuration. The
API check must record `network_mode: slirp4netns` and pass the loopback HTTP test.
This is a configuration fix to validate on NERSC, not a claim of a passed check.
The probe, OpenHands runtime, and grader also explicitly request Slirp in the
Docker-compatible API call. Native Docker requests retain their existing network
settings. The recovery command stops only the matching project API service,
starts it with the current configuration, waits for its ping response and runs
the compatibility check. If a reachable service cannot be identified safely,
recovery stops for inspection rather than signaling unrelated processes.

When upgrading or moving an existing project service, stop it on its old host
before starting the replacement (including an older service on `login23`):

```bash
source cloud/nersc/container_env.sh
.envs/eval/bin/python scripts/stop_container_service.py --stop
```

This matches only this user's Podman API-service command and exact project Unix
socket, sends SIGTERM, and records the affected PIDs. It does not stop model
servers or remove images. Configuration changes require a fresh API service;
sourcing the environment in a client shell does not reconfigure an existing one.

The writable graph store is explicitly scoped under
`/tmp/<uid>_swe_study_podman/graph` on the CPU node, following Podman-HPC's node-local
writable-storage design. Config and migrated images stay under the project.
Per-node physical storage baselines account for the external graph store as well
as project growth. Do not run services on two CPU nodes against this project store.
No global Podman/Docker configuration or unrelated images are modified.

Initialize the frozen study and prepare its first development task:

```bash
.envs/eval/bin/python scripts/validate_task_metadata.py &&

.envs/eval/bin/python scripts/cloud_benchmark.py plan \
  --suite development --state results/benchmark_3b

.envs/eval/bin/python scripts/cloud_benchmark.py prepare \
  --suite development --state results/benchmark_3b --max-tasks 1
```

Preparation pulls only the needed image, runs the reference-patch gold control,
checks the blank-patch report, builds the OpenHands runtime and replays view,
edit, command/test and patch-export operations. The blank control checks empty
handling; the official grader short-circuits it instead of executing tests.

The metadata check compares all five development and 100 final tasks against
the frozen datasets without pulling images or running a model. Preparation uses
the pinned official loader: `task.yaml` contains only short metadata, while
`gold.patch`, `test.patch`, `problem_statement.md`, `tests.json` and `eval.sh`
provide the remaining fields. Comparing the YAML alone incorrectly treats the
absent patch as a mismatch. Actual dataset/task differences still stop the run.

On Podman-HPC, project CLI pulls and selected-task grader API pulls explicitly
qualify Docker Hub images: `swebench/...` becomes `docker.io/swebench/...`, and
`python:...` becomes `docker.io/library/python:...`. Explicit registries and
digests are preserved. This prevents the site's unqualified-name search from
sending Docker Hub task names to `registry.suse.com`. Frozen task metadata is
unchanged. After a code upgrade changes the recorded protocol, preserve the old
study directory and use a fresh `--state` for both plan and subsequent commands.

The first frozen task is `pvlib__pvlib-python-807`. Historical pvlib images failed
under NumPy 2 in the workstation check. If its gold control fails with that same
diagnosed dependency issue, retain the failed control and apply the existing
dependency-only repair, replacing `FAILED_GOLD_RUN` with the printed run path:

```bash
.envs/train/bin/python scripts/repair_task_environment.py \
  --instance-id pvlib__pvlib-python-807 \
  --numpy-version 1.26.4 \
  --evidence FAILED_GOLD_RUN

.envs/eval/bin/python scripts/cloud_benchmark.py prepare \
  --suite development --state results/benchmark_3b --max-tasks 1
```

Use `.envs/eval/bin/python` instead of `.envs/train/bin/python` for that repair if
Docker's Python SDK is unavailable in the training environment; the repair itself
uses the installed grader for its reference controls. Do not apply this repair
to unrelated failures. SQLFluff's published image previously returned HTTP 401;
access/build problems must be resolved using its pinned task source. Keep the
frozen task list; do not replace failing environments or score them as model errors.

Preparation can be run for a small batch using `--max-tasks 2` or `3` if storage
permits, before requesting a GPU. Start with one task to validate the workflow.

## 4. Run the five paired development tasks

### 4a. Start a paired model server — GPU terminal

Only request this GPU after CPU preparation passes. Use a separate login terminal
for the GPU request, so the CPU container service keeps running:

```bash
salloc --account=m4999 --constraint='gpu&hbm40g' \
  --qos=shared_interactive --nodes=1 --ntasks=1 \
  --cpus-per-task=32 --gpus=1 --time=04:00:00 --job-name=eval-3b
srun --ntasks=1 --gpus=1 --cpu-bind=none --gpu-bind=none --pty bash -l
```

Inside the GPU shell, record the host/job for the CPU tunnel:

```bash
hostname -s
echo "$SLURM_JOB_ID"
cd "$SCRATCH/swe-study-a100"
source cloud/nersc/env.sh
unset SWE_STUDY_CONTAINER_ENGINE SWE_STUDY_CONTAINER_STORAGE_MANIFEST DOCKER_HOST
unset PYTORCH_CUDA_ALLOC_CONF
export PYTORCH_ALLOC_CONF=expandable_segments:True

bash cloud/run.sh serve --size 3B \
  --adapter checkpoints/cloud/qwen-3b-32k-1gpu-step-resume-v1 \
  --paired --max-context 8192 --max-new-tokens 2048 --port 8615 \
  --task-budget-file results/benchmark_3b/development/task_budgets.jsonl \
  --budget-ledger-dir results/benchmark_3b/development/token_ledgers
```

Keep this terminal open. Both model aliases use the same base weights, NF4/BF16
precision and generation settings. The base alias disables LoRA. Independent
per-condition, per-task 32,768-token ledgers persist across server restarts.
The benchmark's frozen agent context is 8K even though training used 32K; use
the same context for both conditions, and do not change it after development.

### 4b. Connect the CPU worker to the GPU server

In the existing CPU shell, replace the node value with the GPU hostname printed
above. This SSH tunnel stays entirely inside NERSC and exposes only loopback:

```bash
export GPU_NODE=nid001008
export MODEL_TUNNEL_SOCKET="$STUDY_ROOT/.cache/model-tunnel-${GPU_NODE}.sock"
ssh -M -S "$MODEL_TUNNEL_SOCKET" \
  -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -f -N -L 127.0.0.1:8615:127.0.0.1:8615 "$USER@$GPU_NODE"
curl --fail --retry 20 --retry-connrefused --retry-delay 2 \
  http://127.0.0.1:8615/health
```

Use valid NERSC SSH/sshproxy credentials for compute-node access. If the health
request fails, resolve the tunnel/authentication before starting predictions.
The `health` response must show paired mode, both aliases and the expected adapter.
If a local port already has a listener, preserve it and use a different tunnel
port plus the matching `--base-url` on the prediction command.
Before reconnecting to a new GPU node, close only this study's old SSH tunnel:

```bash
ssh -S "$MODEL_TUNNEL_SOCKET" -O exit "$USER@$GPU_NODE"
```

### 4c. Generate paired patches — CPU controller with GPU server

```bash
.envs/eval/bin/python scripts/cloud_benchmark.py predict \
  --suite development --state results/benchmark_3b --max-tasks 1
```

This runs the same task once with the base and once with LoRA. Each condition has
one worker, at most 30 agent steps, 30 minutes and 32,768 generated tokens.
The agent container receives public issue metadata; reference patches and hidden
test identities remain grader-only. Each saved task record includes predictions,
errors and prepared-image identities. Existing completed predictions are skipped.

If the server reports a Slurm deadline too close for the remaining episodes, the
controller stops before starting another condition. An interrupted episode is
preserved and blocks automatic resampling; review infrastructure failures rather
than silently restarting attempts. Do not remove token ledgers to reset budgets.

Once predictions for the prepared batch are saved, the model server can stop.
For a dedicated evaluation allocation, release that allocation from a login shell
with `scancel JOB_ID`, using the GPU job ID printed in 4a. Preserve the separate
7B training allocation. Stopping just the Python server does not release Slurm.

### 4d. Grade saved patches — CPU only

```bash
.envs/eval/bin/python scripts/cloud_benchmark.py grade \
  --suite development --state results/benchmark_3b --max-tasks 1

.envs/eval/bin/python scripts/cloud_benchmark.py summary \
  --suite development --state results/benchmark_3b
```

No model or GPU is needed for this stage. The pinned official grader applies
the saved patches and runs tests. Infrastructure errors stop grading; they are
not silently turned into valid unsuccessful model results.

After both conditions for this task have been graded, optionally free only newly
created task images so the next task fits the storage cap:

```bash
.envs/eval/bin/python scripts/cloud_benchmark.py cleanup \
  --suite development --state results/benchmark_3b --max-tasks 1
```

This removes only image IDs recorded as new during that task's preparation, never
pre-existing images or unrelated containers. Removal is non-forced; images in use
are preserved. Retain task predictions, logs, image identities and controls.

Repeat 3's `prepare`, then 4a–4d for the next ungraded task. Each stage follows the
frozen order and skips completed records. Keep the CPU allocation/service alive
through a prepared batch: node-local images may be lost when the allocation ends.
If that happens after predictions, restore the exact recorded image identities
before continuing; the runner refuses to rebuild a different environment silently.

For a prepared batch of up to three tasks, use `--max-tasks 3` in predict and grade.
Do not run five-task generation assuming it fits a four-hour allocation: the
worst-case agent limits alone total five hours for both conditions.

The development summary is `results/benchmark_3b/development/summary.json`.
Require complete coverage for both conditions and inspect all agent errors before
proceeding. A one-task setup diagnostic is not a five-task comparison.

## 5. Freeze and run the final 100-task comparison

The final runner refuses to start until both conditions have completed the frozen
five development tasks and their agent errors have been resolved. It preserves
the same adapter hash, agent configuration, source revisions, context and limits.
Changing that protocol requires a separate study directory and development run.

On the CPU worker:

```bash
.envs/eval/bin/python scripts/cloud_benchmark.py plan \
  --suite final --state results/benchmark_3b

.envs/eval/bin/python scripts/cloud_benchmark.py prepare \
  --suite final --state results/benchmark_3b --max-tasks 1
```

Start a fresh paired GPU server using 4a, changing only its task-budget paths:

```bash
bash cloud/run.sh serve --size 3B \
  --adapter checkpoints/cloud/qwen-3b-32k-1gpu-step-resume-v1 \
  --paired --max-context 8192 --max-new-tokens 2048 --port 8615 \
  --task-budget-file results/benchmark_3b/final/task_budgets.jsonl \
  --budget-ledger-dir results/benchmark_3b/final/token_ledgers
```

Update the CPU tunnel to the new GPU node if needed. On the CPU controller:

```bash
.envs/eval/bin/python scripts/cloud_benchmark.py predict \
  --suite final --state results/benchmark_3b --max-tasks 1

# Release a dedicated GPU allocation after its prepared predictions are saved.

.envs/eval/bin/python scripts/cloud_benchmark.py grade \
  --suite final --state results/benchmark_3b --max-tasks 1

.envs/eval/bin/python scripts/cloud_benchmark.py summary \
  --suite final --state results/benchmark_3b

.envs/eval/bin/python scripts/cloud_benchmark.py cleanup \
  --suite final --state results/benchmark_3b --max-tasks 1
```

Repeat prepare/predict/grade for the remaining tasks, using small prepared batches
that fit your allocation and storage. The fixed denominator remains 100; missing
tasks are listed, not dropped. The summary marks partial runs as incomplete and
reports the paired difference only when both conditions have full coverage.

The upper bound from 200 episodes × 30 minutes is 100 GPU-server hours, before
CPU setup/grading. This is a limit calculation, not a runtime prediction. Measure
the development tasks before planning the full run. Use the five development
tasks to choose settings, then hold settings fixed for final evaluation.

When complete, report base resolved/100, LoRA resolved/100, the percentage-point
difference, agent errors, grading failures/incomplete counts, task manifest,
adapter/base revisions and the frozen protocol. Label this the project's frozen
100-task subset of SWE-bench Verified, not the full 500-task Verified benchmark.
Rerun preservation after evaluation to archive the new results too.

## Sources and remaining validation

- [NERSC Podman-HPC](https://docs.nersc.gov/development/containers/podman-hpc/overview/)
- [Podman Docker-compatible API service](https://docs.podman.io/en/latest/markdown/podman-system-service.1.html)
- [NERSC interactive jobs](https://docs.nersc.gov/jobs/interactive/)
- [NERSC scratch preservation](https://docs.nersc.gov/filesystems/perlmutter-scratch/)
- [NERSC HPSS hsi transfers](https://docs.nersc.gov/filesystems/hsi/)
- [Official SWE-bench evaluation](https://www.swebench.com/SWE-bench/guides/evaluation/)

Local tests establish loss/reduction compatibility, adapter switching, persistent
budgets, frozen denominators and fixture preservation. They do not establish
NERSC container compatibility, image availability, full task performance or the
real 3B baseline loss. Save every cluster run's audited output before reporting
those stages as completed.
