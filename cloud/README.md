# A100 cloud training and inference

This bundle runs the existing Qwen2.5-Coder + SWE-Gym QLoRA pipeline on **one Linux
node with one or more NVIDIA GPUs**, including multiple A100s. It includes setup,
pinned downloads, data preparation, distributed training, exact resume, fresh-process
adapter reload validation, prompt inference, a chat API, and a Slurm launcher.

No cloud jobs are submitted by packaging the files. Run the commands below on your GPU server.

## 1. Upload and extract

Upload `swe-study-a100*.tar.gz` and its `.sha256` sidecar to persistent storage.
Use a fresh destination, so earlier checkpoints and accounting stay intact:

```bash
sha256sum -c swe-study-a100-<timestamp>.tar.gz.sha256
tar -xzf swe-study-a100-<timestamp>.tar.gz
cd swe-study-a100
```

Requirements: Linux x86_64, Python **3.11 or 3.12** with `venv`/pip, working NVIDIA
driver and GPU allocation, outbound HTTPS to PyPI/PyTorch/Hugging Face, and persistent
disk. Start with about **100 GB free** for the default 7B workflow. The scripts retain
the project's **70 GB incremental cap and 25 GB free-space floor**; unrelated writes
on that filesystem count conservatively toward the cap. No Docker is needed here.
Use the fresh bundle, rather than copying workstation `.envs`, `.cache`, `results`,
`storage_baseline.json`, or `docker_storage.json`.

## 2. Setup and prepare

Run setup on a GPU node/allocation so its CUDA check can execute:

```bash
bash cloud/run.sh setup
bash cloud/run.sh prepare --size 7B
```

For another interpreter: `PYTHON=/path/to/python3.12 bash cloud/run.sh setup`.
If `venv` is missing, use your provider's Python environment/module with venv support
or install its matching `python3-venv` package using the provider's documented method.
The scripts make no system package or driver changes.

The default install uses the existing pinned stack: PyTorch 2.9.1 from the **CUDA 12.6**
wheel index, Transformers 5.5.3, PEFT 0.18.1 and bitsandbytes 0.49.2. Setup checks
package dependencies and runs a real BF16/NF4 kernel check. Use `setup --cuda cu128`
or `--cuda cu130` only with a compatible driver. CUDA runtime wheels do not require
a locally installed `nvcc`; a working compatible NVIDIA driver is required.
See [official PyTorch wheel choices](https://pytorch.org/get-started/previous-versions/)
and [bitsandbytes installation requirements](https://huggingface.co/docs/bitsandbytes/v0.49.2/en/installation).

Downloads use immutable model/dataset revisions from
`manifests/planned_downloads_4090.json`, verify sizes/LFS SHA256, and reuse matching
files. Existing mismatched files are preserved and reported. All required assets are
public; if your environment requires authentication, export `HF_TOKEN` in your shell.
Tokens are not bundled or passed as command-line arguments.

Preparation fetches five datasets, screens held-out benchmark overlap, recreates the
task-disjoint train/validation split, preserves provenance, and reports assistant-only
tokenization coverage. Use `--size 1.5B`, `--size 3B`, or `--size all` as needed.
Models and data stay under the extracted project.

## 3. Train on one A100 40 GB, or select more

```bash
bash cloud/run.sh doctor
bash cloud/run.sh check-distributed --gpus 1
bash cloud/run.sh train --size 7B
```

Your target is up to **four A100 40 GB GPUs**. The default uses **one GPU** to keep
the allocation small. Select `--gpus 2` or `--gpus 4` when you want more throughput.
The launcher checks the visible allocation and starts `torchrun` internally. For four GPUs,
the default is one example per GPU with two gradient accumulation steps: effective
batch **8**. With one GPU, accumulation is eight; with two, it is four. Use
`--gpus 4` to select the first four visible GPUs, or set `CUDA_VISIBLE_DEVICES=0,1,2,3`.
GPU counts that do not divide eight require explicit `--grad-accum N`. Before using
multiple GPUs, run `bash cloud/run.sh check-distributed --gpus 4` (or 2). This uses
a tiny random model to check actual NF4/LoRA training, NCCL/DDP synchronization,
checkpoint saves and fresh reload; it is separate from real-model validation.

DDP replicates the frozen NF4 base on each GPU and synchronizes LoRA gradients;
VRAM is not pooled. More GPUs improve data parallel throughput, but a longer sequence
must still fit on each GPU. Use one node; this launcher does not implement multi-node training.
There is no CPU offload or model-parallel inference.

Defaults: 3 epochs over **all histories that fit the selected context**, LoRA rank 16,
alpha 32, learning rate 1e-4, BF16, gradient checkpointing, microbatch 1, assistant-only
causal loss, and target-only logits. The best validation-loss adapter is saved to
`checkpoints/cloud/qwen-7b/`. Epoch checkpoints retain optimizer/scheduler/RNG state;
up to two are retained, with the best checkpoint preserved by Trainer. The final
adapter is reloaded and compared against reference logits in a fresh process.
The source/target file hashes, resolved config, world size, GPU hours, losses,
token throughput and per-rank memory events are recorded.

For a short integration run with a separate output:

```bash
bash cloud/run.sh train --size 7B --max-steps 2 --output checkpoints/cloud/qwen-7b-check
```

To inspect the configuration without loading a model:

```bash
bash cloud/run.sh train --size 7B --gpus 4 --dry-run
```

To run setup, prepare and train sequentially:

```bash
SIZE=7B bash cloud/all.sh
```

### Context coverage matters

The existing real-data audit has 439 training and 50 validation histories. Complete
8K histories cover **37 train / 9 validation (9.4%)**. At 16K the count is **229 / 30
(53.0%)**; at 32K it is **400 / 43 (90.6%)**. Overlength histories are excluded and
reported, never silently truncated. The default 8K is the previously tested setting,
and should be treated as an integration starting point with a short-history bias.

For broader data coverage on your A100 allocation, first try a short 32K run:

```bash
bash cloud/run.sh prepare --size 7B --max-length 32768
bash cloud/run.sh train --size 7B --max-length 32768 --max-steps 2 \
  --output checkpoints/cloud/qwen-7b-32k-check
# Inspect losses, exclusions, per-GPU memory and reload validation before a full run.
bash cloud/run.sh train --size 7B --max-length 32768 \
  --output checkpoints/cloud/qwen-7b-32k
```

32K memory/throughput and multi-GPU behavior must be measured on your node. If it
does not fit, use 16384 in both prepare and train and choose a fresh output. The
launcher keeps microbatch 1 for the tested target-only loss implementation.

### Resume after an interruption

Find a complete `checkpoint-N` inside your chosen output, then reuse the **same
training options, GPU count, data, output and epoch/step schedule**:

```bash
bash cloud/run.sh train --size 7B \
  --resume checkpoints/cloud/qwen-7b/checkpoint-10
```

Replace `checkpoint-10` with the directory actually present. Exact resumes require
optimizer, scheduler, Trainer state, adapter and per-rank RNG files. A saved plan
beside the output enforces matching settings/data. Checkpoint numbers depend on
the retained data and GPU configuration. Training never overwrites a nonempty
output unless an explicit compatible resume is supplied. If a job failed before
its first complete checkpoint, keep its evidence and use a fresh `--output`.

### Four-hour 7B sessions with per-step recovery checkpoints

The optional `configs/cloud/qlora_7b_step_resume.yaml` profile keeps the existing
epoch validation and best-adapter selection, while saving recovery state after
every **optimizer step** (after gradient accumulation). It retains three
checkpoints including the best evaluated checkpoint. Saving and hashing optimizer
state adds filesystem I/O; it does not copy the frozen base model each step.

Use two GPUs in every session. The output in this profile is fixed and must not
be changed to the next Slurm job ID. In the first allocated session:

```bash
bash cloud/run.sh train --size 7B --gpus 2 \
  --config configs/cloud/qlora_7b_step_resume.yaml
```

After the prior allocation has ended, request another two-GPU allocation, load
`cloud/nersc/env.sh`, and resume with exactly the same configuration and data:

```bash
bash cloud/run.sh train --size 7B --gpus 2 \
  --config configs/cloud/qlora_7b_step_resume.yaml --resume latest
```

`--resume latest` accepts only checkpoints with a completion record matching the
configuration, step, GPU count, payload hashes and every rank's RNG state. An
interrupted or corrupt save is skipped in favor of an earlier verified one. The
record is published after all ranks finish saving. Progress after that checkpoint
is repeated; the epoch and learning-rate schedule continue rather than restarting.
The original explicit `--resume path/to/checkpoint-N` remains available for
legacy checkpoints. Do not change an already-running experiment to this profile.

The optional synthetic interruption/resume diagnostic is:

```bash
.envs/train/bin/python scripts/validate_checkpoint_resume.py --gpus 2
```

It uses an isolated random-model fixture, interrupts after checkpoint 2, resumes
to step 4 with optimizer state, and validates a fresh adapter reload. It does not
establish 7B memory feasibility or quality. Exact resume does not imply bitwise
identical arithmetic across hardware/software changes.

### Slurm

Edit `cloud/train.slurm` for the cluster partition/account and A100 count/memory,
then submit **from the project root** after setup and data preparation:

```bash
sbatch cloud/train.slurm --size 7B
# Example longer-context job after its memory check:
sbatch cloud/train.slurm --size 7B --max-length 32768 \
  --output checkpoints/cloud/qwen-7b-32k
```

The template requests one A100 GPU, 8 CPUs, 64 GB host RAM, and 24 hours. For four GPUs:

```bash
sbatch --gres=gpu:a100:4 --cpus-per-task=16 --mem=96G \
  --export=ALL,GPUS=4 cloud/train.slurm --size 7B
```

Cluster
resource names vary. Keep `--ntasks=1`; the script launches one torchrun worker per
visible GPU. Logs include `slurm-*.out/err` and the project's audited run directories.
For a plain SSH server, run long commands inside your existing `tmux`/`screen` session.

## 4. Infer with a baseline or adapter

Inference loads one model on the first visible GPU. It uses the same NF4/unquantized
layer precision and BF16 autocast as training. A LoRA adapter requires its original
base model; adapters do not contain the base weights.

```bash
# Baseline
bash cloud/run.sh infer --size 7B --prompt 'Write a Python function that merges two sorted lists.'

# Your trained adapter
bash cloud/run.sh infer --size 7B --adapter checkpoints/cloud/qwen-7b \
  --prompt 'Write a Python function that merges two sorted lists.'

# For a 32K-trained adapter, set its total prompt+completion context explicitly.
CUDA_VISIBLE_DEVICES=0 bash cloud/run.sh infer --size 7B \
  --adapter checkpoints/cloud/qwen-7b-32k --max-context 32768 \
  --prompt 'Explain this traceback and suggest a fix.' --max-new-tokens 1024
```

Use `--messages path/to/messages.json` instead of `--prompt` for a project-local
JSON array of chat messages. Add `--output results/my-generation.json` to retain a
named result; it must be a fresh path. Every invocation also saves `generation.json`
in its audited run. Greedy generation is the default; use `--temperature 0.7` for
sampling. Context overflow is an error, and input is never silently truncated.

## 5. Serve a local chat API

```bash
CUDA_VISIBLE_DEVICES=0 bash cloud/run.sh serve --size 7B \
  --adapter checkpoints/cloud/qwen-7b --port 8615
```

The API binds to `127.0.0.1`. From your laptop, open an SSH tunnel:

```bash
ssh -N -L 8615:127.0.0.1:8615 USER@GPU_HOST
```

Then call it from a second terminal:

```bash
curl http://127.0.0.1:8615/health
curl http://127.0.0.1:8615/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Write a Python binary search."}],"max_tokens":256,"temperature":0}'
```

Omitting `model` uses the loaded model; its name is also listed by `/v1/models`.
Omit `--adapter` to serve the baseline. Use `--max-context 32768` for longer context.
`--development-budget` enables the existing frozen-task-only 32768-token budget;
leave it off for ordinary prompts. Chat generation does not execute tools. SWE-bench
agent rollouts and Docker grading remain a separate, independently validated setup.

## Memory-safe loss for long assistant responses

An opt-in Qwen2 loss path limits FP32 cross-entropy workspaces to token chunks
and omits unused vocabulary logits from labeled training/evaluation outputs.
It preserves target-only causal labels, context limits, example selection and
loss normalization. Checkpoint recomputation frees each chunk's loss workspace;
unlabeled generation and fresh adapter reloads use the original forward path.

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SWE_STUDY_LOSS_CHUNK_TOKENS=128
bash cloud/run.sh train --size 3B --gpus 1 \
  --config configs/cloud/qlora_3b_step_resume.yaml --resume latest
```

Set both variables in each new allocation before resuming. The existing config,
output directory and optimizer checkpoints remain valid; floating-point reduction
order can differ, so this is not a bitwise-identical continuation. Activation is
recorded in worker events and final experiment metadata. It does not change
already-running processes or pool the memory of DDP replicas. Real 32K A100
feasibility still requires completing the affected batches on that hardware.

For a previously extracted NERSC runtime, apply
`deliverables/nersc_memory_patch.py --root "$SCRATCH/swe-study-a100"` with Python
after the checkpoint support installer. It checks source identity, backs up the
changed training source and preserves unfamiliar files. CPU equivalence tests and
optional synthetic GPU/recovery checks are available through
`scripts/validate_memory_loss.py --gpu --checkpoint-resume`.

## Outputs and checks

- `results/cloud_*/<run-id>/`: command records, stdout/stderr, events and source snapshots.
- `results/cloud_train_rankN/<run-id>/`: per-worker training and memory events.
- `checkpoints/cloud/<name>/`: adapter/tokenizer, reload reference, metrics, resume checkpoints.
- `checkpoints/cloud/<name>.plan.json`: exact resume plan and source data hashes.
- `manifests/`: frozen revisions, cloud storage baseline, installed package freeze.
- `data/tokenized/<model>-<context>/manifest.json`: retained/excluded history counts.

Run the regression suite without loading full models:

```bash
.envs/train/bin/python -B -m pytest -q --capture=sys -p no:cacheprovider \
  --basetemp .cache/tmp/pytest-cloud
```

Preserve the adapter directory **and its matching model snapshot**, plan, data hashes,
package freeze and run logs when moving results to another server. Keep complete
`checkpoint-N` folders if you need exact training continuation. Storage guards stop
the work before its floor/cap is exceeded and never prune unrelated files.

These helpers are generated and locally tested with CPU/synthetic checks. Full A100
training, multi-GPU NCCL execution, throughput and 32K memory feasibility still require
your cloud hardware. Successful checkpoint validation is not a benchmark score.
