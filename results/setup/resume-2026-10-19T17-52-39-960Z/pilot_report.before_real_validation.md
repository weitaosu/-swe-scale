# Pilot validation evidence

The GPU tests below used a **random tiny Qwen2 architecture and synthetic text**, created offline by scripts/tiny_gpu_fixture.py. They did not use Qwen2.5-Coder pretrained weights or SWE-Gym. They validate code/runtime plumbing only; the project's LoRA milestone remains incomplete.

| Check | Evidence |
| --- | --- |
| Initial NF4 QLoRA | results/train/20261019T164855-ead23d01/metrics.json |
| Resume from Trainer checkpoint 3 to step 6 | results/train/20261019T165453-6e72dc69/metrics.json |
| Fresh-process reload after resume | results/validate-checkpoint/20261019T165807-f950e3ee/reload_validation.json |
| Real-GPU HTTP interface, in-process ASGI transport | results/synthetic_server/20261019T170520-908242ff/server_validation.json |
| Full shell commands/stdout/stderr | results/setup/setup-2026-10-19T16-35-12-041Z/commands/ |

The fixture has two transformer layers, hidden size 64, and 8,192 trainable LoRA parameters. Base parameters are frozen. Initial training used eight synthetic training tasks and two validation tasks, three optimizer steps, microbatch one, 128-token context, NF4 double quantization, BF16 compute, and rank four. It processed 48 nonpadding tokens, including nine assistant targets, across three microbatches. Training loss fell approximately 4.179 to 4.118; validation loss reached approximately 4.104. Finite nonzero adapter gradient checks and changed adapter hashes were recorded.

Resume restored optimizer and Trainer state, continued to global step six, and retained 28 optimizer state entries. The corrected resume check compares final weights against checkpoint weights; hashes differ and best validation loss is 4.08110. An earlier step-four resume inherited a zero learning rate from the completed three-step schedule. Its logs are retained, but it is not used as evidence of resumed learning. The verified step-six run records the inherited schedule and subsequent nonzero learning rates.

Fresh-process NF4 adapter reload passed a fixed-prefix logits comparison: maximum absolute difference 0.001953125, with atol 0.02 and rtol 0.01. This tolerates small BF16 kernel differences; it is not a claim of bitwise equality.

The server test loaded the synthetic adapter on the real GPU and checked health/models, generation, SSE formatting, wrong-model rejection, context overflow rejection, task identification, persistent per-task token accounting, and HTTP 429 after token exhaustion. It used an in-process TestClient because listening-socket/network access is unavailable; actual networked OpenHands integration is untested.

Tiny training peak allocation was 19,492,864 bytes and peak reservation 25,165,824 bytes. Initial Trainer wall time was about 0.827 seconds; resumed Trainer wall time about 0.657 seconds, including its evaluation/checkpoint activity. Model loading and final export are excluded. These measurements cannot estimate real-model VRAM, throughput, hours, or downstream repair quality.

An initial run failed on an unsupported Transformers TrainingArguments option; the implementation was corrected and the failure logs retained. SFT now uses Transformers Trainer plus PEFT with verified explicit labels. TRL is not required by this implementation.

Real validation still required: inspect Qwen2.5 tool/template masks, perform the 1.5B four-example overfit and bounded 25-step pilot, verify adapter changes/base freezing, reload/resume and server loading, then measure 3B/7B memory probes sequentially. Use actual retained tokens and measured throughput for any training-time estimate. Inspect best-checkpoint and loss reports before accepting useful learning. No 7B fit or production training-time claim follows from these fixture results.
