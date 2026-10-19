# RTX 4090 Constraints for the CS6158 Project

## Assessment and recommendation

A single desktop NVIDIA RTX 4090 with 24 GB VRAM is a viable starting platform for the first two project milestones and for a modified version of the core adapter-training study. The main changes concern training precision, context length, experiment scheduling, and evaluation concurrency. The 7B model at a 32K training context requires a feasibility test before we commit to it.

We should retain the proposed three model sizes, three training-data conditions, three unmodified baselines, and approximately 100 evaluation tasks. We should start with a common QLoRA recipe, profile the longest 7B examples, and choose a shared production configuration before launching the nine training runs. The optional 32B baseline in BF16 cannot reside entirely on this GPU and should use separate hardware or be omitted.

The assessments below are hardware planning estimates. Exact memory use and throughput depend on the implementation and must be measured on our machine.

## Project scope

The proposal studies Qwen2.5-Coder 1.5B, 3B, and 7B trained separately on SWE-Gym, SWE-smith, and SWE-Master-derived data. Nine adapted models and three corresponding unmodified checkpoints form twelve evaluation conditions. Each condition uses the same agent scaffold and the same approximately 100 SWE-bench Verified tasks. The proposal also permits an optional unmodified 32B comparison if compute allows.

Our first two milestones are:

- Sep 14-27: set up SWE-bench, Qwen models, and the common agent scaffold.
- Sep 28-Oct 11: prepare SWE-Gym data and validate LoRA training.

## Hardware constraint

The desktop RTX 4090 provides 24 GB of GPU memory. [1] A model fitting in memory for inference does not establish that its training configuration will fit. Training also requires activations, gradients for trainable parameters, optimizer state, and temporary buffers. Sequence length and microbatch size can substantially increase peak memory. [2]

LoRA trains a small set of adapter parameters while freezing the original model. QLoRA additionally stores the frozen model in 4-bit form, reducing weight memory. This makes QLoRA a useful candidate for the 4090, but it does not remove activation memory or guarantee that a long trajectory will fit. [2, 3]

## Tasks affected by the constraint

| Project task | Assessment on one 4090 | Required or proposed adjustment |
| --- | --- | --- |
| SWE-bench setup and scaffold debugging | Feasible | Run one model at a time and begin with development tasks. |
| Data cleaning, overlap checks, splitting, and tokenization | Feasible; mainly CPU, RAM, and storage | Follow the data preparation plan. GPU memory is not the main constraint. |
| Initial 1.5B training validation | Feasible with modest sequence lengths | Begin with 4K-8K examples, then test longer examples. |
| Production 1.5B and 3B adapter training | Expected to be feasible with suitable settings | Use the common training recipe selected by the pilot. |
| Production 7B training | Expected to be feasible with QLoRA at moderate context | Start with microbatch 1, checkpointing, and memory-efficient attention. Test 8K, 16K, and 32K separately. |
| 7B BF16 LoRA at 32K context | Unverified and memory constrained | Do not treat this as a guaranteed configuration. Use measured optimizations or additional GPU access if needed. |
| Several agent evaluations with long histories at once | Concurrency may be limited | Start with one active agent task and increase only after measuring memory and throughput. |
| Optional 32B inference in BF16 entirely on GPU | Does not fit | Use separate hardware or omit the optional experiment. Quantized or CPU-offloaded inference is a different configuration. |

The 32B model requires roughly 64-65 GB for BF16 weights alone, calculated at two bytes per parameter, before inference buffers and the attention cache. [4] This is a definite memory mismatch. The 7B context limit is an implementation-dependent question rather than a proven impossibility.

## Proposed revisions to the plan

### Use one consistent adapter training method

Select a shared 4-bit QLoRA recipe if ordinary BF16 LoRA cannot support the intended configuration on the 4090. A reasonable pilot uses NF4 weights, BF16 computation where supported, LoRA rank 16, microbatch 1, gradient accumulation, gradient checkpointing, and memory-efficient attention. [3]

Once selected, apply the same quantization policy across all nine main training conditions. Record any justified model-specific changes. Keep inference precision consistent between each adapted model and its matching unmodified baseline.

### Select the context limit from evidence

Treat 32K as a target to test, not an established capability. Begin the mechanics check at 4K-8K, then profile representative 16K and 32K examples on 7B. Select the largest shared production limit that fits reliably and retains a useful range of trajectories.

Before adopting a shorter limit, measure retained task counts and length distributions for every data source. Use a documented policy that preserves the issue, complete tool interactions, and valid assistant targets. Record dropped or shortened examples and their token counts. If a common shorter limit excludes too much useful data or changes the intended research question, use A100 access for production training.

### Schedule training and inference sequentially

Run one production experiment at a time. Unload the training model before starting the inference server and unload the server before training. A single 4090 does not support the proposed two-GPU schedule of independent experiments on separate cards.

During evaluation, begin with one active agent task. Increase concurrency only if the model, histories, and attention cache fit with sufficient headroom. Serving configuration and concurrency affect memory use. [5] Execute repository tests and standalone grading on CPU resources where appropriate.

### Recalculate the runtime budget

The previous estimate for two A100 80 GB GPUs does not establish a runtime estimate for one 4090. Recalculate after measuring training throughput for each size and agent-task duration under the chosen concurrency. Include repeated epochs, repeated context in any derived examples, checkpoint operations, and reruns.

Estimate training time as processed tokens divided by measured processed tokens per second. Count total non-padding tokens separately from loss-bearing assistant tokens. Estimate evaluation time from task count, measured task duration under load, and measured concurrency. Do not assume a larger dataset needs more VRAM when sequence length and batch settings remain fixed; it primarily increases total runtime.

### Defer the optional 32B comparison

Preserve the twelve core conditions. Use published larger-model results as contextual comparisons, with their evaluation differences documented. Run the optional local 32B baseline only if suitable additional hardware is available. A quantized or offloaded 32B result must identify that configuration explicitly.

## Validation before production runs

Weitao leads the training and memory measurements. Qingyao leads inference concurrency and agent evaluation measurements. Both review the common configuration before production runs.

1. Record the actual GPU, usable VRAM, software versions, and other GPU memory consumers.
2. Run representative 7B forward and backward passes at 8K, 16K, and 32K. For each viable setting, complete several training steps and measure peak allocated and reserved memory and steady-state throughput.
3. Confirm correct assistant-only labels and adapter updates, then save and reload the checkpoint in a fresh process.
4. Run a small, disjoint development task set through the full agent and official grader. Measure task duration, peak memory, valid actions, and timeout rates at the proposed evaluation concurrency.
5. Audit data retained under the selected context policy, freeze the settings, and revise the training and evaluation schedule.

Move the affected production work to larger hardware if the required 7B context does not fit reliably, the shorter-context data audit is unacceptable, or measured runtime exceeds the available schedule. Additional GPU access is a conditional resource need; it is not required to begin the first two milestones.

## References

Project scope: [CS6158 Project Proposal](<../CS6158 Project Proposal.pdf>) and [RTX 4090 implementation plan](rtx_4090_implementation_plan.md).

[1] NVIDIA. [GeForce RTX 4090 specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/).

[2] Hugging Face Transformers. [GPU memory usage](https://huggingface.co/docs/transformers/model_memory_anatomy).

[3] Hugging Face PEFT. [Quantization and QLoRA training](https://huggingface.co/docs/peft/developer_guides/quantization).

[4] Qwen. [Qwen2.5 Coder 32B Instruct model card](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct).

[5] vLLM. [Optimization and tuning](https://docs.vllm.ai/en/latest/configuration/optimization/).
