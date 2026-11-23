"""Bound the FP32 loss workspace without changing the causal training objective."""
from functools import partial, wraps


def chunked_causal_loss(logits, labels, vocab_size, num_items_in_batch=None,
                        ignore_index=-100, shift_labels=None, chunk_tokens=128, **kwargs):
    import torch
    from torch.nn import functional as F
    from torch.utils.checkpoint import checkpoint

    if type(chunk_tokens) is not int or chunk_tokens < 1:
        raise ValueError("chunk_tokens must be a positive integer")
    if shift_labels is None:
        shift_labels = F.pad(labels, (0, 1), value=ignore_index)[..., 1:]
    scores = logits.reshape(-1, vocab_size)
    targets = shift_labels.reshape(-1).to(scores.device)
    if scores.shape[0] != targets.numel():
        raise ValueError("Logits and shifted labels have different token counts")

    def summed_loss(part, target):
        return F.cross_entropy(part.float(), target, ignore_index=ignore_index, reduction="sum")

    total = scores.new_zeros((), dtype=torch.float32)
    for start in range(0, targets.numel(), chunk_tokens):
        part = scores[start:start + chunk_tokens]
        target = targets[start:start + chunk_tokens]
        # Recompute each chunk during backward instead of retaining every chunk's
        # FP32 log-softmax. Merely looping over CE calls would keep that memory.
        if torch.is_grad_enabled() and scores.requires_grad:
            value = checkpoint(summed_loss, part, target, use_reentrant=False,
                               preserve_rng_state=False)
        else:
            value = summed_loss(part, target)
        total = total + value
    denominator = (targets != ignore_index).sum() if num_items_in_batch is None else num_items_in_batch
    if torch.is_tensor(denominator):
        denominator = denominator.to(total.device)
    return total / denominator


def enable_chunked_loss(model, chunk_tokens=128):
    """Enable only for labeled Qwen2 forwards; generation/reload stays unchanged."""
    from transformers import Qwen2ForCausalLM
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    if not isinstance(base, Qwen2ForCausalLM):
        raise ValueError("Memory-safe loss currently supports Qwen2ForCausalLM only")
    if type(chunk_tokens) is not int or chunk_tokens < 1:
        raise ValueError("chunk_tokens must be a positive integer")
    if hasattr(base, "_study_chunked_loss_tokens"):
        raise ValueError("Chunked loss is already enabled")
    original = base.forward

    @wraps(original)
    def forward(*args, **kwargs):
        output = original(*args, **kwargs)
        if kwargs.get("labels") is not None:
            if not hasattr(output, "logits"):
                raise ValueError("Memory-safe training requires return_dict=True")
            # Trainer needs only the loss. Returning vocabulary logits makes
            # Accelerate convert the entire BF16 tensor to FP32 unnecessarily.
            output = type(output)(**{key: value for key, value in output.items()
                                     if key != "logits"})
        return output

    base.loss_function = partial(chunked_causal_loss, chunk_tokens=chunk_tokens)
    base.forward = forward
    base._study_chunked_loss_tokens = chunk_tokens
