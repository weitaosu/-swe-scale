"""Reduce Qwen feed-forward activation peaks without shortening attention context."""
from functools import wraps


def _forward_in_chunks(module, original, chunk_tokens):
    @wraps(original)
    def forward(x):
        import torch
        from torch.utils.checkpoint import checkpoint
        if not module.training or not torch.is_grad_enabled() or x.shape[-2] <= chunk_tokens:
            return original(x)
        outputs = []
        for start in range(0, x.shape[-2], chunk_tokens):
            part = x[..., start:start + chunk_tokens, :]
            # MLP operations are token-independent. Recompute each part's
            # expanded activations in backward rather than retaining them all.
            # LoRA dropout must replay its original mask during recomputation.
            outputs.append(checkpoint(original, part, use_reentrant=False,
                                      preserve_rng_state=True))
        return torch.cat(outputs, dim=-2)
    return forward


def enable_chunked_mlp(model, chunk_tokens=1024):
    from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP
    if type(chunk_tokens) is not int or chunk_tokens < 1:
        raise ValueError("MLP chunk_tokens must be a positive integer")
    modules = [module for module in model.modules() if isinstance(module, Qwen2MLP)]
    if not modules:
        raise ValueError("No supported Qwen2 feed-forward modules found")
    if any(hasattr(module, "_study_mlp_chunk_tokens") for module in modules):
        raise ValueError("MLP chunking is already enabled")
    for module in modules:
        module.forward = _forward_in_chunks(module, module.forward, chunk_tokens)
        module._study_mlp_chunk_tokens = chunk_tokens
    return len(modules)
