"""Bound vocabulary projection AND loss memory for labeled Qwen2 forwards."""
from functools import wraps


def projected_causal_loss(hidden_states, head, labels, *, shift_labels=None,
                          num_items_in_batch=None, chunk_tokens=128, ignore_index=-100):
    """Exact token-summed CE, recomputing each small projection during backward."""
    import torch
    from torch.nn import functional as F
    from torch.utils.checkpoint import checkpoint

    if type(chunk_tokens) is not int or chunk_tokens < 1:
        raise ValueError("Head chunk_tokens must be a positive integer")
    if shift_labels is None:
        shift_labels = F.pad(labels, (0, 1), value=ignore_index)[..., 1:]
    hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
    targets = shift_labels.reshape(-1).to(hidden.device)
    if hidden.shape[0] != targets.numel():
        raise ValueError("Hidden states and shifted labels have different token counts")

    def project_and_sum(part, target):
        # Autocast and the existing head's parameter precision remain in force.
        # Keeping projection INSIDE checkpoint avoids a full logits gradient.
        return F.cross_entropy(head(part).float(), target,
                               ignore_index=ignore_index, reduction="sum")

    total = hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, targets.numel(), chunk_tokens):
        part = hidden[start:start + chunk_tokens]
        target = targets[start:start + chunk_tokens]
        if torch.is_grad_enabled():
            value = checkpoint(project_and_sum, part, target, use_reentrant=False,
                               preserve_rng_state=True)
        else:
            value = project_and_sum(part, target)
        total = total + value
    denominator = (targets != ignore_index).sum() if num_items_in_batch is None else num_items_in_batch
    if torch.is_tensor(denominator):
        denominator = denominator.to(total.device)
    return total / denominator


def enable_chunked_head_loss(model, chunk_tokens=128):
    """Labeled Qwen2 only. Generation, weights and checkpoint keys are unchanged."""
    from transformers import Qwen2ForCausalLM
    from transformers.modeling_outputs import CausalLMOutputWithPast
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    if not isinstance(base, Qwen2ForCausalLM):
        raise ValueError("Chunked head loss supports Qwen2ForCausalLM only")
    if type(chunk_tokens) is not int or chunk_tokens < 1:
        raise ValueError("Head chunk_tokens must be a positive integer")
    if hasattr(base, "_study_head_chunk_tokens") or hasattr(base, "_study_chunked_loss_tokens"):
        raise ValueError("A study loss patch is already enabled; select head loss instead")
    original = base.forward

    @wraps(original)
    def forward(input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, labels=None,
                use_cache=None, logits_to_keep=0, **kwargs):
        if labels is None:
            return original(input_ids=input_ids, attention_mask=attention_mask,
                            position_ids=position_ids, past_key_values=past_key_values,
                            inputs_embeds=inputs_embeds, labels=labels, use_cache=use_cache,
                            logits_to_keep=logits_to_keep, **kwargs)
        if kwargs.pop("return_dict", True) is False:
            raise ValueError("Chunked head training requires return_dict=True")
        shift_labels = kwargs.pop("shift_labels", None)
        num_items = kwargs.pop("num_items_in_batch", None)
        outputs = base.model(input_ids=input_ids, attention_mask=attention_mask,
                             position_ids=position_ids, past_key_values=past_key_values,
                             inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs)
        indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        loss = projected_causal_loss(outputs.last_hidden_state[:, indices, :], base.lm_head,
                                     labels, shift_labels=shift_labels,
                                     num_items_in_batch=num_items, chunk_tokens=chunk_tokens)
        # Trainer needs the loss only. In particular Accelerate must not upcast
        # a full vocabulary output when converting outputs to FP32.
        return CausalLMOutputWithPast(loss=loss, past_key_values=outputs.past_key_values,
                                     hidden_states=outputs.hidden_states, attentions=outputs.attentions)

    base.forward = forward
    base._study_head_chunk_tokens = chunk_tokens
