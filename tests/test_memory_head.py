import copy
import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen2Config, Qwen2ForCausalLM
from transformers.loss.loss_utils import ForCausalLMLoss
from swe_study.memory_head import projected_causal_loss, enable_chunked_head_loss
from swe_study.memory_mlp import enable_chunked_mlp
from swe_study.training import assistant_target_inputs


@pytest.mark.parametrize("selected", [False, True])
@pytest.mark.parametrize("denominator", [None, 19, torch.tensor(19)])
@pytest.mark.parametrize("frozen_head", [False, True])
def test_projection_loss_and_all_gradients(selected, denominator, frozen_head):
    torch.manual_seed(6158)
    head = torch.nn.Linear(16, 61, bias=False)
    head.weight.requires_grad_(not frozen_head)
    other = copy.deepcopy(head)
    x = torch.randn(1, 13, 16, requires_grad=True)
    labels = torch.randint(0, 61, (1, 13)); labels[:, ::3] = -100
    extra = {}
    if selected:
        shifted = torch.nn.functional.pad(labels, (0, 1), value=-100)[:, 1:]
        positions = (shifted[0] != -100).nonzero().flatten()
        x = x[:, positions].detach().requires_grad_()
        extra["shift_labels"] = shifted[:, positions]
    y = x.detach().clone().requires_grad_()
    ref = ForCausalLMLoss(head(x), labels, 61, num_items_in_batch=denominator, **extra)
    got = projected_causal_loss(y, other, labels, num_items_in_batch=denominator,
                                chunk_tokens=3, **extra)
    ref.backward(); got.backward()
    torch.testing.assert_close(ref, got, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(x.grad, y.grad, atol=1e-7, rtol=1e-6)
    if not frozen_head:
        torch.testing.assert_close(head.weight.grad, other.weight.grad, atol=1e-7, rtol=1e-6)


def small_model():
    cfg = Qwen2Config(vocab_size=61, hidden_size=16, intermediate_size=32,
                      num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                      attention_dropout=0., use_cache=False, tie_word_embeddings=False)
    cfg._attn_implementation = "eager"
    return Qwen2ForCausalLM(cfg)


@pytest.mark.parametrize("selected", [False, True])
@pytest.mark.parametrize("lora", [False, True])
def test_qwen_training_evaluation_generation_and_keys(selected, lora):
    torch.manual_seed(6158)
    reference = small_model().train()
    if lora:
        reference = get_peft_model(reference, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.,
                                  target_modules=["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
                                  task_type="CAUSAL_LM"))
        for name, param in reference.named_parameters():
            if "lora_B" in name: torch.nn.init.normal_(param, std=.1)
    model = copy.deepcopy(reference)
    keys = list(model.state_dict())
    enable_chunked_head_loss(model, 2); enable_chunked_mlp(model, 2)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    if lora: model.enable_input_require_grads()
    assert list(model.state_dict()) == keys
    inputs = {"input_ids":torch.tensor([[3,4,5,6,7,8,9,10,11]]),
              "attention_mask":torch.ones(1,9,dtype=torch.long),
              "labels":torch.tensor([[-100,-100,5,6,-100,8,9,-100,11]])}
    expected = reference(**inputs).loss
    batch = assistant_target_inputs(inputs) if selected else inputs
    output = model(**batch)
    assert output.logits is None and "logits" not in output
    torch.testing.assert_close(expected, output.loss, atol=1e-6, rtol=1e-6)
    expected.backward(); output.loss.backward()
    for (lname, left), (rname, right) in zip(reference.named_parameters(), model.named_parameters()):
        assert lname == rname and (left.grad is None) == (right.grad is None)
        if left.grad is not None:
            torch.testing.assert_close(left.grad, right.grad, atol=2e-6, rtol=2e-5)
    model.eval(); reference.eval()
    bare = {k:v for k,v in inputs.items() if k != "labels"}
    with torch.no_grad():
        torch.testing.assert_close(model(**batch).loss, reference(**inputs).loss)
        torch.testing.assert_close(model(**bare).logits, reference(**bare).logits)
        assert torch.equal(model.generate(**bare, max_new_tokens=2, do_sample=False),
                           reference.generate(**bare, max_new_tokens=2, do_sample=False))


def test_projection_sizes_are_bounded_in_forward_and_backward():
    head = torch.nn.Linear(16, 61, bias=False)
    sizes = []
    hook = head.register_forward_pre_hook(lambda mod, args: sizes.append(args[0].shape[0]))
    hidden = torch.randn(1, 19, 16, requires_grad=True)
    projected_causal_loss(hidden, head, torch.arange(19).unsqueeze(0), chunk_tokens=4).backward()
    hook.remove()
    assert len(sizes) >= 10 and max(sizes) <= 4


def test_bfloat16_autocast_recomputation_preserves_fp32_head():
    torch.manual_seed(6158)
    head = torch.nn.Linear(16, 61, bias=False).requires_grad_(False)
    x = torch.randn(1, 19, 16, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    labels = torch.arange(19).unsqueeze(0)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        reference = ForCausalLMLoss(head(x), labels, 61)
        got = projected_causal_loss(y, head, labels, chunk_tokens=4)
    reference.backward(); got.backward()
    assert head.weight.dtype == torch.float32
    torch.testing.assert_close(reference, got, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(x.grad, y.grad, atol=1e-4, rtol=.03)


@pytest.mark.parametrize("tokens", [0,-1,True])
def test_invalid_chunk_size(tokens):
    with pytest.raises(ValueError, match="positive integer"):
        enable_chunked_head_loss(small_model(), tokens)


def test_second_loss_patch_rejected():
    from swe_study.memory_loss import enable_chunked_loss
    model = small_model(); enable_chunked_loss(model, 2)
    with pytest.raises(ValueError, match="already enabled"):
        enable_chunked_head_loss(model, 2)
