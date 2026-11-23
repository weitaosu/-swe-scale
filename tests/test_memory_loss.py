import copy
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from transformers.loss.loss_utils import ForCausalLMLoss
from swe_study.memory_loss import chunked_causal_loss, enable_chunked_loss
from swe_study.training import assistant_target_inputs


@pytest.mark.parametrize("selected", [False, True])
@pytest.mark.parametrize("denominator", [None, 19, torch.tensor(19)])
def test_loss_and_gradients_match_transformers(selected, denominator):
    torch.manual_seed(6158)
    left = torch.randn(1, 13, 61, requires_grad=True)
    labels = torch.randint(0, 61, (1, 13))
    labels[:, ::3] = -100
    extra = {}
    if selected:
        shifted = torch.nn.functional.pad(labels, (0, 1), value=-100)[:, 1:]
        positions = (shifted[0] != -100).nonzero().flatten()
        left = left[:, positions].detach().requires_grad_()
        extra["shift_labels"] = shifted[:, positions]
    right = left.detach().clone().requires_grad_()
    ref = ForCausalLMLoss(left, labels, 61, num_items_in_batch=denominator, **extra)
    got = chunked_causal_loss(right, labels, 61, num_items_in_batch=denominator,
                              chunk_tokens=3, **extra)
    torch.testing.assert_close(ref, got, atol=1e-6, rtol=1e-6)
    ref.backward(); got.backward()
    torch.testing.assert_close(left.grad, right.grad, atol=1e-7, rtol=1e-6)


def test_qwen_loss_gradients_evaluation_and_unlabeled_forward():
    torch.manual_seed(6158)
    cfg = Qwen2Config(vocab_size=61, hidden_size=16, intermediate_size=32,
                      num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                      attention_dropout=0.0, use_cache=False)
    cfg._attn_implementation = "eager"
    ref = Qwen2ForCausalLM(cfg).eval()
    model = copy.deepcopy(ref)
    enable_chunked_loss(model, 2)
    inputs = {"input_ids": torch.tensor([[3,4,5,6,7,8]]),
              "attention_mask": torch.ones(1,6,dtype=torch.long),
              "labels": torch.tensor([[-100,-100,5,6,-100,8]])}
    expected = ref(**inputs).loss
    output = model(**assistant_target_inputs(inputs))
    assert output.logits is None and "logits" not in output
    torch.testing.assert_close(expected, output.loss, atol=1e-6, rtol=1e-6)
    expected.backward(); output.loss.backward()
    for (_, left), (_, right) in zip(ref.named_parameters(), model.named_parameters()):
        assert (left.grad is None) == (right.grad is None)
        if left.grad is not None:
            torch.testing.assert_close(left.grad, right.grad, atol=1e-6, rtol=1e-5)
    with torch.no_grad():
        torch.testing.assert_close(model(**assistant_target_inputs(inputs)).loss, expected)
        bare = {key: value for key, value in inputs.items() if key != "labels"}
        torch.testing.assert_close(model(**bare).logits, ref(**bare).logits)


def test_bfloat16_logits_preserve_loss_and_gradient():
    torch.manual_seed(6158)
    left = torch.randn(1, 19, 61, dtype=torch.bfloat16, requires_grad=True)
    right = left.detach().clone().requires_grad_()
    labels = torch.randint(0, 61, (1, 19))
    ref = ForCausalLMLoss(left, labels, 61)
    got = chunked_causal_loss(right, labels, 61, chunk_tokens=4)
    ref.backward(); got.backward()
    torch.testing.assert_close(ref, got)
    torch.testing.assert_close(left.grad, right.grad, atol=0, rtol=0)
