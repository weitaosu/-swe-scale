import copy
import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen2Config, Qwen2ForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP
from swe_study.memory_mlp import enable_chunked_mlp
from swe_study.memory_loss import enable_chunked_loss
from swe_study.training import assistant_target_inputs


def config():
    cfg=Qwen2Config(vocab_size=61,hidden_size=16,intermediate_size=32,
                    num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,
                    use_cache=False,attention_dropout=0.0)
    cfg._attn_implementation="eager"
    return cfg


def compare_gradients(left,right):
    for (lname,lparam),(rname,rparam) in zip(left.named_parameters(),right.named_parameters()):
        assert lname==rname and (lparam.grad is None)==(rparam.grad is None)
        if lparam.grad is not None:
            torch.testing.assert_close(lparam.grad,rparam.grad,atol=2e-6,rtol=2e-5)


def test_chunked_mlp_full_model_loss_gradients_and_state_keys():
    torch.manual_seed(6158)
    reference=Qwen2ForCausalLM(config()).train()
    model=copy.deepcopy(reference)
    keys=list(model.state_dict())
    assert enable_chunked_mlp(model,2)==2
    enable_chunked_loss(model,2)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    assert list(model.state_dict())==keys
    inputs={"input_ids":torch.tensor([[3,4,5,6,7,8,9,10,11]]),
            "attention_mask":torch.ones(1,9,dtype=torch.long),
            "labels":torch.tensor([[-100,-100,5,6,-100,8,9,-100,11]])}
    expected=reference(**inputs).loss
    got=model(**assistant_target_inputs(inputs)).loss
    torch.testing.assert_close(expected,got,atol=1e-6,rtol=1e-6)
    expected.backward();got.backward();compare_gradients(reference,model)
    model.eval();reference.eval()
    with torch.no_grad():
        bare={key:value for key,value in inputs.items() if key!="labels"}
        torch.testing.assert_close(model(**bare).logits,reference(**bare).logits)


def test_lora_dropout_replays_masks_during_mlp_recomputation():
    torch.manual_seed(6158)
    reference=get_peft_model(Qwen2MLP(config()),LoraConfig(r=4,lora_alpha=8,lora_dropout=0.25,
                              target_modules=["gate_proj","up_proj","down_proj"])).train()
    # Nonzero B exercises gradients through dropout and both LoRA factors.
    for name,param in reference.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(param, std=0.1)
    model=copy.deepcopy(reference)
    enable_chunked_mlp(model,2)
    x=torch.randn(1,9,16,requires_grad=True)
    y=x.detach().clone().requires_grad_()
    torch.manual_seed(99)
    expected=torch.cat([reference(x[:,start:start+2]) for start in range(0,9,2)],dim=1)
    expected.square().mean().backward()
    torch.manual_seed(99)
    got=model(y)
    got.square().mean().backward()
    torch.testing.assert_close(expected,got)
    torch.testing.assert_close(x.grad,y.grad)
    compare_gradients(reference,model)


@pytest.mark.parametrize("tokens", [0,-1,True])
def test_invalid_mlp_chunk_size_is_rejected(tokens):
    with pytest.raises(ValueError,match="positive integer"):
        enable_chunked_mlp(Qwen2MLP(config()),tokens)
