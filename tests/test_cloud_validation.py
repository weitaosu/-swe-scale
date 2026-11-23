from pathlib import Path
from types import SimpleNamespace
import copy
import json
import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen2Config, Qwen2ForCausalLM
from swe_study.memory_head import enable_chunked_head_loss
from swe_study.training import assistant_target_inputs
from swe_study.validation import evaluate_conditions, validation_metadata


def test_paired_trainer_matches_history_mean_and_preserves_weights(tmp_path):
    torch.manual_seed(6158)
    cfg=Qwen2Config(vocab_size=61,hidden_size=16,intermediate_size=32,num_hidden_layers=1,
                    num_attention_heads=2,num_key_value_heads=1,use_cache=False)
    cfg._attn_implementation="eager"
    model=get_peft_model(Qwen2ForCausalLM(cfg),LoraConfig(r=4,lora_alpha=8,lora_dropout=.1,
                          target_modules=["q_proj","v_proj"],task_type="CAUSAL_LM"))
    for name,param in model.named_parameters():
        if "lora_B" in name: torch.nn.init.normal_(param,std=.1)
    model.eval()
    rows=[{"input_ids":list(range(3,3+n)),"attention_mask":[1]*n,
           "labels":[-100,-100]+list(range(5,3+n))} for n in [5,9,13]]
    expected={}
    with torch.no_grad():
        for condition in ["base","adapter"]:
            from contextlib import nullcontext
            with model.disable_adapter() if condition=="base" else nullcontext():
                losses=[]
                for row in rows:
                    batch={k:torch.tensor([v]) for k,v in row.items()}
                    count=(batch["labels"]!=-100).sum()
                    losses.append(model(**assistant_target_inputs(batch),num_items_in_batch=count).loss)
                expected[condition]=torch.stack(losses).mean()
    before=copy.deepcopy(model.state_dict())
    enable_chunked_head_loss(model,2)
    events=[]
    run=SimpleNamespace(path=tmp_path,event=lambda *args,**kwargs:events.append((args,kwargs)))
    tokenizer=SimpleNamespace(pad_token_id=0,model_input_names=["input_ids","attention_mask"])
    results=evaluate_conditions(run,model,tokenizer,rows,use_cpu=True)
    for condition in ["base","adapter"]:
        assert results[condition][condition+"_loss"]==pytest.approx(float(expected[condition]),abs=1e-6)
    assert results["base"]["base_loss"]!=results["adapter"]["adapter_loss"]
    for name,value in model.state_dict().items(): torch.testing.assert_close(value,before[name],rtol=0,atol=0)
    assert all(p.grad is None for p in model.parameters())
    from peft.tuners.tuners_utils import BaseTunerLayer
    assert all(not module.disable_adapters for module in model.modules() if isinstance(module,BaseTunerLayer))
    assert len(events)==4


def test_changed_validation_source_is_rejected_before_gpu_load(tmp_path,monkeypatch):
    import swe_study.validation as module
    monkeypatch.setattr(module,"local_path",lambda path:tmp_path/Path(path))
    (tmp_path/"adapter").mkdir()
    (tmp_path/"validation.jsonl").write_text("{}\n")
    (tmp_path/"adapter/experiment.json").write_text(json.dumps({
        "config":{"validation_file":"validation.jsonl"},"validation_source_sha256":"wrong"}))
    with pytest.raises(ValueError,match="differs"):
        validation_metadata("adapter")
