#!/usr/bin/env python
"""Create a tiny RANDOM Qwen2 fixture to validate code, not model capability."""
from pathlib import Path
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from swe_study.common import ROOT,setup_environment,write_json,write_jsonl
setup_environment()
from swe_study.audit import Run
with Run("synthetic_fixture",{"purpose":"random tiny Qwen2 GPU integration fixture; not SWE-Gym or downloaded Qwen weights"}) as run:
    import torch
    from tokenizers import Tokenizer,models,pre_tokenizers
    from transformers import PreTrainedTokenizerFast,Qwen2Config,Qwen2ForCausalLM
    directory=ROOT/".cache/tmp/tiny-qwen-fixture"
    directory.mkdir(parents=True,exist_ok=True)
    vocabulary={"[UNK]":0,"<|im_start|>":1,"<|im_end|>":2,"<pad>":3,"system":4,"user":5,"assistant":6,"tool":7,"fix":8,"issue":9,"done":10,"hello":11,"world":12,"code":13,"test":14}
    tokenizer=Tokenizer(models.WordLevel(vocabulary,unk_token="[UNK]"));tokenizer.pre_tokenizer=pre_tokenizers.Whitespace()
    fast=PreTrainedTokenizerFast(tokenizer_object=tokenizer,unk_token="[UNK]",pad_token="<pad>",eos_token="<|im_end|>",additional_special_tokens=["<|im_start|>"])
    fast.chat_template="{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    fast.save_pretrained(directory)
    torch.manual_seed(6158)
    model=Qwen2ForCausalLM(Qwen2Config(vocab_size=64,hidden_size=64,intermediate_size=128,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=256,tie_word_embeddings=False,bos_token_id=1,eos_token_id=2,pad_token_id=3))
    model.save_pretrained(directory,safe_serialization=True)
    rows=[{"instance_id":f"synthetic-{i}","trajectory_id":f"synthetic-{i}","messages":[{"role":"system","content":"code test"},{"role":"user","content":"issue hello world"},{"role":"assistant","content":"fix done"}],"tools":[],"synthetic_fixture":True} for i in range(10)]
    write_jsonl(".cache/tmp/fixture_train.jsonl",rows[:8]);write_jsonl(".cache/tmp/fixture_validation.jsonl",rows[8:])
    import yaml
    cfg={"model":".cache/tmp/tiny-qwen-fixture","train_file":".cache/tmp/fixture_train.jsonl","validation_file":".cache/tmp/fixture_validation.jsonl","output":"checkpoints/pilot/synthetic-tiny-qwen","max_length":128,"max_train_examples":8,"max_eval_examples":2,"max_steps":3,"learning_rate":0.001,"warmup_ratio":0.0,"gradient_accumulation_steps":1,"lora_rank":4,"lora_alpha":8,"lora_dropout":0.0,"save_steps":1,"eval_steps":1,"save_total_limit":2,"seed":6158,"quantization":"nf4","synthetic_fixture":True}
    (ROOT/".cache/tmp/fixture_training.yaml").write_text(yaml.safe_dump(cfg),encoding="utf-8")
    run.event("synthetic_fixture_created",model_path=str(directory),config=cfg)
    print(json.dumps(cfg,indent=2))
