from __future__ import annotations
import gc
import hashlib
import json
import math
import random
import time
from pathlib import Path
from .common import ROOT, canonical_hash, local_path, read_json, read_jsonl, sha256_file, write_json
from .storage import StorageGuard
from .tokenization import SupervisedCollator, encode_assistant_only


def adapter_digest(model):
    import torch
    h=hashlib.sha256()
    for name,value in sorted(model.named_parameters()):
        if value.requires_grad:
            h.update(name.encode());h.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def load_model(model_path,quantization="nf4",adapter=None):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
    model_path=local_path(model_path)
    if not (model_path/"config.json").exists(): raise FileNotFoundError(f"Model snapshot missing: {model_path}")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA GPU unavailable; this command will not silently use CPU offload")
    tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True,use_fast=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
    kwargs={"local_files_only":True,"device_map":{"":0},"dtype":torch.bfloat16,"attn_implementation":"sdpa"}
    if quantization=="nf4": kwargs["quantization_config"]=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
    elif quantization!="none": raise ValueError("Unsupported quantization")
    model=AutoModelForCausalLM.from_pretrained(model_path,**kwargs)
    if adapter:
        from peft import PeftModel
        model=PeftModel.from_pretrained(model,local_path(adapter),local_files_only=True)
    return model,tokenizer


def train(run,config_path,resume=None):
    import torch
    import yaml
    from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training
    from transformers import Trainer,TrainingArguments,TrainerCallback,set_seed
    from .audit import versions
    cfg=yaml.safe_load(local_path(config_path).read_text(encoding="utf-8"))
    guard=StorageGuard();guard.check(2*1024**3)
    output=local_path(cfg["output"])
    if output.exists() and any(output.iterdir()) and not resume:
        raise ValueError("Training output is nonempty. Resume explicitly or choose a fresh output directory.")
    set_seed(cfg.get("seed",6158))
    model,tokenizer=load_model(cfg["model"],cfg.get("quantization","nf4"))
    if cfg.get("quantization","nf4")=="nf4": model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True,gradient_checkpointing_kwargs={"use_reentrant":False})
    modules=cfg.get("target_modules",["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
    present={name.split(".")[-1] for name,_ in model.named_modules()}
    if not set(modules)<=present: raise ValueError(f"Missing target modules: {set(modules)-present}")
    model=get_peft_model(model,LoraConfig(r=cfg.get("lora_rank",16),lora_alpha=cfg.get("lora_alpha",32),lora_dropout=cfg.get("lora_dropout",0.05),target_modules=modules,bias="none",task_type="CAUSAL_LM"))
    model.config.use_cache=False
    unexpected=[n for n,p in model.named_parameters() if p.requires_grad and "lora_" not in n]
    if unexpected: raise ValueError(f"Unexpected trainable base parameters: {unexpected}")
    counts={"trainable":sum(p.numel() for p in model.parameters() if p.requires_grad),"total_stored_parameters":sum(p.numel() for p in model.parameters())}
    def examples(filename,limit):
        source=[row for _,row in read_jsonl(filename)]
        random.Random(cfg.get("seed",6158)).shuffle(source)
        rows=[];excluded=[]
        for row in source[:limit]:
            try: rows.append(encode_assistant_only(tokenizer,row,cfg["max_length"]))
            except ValueError as e: excluded.append({"instance_id":row.get("instance_id"),"reason":str(e)})
        if not rows: raise ValueError(f"No usable examples in {filename}")
        return rows,excluded
    train_rows,train_excluded=examples(cfg["train_file"],cfg.get("max_train_examples",64))
    eval_rows,eval_excluded=examples(cfg["validation_file"],cfg.get("max_eval_examples",16))
    train_ids={r.get("instance_id") for r in train_rows};eval_ids={r.get("instance_id") for r in eval_rows}
    if train_ids&eval_ids: raise ValueError("Training and validation task IDs overlap")
    token_counts={"processed":0,"targets":0,"training_step_seconds":0.0,"microbatches":0,"post_warmup_tokens":0,"post_warmup_seconds":0.0}
    gradient_checks=[]
    class Metrics(TrainerCallback):
        def on_log(self,args,state,control,logs=None,**kwargs):
            run.event("trainer_metrics",step=state.global_step,metrics=logs or {},tokens=dict(token_counts));guard.snapshot()
        def on_pre_optimizer_step(self,args,state,control,model=None,**kwargs):
            if len(gradient_checks)<3:
                norm=sum(float(p.grad.detach().float().norm().cpu()) for p in model.parameters() if p.requires_grad and p.grad is not None)
                gradient_checks.append(norm);run.event("adapter_gradient_check",step=state.global_step,sum_parameter_gradient_norms=norm)
                if not math.isfinite(norm) or not norm>0: raise ValueError("No nonzero adapter gradients")
    class TrackedTrainer(Trainer):
        def training_step(self,model,inputs,*args,**kwargs):
            guard.check();torch.cuda.synchronize();started=time.perf_counter()
            value=super().training_step(model,inputs,*args,**kwargs)
            if not torch.isfinite(value).all(): raise ValueError("Non-finite training loss")
            torch.cuda.synchronize();elapsed=time.perf_counter()-started
            processed=int(inputs["attention_mask"].sum());targets=int((inputs["labels"]!=-100).sum())
            token_counts["processed"]+=processed;token_counts["targets"]+=targets;token_counts["training_step_seconds"]+=elapsed;token_counts["microbatches"]+=1
            if token_counts["microbatches"]>2:
                token_counts["post_warmup_tokens"]+=processed;token_counts["post_warmup_seconds"]+=elapsed
            run.event("training_microbatch",processed_tokens=processed,target_tokens=targets,seconds=elapsed,loss=float(value.detach().cpu()))
            return value
    if resume:
        from peft import set_peft_model_state_dict
        from peft.utils.save_and_load import load_peft_weights
        checkpoint=local_path(resume)
        if not (checkpoint/"optimizer.pt").exists() or not (checkpoint/"trainer_state.json").exists():
            raise ValueError("Resume requires a complete Trainer checkpoint with optimizer and step state")
        previous=read_json(checkpoint/"adapter_config.json")
        if previous["r"]!=cfg.get("lora_rank",16) or previous["lora_alpha"]!=cfg.get("lora_alpha",32):
            raise ValueError("Resume adapter dimensions differ from this configuration")
        set_peft_model_state_dict(model,load_peft_weights(str(checkpoint),device="cpu"))
    before=adapter_digest(model)
    args=TrainingArguments(output_dir=str(output),max_steps=cfg.get("max_steps",25),per_device_train_batch_size=1,per_device_eval_batch_size=1,
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps",8),learning_rate=cfg.get("learning_rate",1e-4),warmup_ratio=cfg.get("warmup_ratio",0.03),
        max_grad_norm=1.0,bf16=True,gradient_checkpointing=True,gradient_checkpointing_kwargs={"use_reentrant":False},optim="adamw_torch",logging_steps=1,
        eval_strategy="steps",eval_steps=cfg.get("eval_steps",5),save_strategy="steps",save_steps=cfg.get("save_steps",5),save_total_limit=cfg.get("save_total_limit",2),
        load_best_model_at_end=True,metric_for_best_model="eval_loss",greater_is_better=False,report_to="none",seed=cfg.get("seed",6158),data_seed=cfg.get("seed",6158),dataloader_num_workers=0,remove_unused_columns=False,disable_tqdm=True)
    trainer=TrackedTrainer(model=model,args=args,train_dataset=train_rows,eval_dataset=eval_rows,data_collator=SupervisedCollator(tokenizer.pad_token_id),processing_class=tokenizer,callbacks=[Metrics()])
    run.event("training_setup",counts=counts,config=cfg,train_examples=len(train_rows),validation_examples=len(eval_rows),excluded={"train":train_excluded,"validation":eval_excluded},versions=versions())
    torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
    result=trainer.train(resume_from_checkpoint=str(local_path(resume)) if resume else None)
    elapsed=time.perf_counter()-started
    after=adapter_digest(model)
    if before==after: raise ValueError("Adapter parameters did not change")
    trainer.save_model(str(output));tokenizer.save_pretrained(output);trainer.save_state()
    model.eval();reference_ids=train_rows[0]["input_ids"][:min(128,len(train_rows[0]["input_ids"]))]
    with torch.no_grad(): logits=model(input_ids=torch.tensor([reference_ids],device="cuda"),attention_mask=torch.ones((1,len(reference_ids)),dtype=torch.long,device="cuda")).logits[0,-1].float().cpu()
    torch.save({"input_ids":reference_ids,"last_logits":logits},output/"reload_reference.pt")
    metadata={"config":cfg,"config_path":local_path(config_path).relative_to(ROOT).as_posix(),"versions":versions(),"parameters":counts,"adapter_hash_before":before,"adapter_hash_after":after,"adapter_updated":True,
      "gradient_checks":gradient_checks,"token_counts":token_counts,"wall_seconds_including_eval_and_checkpointing":elapsed,"gpu_count":1,"gpu_hours":elapsed/3600,
      "steady_state_tokens_per_second":token_counts["post_warmup_tokens"]/token_counts["post_warmup_seconds"] if token_counts["post_warmup_seconds"] else None,"peak_allocated_bytes":torch.cuda.max_memory_allocated(),"peak_reserved_bytes":torch.cuda.max_memory_reserved(),
      "train_source_sha256":sha256_file(local_path(cfg["train_file"])),"validation_source_sha256":sha256_file(local_path(cfg["validation_file"])),"train_examples":len(train_rows),"validation_examples":len(eval_rows),
      "best_checkpoint":trainer.state.best_model_checkpoint,"best_eval_loss":trainer.state.best_metric,"global_step":trainer.state.global_step,"resume_from_checkpoint":resume,"optimizer_state_entries":len(trainer.optimizer.state),"training_result":result.metrics,"log_history":trainer.state.log_history,"synthetic_fixture":cfg.get("synthetic_fixture",False)}
    write_json(output/"experiment.json",metadata);write_json(run.path/"metrics.json",metadata)
    print(json.dumps({"output":str(output),"adapter_updated":True,"gpu_hours":elapsed/3600,"tokens":token_counts},indent=2))
    del trainer,model;gc.collect();torch.cuda.empty_cache()
    return metadata


def validate_checkpoint(run,adapter):
    import torch
    adapter=local_path(adapter);metadata=read_json(adapter/"experiment.json");cfg=metadata["config"]
    model,_=load_model(cfg["model"],cfg.get("quantization","nf4"),adapter)
    reference=torch.load(adapter/"reload_reference.pt",map_location="cpu",weights_only=True)
    model.eval()
    with torch.no_grad(): actual=model(input_ids=torch.tensor([reference["input_ids"]],device="cuda"),attention_mask=torch.ones((1,len(reference["input_ids"])),dtype=torch.long,device="cuda")).logits[0,-1].float().cpu()
    maximum=float((actual-reference["last_logits"]).abs().max())
    passed=torch.allclose(actual,reference["last_logits"],atol=0.02,rtol=0.01)
    report={"adapter":adapter.relative_to(ROOT).as_posix(),"passed":passed,"max_absolute_logit_difference":maximum,"atol":0.02,"rtol":0.01,"synthetic_fixture":metadata.get("synthetic_fixture",False)}
    write_json(run.path/"reload_validation.json",report);print(json.dumps(report,indent=2))
    if not passed: raise ValueError("Checkpoint reload logits differ outside tolerance")
    return report
