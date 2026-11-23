#!/usr/bin/env python3
"""Audit output-head loss equivalence, DDP, GPU memory and old-checkpoint resume."""
import argparse
from contextlib import nullcontext
from datetime import timedelta
import gc
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from swe_study.audit import Run, versions
from swe_study.common import setup_environment, write_json


def ddp_worker(rank, directory):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from swe_study.memory_head import enable_chunked_head_loss
    from swe_study.memory_mlp import enable_chunked_mlp
    from swe_study.training import assistant_target_inputs, adapter_digest
    torch.set_num_threads(1); torch.manual_seed(6158)
    torch.distributed.init_process_group("gloo", init_method="file://"+str(Path(directory)/"rendezvous"),
                                        rank=rank, world_size=2, timeout=timedelta(seconds=120))
    cfg = Qwen2Config(vocab_size=61, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                      num_attention_heads=2, num_key_value_heads=1, use_cache=False)
    cfg._attn_implementation = "eager"
    model = get_peft_model(Qwen2ForCausalLM(cfg), LoraConfig(r=4, lora_alpha=8, lora_dropout=.05,
                           target_modules=["q_proj","v_proj","gate_proj","up_proj","down_proj"],
                           task_type="CAUSAL_LM"))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    model.enable_input_require_grads()
    enable_chunked_head_loss(model, 2); enable_chunked_mlp(model, 2)
    wrapped = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=False)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
    before = adapter_digest(model)
    for update in range(2):
        optimizer.zero_grad()
        for microbatch in range(4):
            length = 7+rank+microbatch
            ids = torch.arange(3,3+length).unsqueeze(0)
            labels = ids.clone(); labels[:,:2] = -100
            inputs = assistant_target_inputs({"input_ids":ids,"attention_mask":torch.ones_like(ids),"labels":labels})
            with wrapped.no_sync() if microbatch<3 else nullcontext():
                (wrapped(**inputs).loss/4).backward()
        optimizer.step()
    after = adapter_digest(model)
    hashes = [None,None]; torch.distributed.all_gather_object(hashes, after)
    if before == after or len(set(hashes)) != 1:
        raise AssertionError("CPU DDP adapters did not update/synchronize")
    (Path(directory)/f"rank-{rank}.json").write_text(json.dumps({"passed":True,"adapter_sha256":after}))
    torch.distributed.destroy_process_group()


def gpu_measurement(optimized, length):
    import torch
    from swe_study.memory_loss import chunked_causal_loss
    from swe_study.memory_head import projected_causal_loss
    torch.manual_seed(6158)
    head = torch.nn.Linear(3584, 152064, bias=False, device="cuda", dtype=torch.float32)
    head.requires_grad_(False)
    x = torch.randn(1,length,3584,device="cuda",dtype=torch.float32,requires_grad=True)
    labels = torch.arange(length,device="cuda").remainder(152064).unsqueeze(0)
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    with torch.autocast("cuda",dtype=torch.bfloat16):
        if optimized:
            loss = projected_causal_loss(x,head,labels,chunk_tokens=128)
        else:
            loss = chunked_causal_loss(head(x),labels,152064,chunk_tokens=128)
    loss.backward(); torch.cuda.synchronize()
    sample = x.grad[0,:8,:32].detach().cpu().clone()
    return {"optimized":optimized,"target_positions":length,"vocab_size":152064,
            "hidden_size":3584,"head_parameter_dtype":str(head.weight.dtype),
            "loss":float(loss.detach()),"peak_allocated_bytes":torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes":torch.cuda.max_memory_reserved()}, sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu",action="store_true")
    parser.add_argument("--cpu-ddp",action="store_true")
    parser.add_argument("--resume-fixture",type=Path)
    args = parser.parse_args(); setup_environment()
    config = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    with Run("head_memory_validation",config) as run:
        run.command([sys.executable,"-m","pytest","tests/test_memory_head.py","-q","--capture=sys",
                     "-p","no:cacheprovider","--basetemp",str(ROOT/".cache/tmp"/("pytest-head-"+run.id))],timeout=300)
        report = {"passed":True,"synthetic_fixture":True,"versions":versions(),
                  "loss_all_gradients_generation_and_state_keys":"passed",
                  "limitation":"Synthetic models and isolated real-size head; no full 7B A100 feasibility claim"}
        if args.cpu_ddp:
            import torch.multiprocessing as mp
            directory = ROOT/".cache/tmp"/("head-ddp-"+run.id); directory.mkdir()
            mp.spawn(ddp_worker,args=(str(directory),),nprocs=2,join=True)
            report["two_process_cpu_ddp_with_accumulation_dropout_and_checkpointing"] = "passed"
        if args.gpu:
            import torch
            length = 14720 if torch.cuda.mem_get_info()[0] > 16*1024**3 else 4096
            results=[]; samples=[]
            for optimized in (False,True):
                result,sample = gpu_measurement(optimized,length)
                results.append(result); samples.append(sample)
                gc.collect(); torch.cuda.empty_cache()
            if results[1]["peak_allocated_bytes"] >= results[0]["peak_allocated_bytes"]*.75:
                raise AssertionError("Output head peak did not shrink materially")
            torch.testing.assert_close(samples[0],samples[1],atol=1e-7,rtol=.03)
            if abs(results[0]["loss"]-results[1]["loss"]) > .001:
                raise AssertionError("BF16 GPU head loss differs")
            report.update(gpu=torch.cuda.get_device_name(),real_size_frozen_qwen_7b_head=results)
        if args.resume_fixture:
            from validate_mlp_memory import resume_fixture
            os.environ["SWE_STUDY_HEAD_CHUNK_TOKENS"]="2"
            report["old_optimizer_checkpoint_resume_and_fresh_reload"]=resume_fixture(run,args.resume_fixture)
            project=ROOT/report["old_optimizer_checkpoint_resume_and_fresh_reload"]["project"]
            metadata=json.loads((project/"checkpoints/step-resume/experiment.json").read_text())
            if metadata.get("head_chunk_tokens")!=2 or metadata.get("loss_chunk_tokens")!=0:
                raise AssertionError("Head loss did not take priority over legacy CE chunking")
        write_json(run.path/"validation.json",report); print(json.dumps(report,indent=2))


if __name__ == "__main__": main()
