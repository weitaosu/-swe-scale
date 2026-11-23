#!/usr/bin/env python3
"""Audit feed-forward loss/gradient equivalence, CPU DDP and optional NF4 GPU memory."""
import argparse
from contextlib import nullcontext
from datetime import timedelta
import gc
import json
import os
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from swe_study.audit import Run
from swe_study.common import setup_environment,write_json


def ddp_worker(rank, directory):
    import torch
    from peft import LoraConfig,get_peft_model
    from transformers import Qwen2Config,Qwen2ForCausalLM
    from swe_study.memory_loss import enable_chunked_loss
    from swe_study.memory_mlp import enable_chunked_mlp
    from swe_study.training import assistant_target_inputs,adapter_digest
    torch.set_num_threads(1);torch.manual_seed(6158)
    torch.distributed.init_process_group("gloo",init_method="file://"+str(Path(directory)/"rendezvous"),
                                        rank=rank,world_size=2,timeout=timedelta(seconds=120))
    cfg=Qwen2Config(vocab_size=61,hidden_size=16,intermediate_size=32,num_hidden_layers=2,
                    num_attention_heads=2,num_key_value_heads=1,use_cache=False)
    cfg._attn_implementation="eager"
    model=get_peft_model(Qwen2ForCausalLM(cfg),LoraConfig(r=4,lora_alpha=8,lora_dropout=0.05,
                           target_modules=["q_proj","v_proj","gate_proj","up_proj","down_proj"],
                           task_type="CAUSAL_LM"))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    model.enable_input_require_grads()
    enable_chunked_loss(model,2);enable_chunked_mlp(model,2)
    wrapped=torch.nn.parallel.DistributedDataParallel(model,find_unused_parameters=False)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=0.001)
    before=adapter_digest(model)
    for update in range(2):
        optimizer.zero_grad()
        for microbatch in range(4):
            length=7+rank+microbatch
            ids=torch.arange(3,3+length).unsqueeze(0)
            labels=ids.clone();labels[:,:2]=-100
            inputs=assistant_target_inputs({"input_ids":ids,"attention_mask":torch.ones_like(ids),"labels":labels})
            with wrapped.no_sync() if microbatch<3 else nullcontext():
                loss=wrapped(**inputs).loss/4
                loss.backward()
        optimizer.step()
    after=adapter_digest(model)
    hashes=[None,None];torch.distributed.all_gather_object(hashes,after)
    if before==after or len(set(hashes))!=1:
        raise AssertionError("CPU DDP adapters did not update/synchronize")
    (Path(directory)/f"rank-{rank}.json").write_text(json.dumps({"passed":True,"adapter_sha256":after}))
    torch.distributed.destroy_process_group()


def gpu_measurement(optimized, length):
    import torch
    import bitsandbytes as bnb
    from peft import LoraConfig,get_peft_model
    from transformers import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP
    from swe_study.memory_mlp import enable_chunked_mlp
    torch.manual_seed(6158)
    cfg=Qwen2Config(hidden_size=3584,intermediate_size=18944,hidden_act="silu")
    mlp=Qwen2MLP(cfg)
    for name in ("gate_proj","up_proj","down_proj"):
        layer=getattr(mlp,name)
        replacement=bnb.nn.Linear4bit(layer.in_features,layer.out_features,bias=False,
                        compute_dtype=torch.bfloat16,compress_statistics=True,quant_type="nf4")
        replacement.weight=bnb.nn.Params4bit(layer.weight.detach(),requires_grad=False,
                                              compress_statistics=True,quant_type="nf4")
        setattr(mlp,name,replacement)
    del layer,replacement
    mlp=mlp.cuda()
    for param in mlp.parameters():param.requires_grad=False
    model=get_peft_model(mlp,LoraConfig(r=16,lora_alpha=32,lora_dropout=0,
                                      target_modules=["gate_proj","up_proj","down_proj"])).train()
    if optimized:enable_chunked_mlp(model,1024)
    gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(777)
    x=torch.randn(1,length,3584,device="cuda",dtype=torch.float32,requires_grad=True)
    with torch.autocast("cuda",dtype=torch.bfloat16):
        output=model(x)
        loss=output.square().mean()
    loss.backward();torch.cuda.synchronize()
    return {"optimized":optimized,"sequence_length":length,"loss":float(loss.detach()),
            "peak_allocated_bytes":torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes":torch.cuda.max_memory_reserved()},x.grad[0,:4,:16].detach().cpu().clone()


def resume_fixture(run, source):
    import importlib.util
    import yaml
    source=source.resolve()
    if not source.is_relative_to(ROOT/'.cache/tmp'):
        raise ValueError("Only isolated synthetic fixtures are supported")
    cfg=yaml.safe_load((source/'configs/cloud/synthetic_step_resume.yaml').read_text())
    if not cfg.get('synthetic_fixture') or cfg['world_size']!=1:
        raise ValueError("Expected a one-GPU synthetic resume fixture")
    spec=importlib.util.spec_from_file_location('cloud_package',ROOT/'scripts/package_cloud.py')
    package=importlib.util.module_from_spec(spec);spec.loader.exec_module(package)
    project=ROOT/'.cache/tmp'/('mlp-resume-'+run.id)
    for name,data in package.bundle_files().items():
        target=project/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    shutil.copytree(source/cfg['model'],project/cfg['model'])
    for name in ('train_file','validation_file'):
        target=project/cfg[name];target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source/cfg[name],target)
    checkpoint=Path(cfg['output'])/'checkpoint-2'
    shutil.copytree(source/checkpoint,project/checkpoint)
    for name in ('configs/cloud/synthetic_step_resume.yaml',cfg['output']+'.plan.json'):
        target=project/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/name,target)
    run.command([sys.executable,'scripts/cloud.py','init','--storage-accounting','project'],cwd=project)
    os.environ['SWE_STUDY_LOSS_CHUNK_TOKENS']='2';os.environ['SWE_STUDY_MLP_CHUNK_TOKENS']='2'
    run.command([sys.executable,'scripts/cloud.py','train','--gpus','1','--config',
                 'configs/cloud/synthetic_step_resume.yaml','--resume','latest'],cwd=project,timeout=900)
    metadata=json.loads((project/cfg['output']/'experiment.json').read_text())
    if (metadata['global_step']!=4 or metadata['mlp_chunk_tokens']!=2
            or not metadata['resume_from_checkpoint'].endswith('checkpoint-2')):
        raise AssertionError('Synthetic optimizer resume failed')
    return {'passed':True,'project':str(project.relative_to(ROOT)),'from_step':2,'to_step':4}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',action='store_true')
    parser.add_argument('--cpu-ddp',action='store_true')
    parser.add_argument('--resume-fixture',type=Path)
    args=parser.parse_args();setup_environment()
    config={key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()}
    with Run('mlp_memory_validation',config) as run:
        run.command([sys.executable,'-m','pytest','tests/test_memory_mlp.py','-q','--capture=sys',
                     '-p','no:cacheprovider','--basetemp',str(ROOT/'.cache/tmp'/('pytest-mlp-'+run.id))],timeout=300)
        report={'passed':True,'synthetic_fixture':True,'loss_gradient_dropout_checks':'passed',
                'limitation':'MLP block and synthetic distributed/resume tests; no full A100 model feasibility claim'}
        if args.cpu_ddp:
            import torch.multiprocessing as mp
            directory=ROOT/'.cache/tmp'/('mlp-ddp-'+run.id);directory.mkdir()
            mp.spawn(ddp_worker,args=(str(directory),),nprocs=2,join=True)
            report['two_process_cpu_ddp_accumulation']='passed'
        if args.gpu:
            import torch
            free=torch.cuda.mem_get_info()[0]
            length=32768 if free>18*1024**3 else 8192
            results=[];samples=[]
            for optimized in (False,True):
                try:
                    result,sample=gpu_measurement(optimized,length)
                    results.append(result);samples.append(sample)
                except torch.OutOfMemoryError:
                    if optimized:raise
                    results.append({'optimized':False,'sequence_length':length,'out_of_memory':True})
                gc.collect();torch.cuda.empty_cache()
            if not results[0].get('out_of_memory'):
                if results[1]['peak_allocated_bytes']>=results[0]['peak_allocated_bytes']*0.8:
                    raise AssertionError('MLP activation peak did not shrink materially')
                torch.testing.assert_close(samples[0],samples[1],atol=1e-8,rtol=0.03)
                if abs(results[0]['loss']-results[1]['loss'])>1e-5:
                    raise AssertionError('NF4 MLP loss differs')
            report.update(gpu=torch.cuda.get_device_name(),nf4_rank16_qwen_7b_mlp=results)
        if args.resume_fixture:report['optimizer_resume_and_fresh_reload']=resume_fixture(run,args.resume_fixture)
        write_json(run.path/'validation.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
