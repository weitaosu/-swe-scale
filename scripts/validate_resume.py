#!/usr/bin/env python3
"""Validate optimizer/step continuation with a two-step schedule extension."""
from pathlib import Path
import sys,shutil,json
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from swe_study.common import setup_environment,read_json,write_json,sha256_file
setup_environment()
from swe_study.audit import Run
from swe_study.storage import StorageGuard
with Run('resume_validation',{'scope':'25-step real pilot extended to 27 only to validate state restoration'}) as run:
    import yaml,torch
    from safetensors.torch import load_file
    guard=StorageGuard();guard.check(2*10**9)
    output=root/'checkpoints/pilot/qwen-1.5b';experiment=read_json(output/'experiment.json')
    if experiment['global_step']!=25 or experiment.get('synthetic_fixture'): raise ValueError('Expected completed real 25-step pilot')
    checkpoint=output/'checkpoint-25';before=run.path/'checkpoint-25-before'
    shutil.copytree(checkpoint,before)
    for name in ['experiment.json','adapter_model.safetensors','adapter_config.json','reload_reference.pt']:
        shutil.copy2(output/name,run.path/('pilot25_'+name))
    cfg=dict(experiment['config']);cfg.update(max_steps=27,save_steps=1,eval_steps=1,resume_validation_only=True,schedule_note='Original 25-step horizon extended by two; this checks restoration, not exact uninterrupted-schedule equivalence.')
    config=run.path/'resume.yaml';config.write_text(yaml.safe_dump(cfg,sort_keys=False))
    run.command([root/'.envs/train/bin/python',root/'scripts/study.py','train','--config',config,'--resume',before],guard=guard,timeout=2400)
    after=output/'checkpoint-27'
    if read_json(after/'trainer_state.json')['global_step']!=27: raise ValueError('Resume global step incorrect')
    left=load_file(str(before/'adapter_model.safetensors'));right=load_file(str(after/'adapter_model.safetensors'))
    changed=[k for k in left if not torch.equal(left[k],right[k])]
    if not changed: raise ValueError('Resume checkpoint adapter did not change')
    opt_before=torch.load(before/'optimizer.pt',map_location='cpu',weights_only=True)
    opt_after=torch.load(after/'optimizer.pt',map_location='cpu',weights_only=True)
    deltas={str(k):int(opt_after['state'][k]['step'])-int(v['step']) for k,v in opt_before['state'].items()}
    if not deltas or any(v!=2 for v in deltas.values()): raise ValueError('Optimizer steps did not continue by two')
    report={'passed':True,'before_step':25,'after_step':27,'changed_adapter_tensors':len(changed),'optimizer_entries':len(deltas),'optimizer_step_delta':2,'resume_source':str(before),'schedule_note':cfg['schedule_note']}
    write_json(run.path/'resume_validation.json',report)
    run.command([root/'.envs/train/bin/python',root/'scripts/study.py','validate-checkpoint','--adapter',output],guard=guard,timeout=600)
    print(json.dumps(report,indent=2))
