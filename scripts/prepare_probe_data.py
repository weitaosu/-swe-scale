#!/usr/bin/env python3
"""Freeze long complete histories for size-specific 8K memory probes."""
from pathlib import Path
import sys,json
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from swe_study.common import setup_environment,read_json,read_jsonl,write_json,write_jsonl,sha256_file
setup_environment()
from swe_study.audit import Run
with Run('prepare_probe_data',{'context':8192,'selection':'longest retained complete histories; resources only'}) as run:
    manifest=read_json('data/tokenized/Qwen2.5-Coder-1.5B-Instruct-8192/manifest.json');selected={};report={}
    for split,limit in [('train',8),('validation',2)]:
        record=manifest['files'][split]
        if sha256_file(root/record['source'])!=record['source_sha256']: raise ValueError('Token audit source changed')
        lengths={r['trajectory_id']:len(r['input_ids']) for _,r in read_jsonl(record['output'])}
        eligible=[r for _,r in read_jsonl(record['source']) if r['trajectory_id'] in lengths]
        eligible.sort(key=lambda r:(-lengths[r['trajectory_id']],r['trajectory_id']))
        selected[split]=eligible[:limit]
        if len(selected[split])!=limit: raise ValueError('Not enough probe histories')
        target='data/processed/swegym/probe_8k/'+split+'.jsonl';write_jsonl(target,selected[split])
        report[split]={'output':target,'sha256':sha256_file(root/target),'examples':len(selected[split]),'lengths':[lengths[r['trajectory_id']] for r in selected[split]],'trajectory_ids':[r['trajectory_id'] for r in selected[split]]}
    if {r['instance_id'] for r in selected['train']}&{r['instance_id'] for r in selected['validation']}: raise ValueError('Task split overlap')
    write_json(run.path/'selection.json',report);write_json('data/processed/swegym/probe_8k/manifest.json',report);print(json.dumps(report,indent=2))
