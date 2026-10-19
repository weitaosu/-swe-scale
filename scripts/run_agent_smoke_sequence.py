#!/usr/bin/env python3
"""Bounded first-frozen-task model/adapter integration checks; no study scores."""
from pathlib import Path
import sys,json,time,argparse
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from swe_study.common import setup_environment,write_json,read_json
setup_environment()
from swe_study.audit import Run
from swe_study.storage import StorageGuard
parser=argparse.ArgumentParser();parser.add_argument('--interfaces',required=True);args=parser.parse_args()
with Run('agent_smoke_sequence',{'scope':'single_first_frozen_task_smoke','models':['baseline-1.5b','swegym-pilot-1.5b','baseline-3b','baseline-7b'],'max_steps':5,'task_timeout_seconds':300,'replaces_five_task_validation':False}) as run:
    guard=StorageGuard();parent=(root/args.interfaces).resolve()
    if not parent.is_relative_to(root/'results/model_interface_sequence'):raise ValueError('Interface run outside expected stage')
    start=time.monotonic()

    while read_json(parent)['status']=='running':
        guard.check()
        if time.monotonic()-start>3600:raise TimeoutError('Interface checks unfinished')
        time.sleep(2)
    if read_json(parent)['status']!='passed':raise ValueError('Interface checks failed; preserve evidence before smoke runs')
    checks=[]
    for index,(size,adapter,label) in enumerate([('1.5B',None,'baseline-1.5b-smoke'),('1.5B','checkpoints/pilot/qwen-1.5b','swegym-pilot-1.5b-smoke'),('3B',None,'baseline-3b-smoke'),('7B',None,'baseline-7b-smoke')]):
        command=[root/'.envs/train/bin/python',root/'scripts/run_development_condition.py','--model','models/Qwen2.5-Coder-'+size+'-Instruct','--label',label,'--rollout','--smoke','--condition-id',run.id+'-'+str(index)]
        if adapter:command.extend(['--adapter',adapter])
        result=run.command(command,guard=guard,timeout=2400)
        paths=[Path(line[len('Run log: '):]) for line in Path(result['stdout_path']).read_text().splitlines() if line.startswith('Run log: '+str(root/'results/development_condition')+'/')]
        if len(paths)!=1:raise ValueError('Expected one audited condition')
        report=read_json(paths[0]/'development_validation.json')
        if report['scope']!='single_first_frozen_task_smoke':raise ValueError('Smoke result lacks correct scope')
        checks.append({'label':label,'evidence':str(paths[0]/'development_validation.json'),'report':report})
    write_json(run.path/'smoke_sequence.json',{'passed':True,'scope':'single_first_frozen_task_smoke','replaces_five_task_validation':False,'checks':checks})
