#!/usr/bin/env python3
"""Sequential real-model API checks, without benchmark episodes."""
from pathlib import Path
import sys,argparse
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from swe_study.common import setup_environment,write_json
setup_environment()
from swe_study.audit import Run
from swe_study.storage import StorageGuard
parser=argparse.ArgumentParser();parser.add_argument('--reuse-adapter-check');args=parser.parse_args()
with Run('model_interface_sequence',{'scope':'1.5B pilot adapter, 3B baseline, 7B baseline HTTP validation; no benchmark score'}) as run:
    checks=[]
    conditions=[('1.5B','checkpoints/pilot/qwen-1.5b','swegym-pilot-1.5b'),('3B',None,'baseline-3b'),('7B',None,'baseline-7b')]
    if args.reuse_adapter_check:
        previous=(root/args.reuse_adapter_check).resolve()
        if not previous.is_relative_to(root/'results/development_condition'):raise ValueError('Existing API evidence outside the expected stage')
        report=__import__('json').loads(previous.read_text())
        if not report['passed'] or report['health'].get('adapter')!='checkpoints/pilot/qwen-1.5b':raise ValueError('Existing adapter check is invalid')
        checks.append({'label':'swegym-pilot-1.5b','evidence':str(previous),'reused_completed_check':True})
        conditions=conditions[1:]
    for index,(size,adapter,label) in enumerate(conditions):
        command=[root/'.envs/train/bin/python',root/'scripts/run_development_condition.py','--model','models/Qwen2.5-Coder-'+size+'-Instruct','--label',label,'--condition-id',run.id+'-'+str(index)]
        if adapter:command.extend(['--adapter',adapter])
        result=run.command(command,guard=StorageGuard(),timeout=1200)
        paths=[Path(line[len('Run log: '):]) for line in Path(result['stdout_path']).read_text().splitlines() if line.startswith('Run log: '+str(root/'results/development_condition')+'/')]
        if len(paths)!=1:raise ValueError('Expected one model condition')
        checks.append({'label':label,'evidence':str(paths[0]/'interface_validation.json')})
    write_json(run.path/'interface_sequence.json',{'passed':True,'checks':checks})
