from pathlib import Path
import sys,time,json
root=Path(__file__).resolve().parents[3];sys.path.insert(0,str(root/'src'))
from swe_study.common import setup_environment
setup_environment()
from swe_study.audit import Run
from swe_study.storage import StorageGuard
with Run('development_validation_sequence',{'scope':'three baselines and 1.5B pilot adapter on frozen development tasks'}) as run:
    guard=StorageGuard();started=time.monotonic();parent=root/'results/post_pilot_validation/20261019T213122-d596967c/run.json'
    run.event('waiting_for_probes',parent=str(parent))
    while True:
        guard.check()
        try: status=json.loads(parent.read_text())['status']
        except json.JSONDecodeError: status='running'
        if status!='running': break
        if time.monotonic()-started>3600: raise TimeoutError('Larger probes did not finish')
        time.sleep(5)
    if status!='passed': raise ValueError('Larger probe or reload failed; development sequence not started')
    conditions=[('1.5B',None,'baseline-1.5b'),('1.5B','checkpoints/pilot/qwen-1.5b','swegym-pilot-1.5b'),('3B',None,'baseline-3b'),('7B',None,'baseline-7b')]
    for index,(size,adapter,label) in enumerate(conditions):
        command=[root/'.envs/train/bin/python',root/'scripts/run_development_condition.py','--model','models/Qwen2.5-Coder-'+size+'-Instruct','--label',label,'--rollout','--condition-id',run.id+'-'+str(index)]
        if adapter: command.extend(['--adapter',adapter])
        run.command(command,guard=guard,timeout=18000)
