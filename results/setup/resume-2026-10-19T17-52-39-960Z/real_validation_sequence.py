from pathlib import Path
import sys
root=Path(__file__).resolve().parents[3];sys.path.insert(0,str(root/'src'))
from swe_study.common import setup_environment
setup_environment()
from swe_study.audit import Run
from swe_study.storage import StorageGuard
with Run('real_validation_sequence',{'scope':'overfit reload, selected 8K 32-example pilot, pilot reload'}) as run:
    guard=StorageGuard();python=root/'.envs/train/bin/python';study=root/'scripts/study.py'
    for args in [['validate-checkpoint','--adapter','checkpoints/pilot/qwen-1.5b-overfit'],['train','--config','configs/qlora_pilot.yaml'],['validate-checkpoint','--adapter','checkpoints/pilot/qwen-1.5b']]:
        run.command([python,study,*args],guard=guard,timeout=2400)
