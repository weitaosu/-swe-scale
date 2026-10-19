#!/usr/bin/env python
"""Resume approved stages from a shell with network access. No Docker images are pulled."""
import argparse
from pathlib import Path
import sys
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/"src"))
from swe_study.common import setup_environment
setup_environment()
from swe_study.audit import Run
from swe_study.storage import StorageGuard
p=argparse.ArgumentParser();p.add_argument("--stage",required=True,choices=["assets","data","overfit","pilot","larger-probes"])
p.add_argument("--resume",help="Explicit Trainer checkpoint for an interrupted overfit/pilot run")
a=p.parse_args()
with Run("continue_setup",vars(a)) as run:
    guard=StorageGuard();guard.check()
    def study(*args,timeout=7200):
        return run.command([sys.executable,"scripts/study.py",*args],timeout=timeout,guard=guard)
    if a.stage=="assets":
        study("preflight",timeout=150);study("checkout")
        study("download","--kind","model","--only","1.5B",timeout=14400)
        study("download","--kind","dataset",timeout=14400)
    elif a.stage=="data":
        study("select-tasks");study("prepare");study("tokenize","--model","models/Qwen2.5-Coder-1.5B-Instruct","--max-length","4096")
    elif a.stage in {"overfit","pilot"}:
        config="configs/qlora_overfit.yaml" if a.stage=="overfit" else "configs/qlora_pilot.yaml"
        adapter="checkpoints/pilot/qwen-1.5b-overfit" if a.stage=="overfit" else "checkpoints/pilot/qwen-1.5b"
        command=["train","--config",config]+(["--resume",a.resume] if a.resume else [])
        study(*command,timeout=14400);study("validate-checkpoint","--adapter",adapter)
    else:
        if not (root/"checkpoints/pilot/qwen-1.5b/experiment.json").exists(): raise RuntimeError("Complete the real 1.5B pilot before larger model probes")
        for size in ["3B","7B"]:
            study("download","--kind","model","--only",size,timeout=14400)
            study("tokenize","--model",f"models/Qwen2.5-Coder-{size}-Instruct","--max-length","4096")
            study("train","--config",f"configs/qlora_probe_{size.lower()}.yaml",timeout=14400)
            study("validate-checkpoint","--adapter",f"checkpoints/pilot/qwen-{size.lower()}-probe")
