#!/usr/bin/env python
"""Logged local training environment creation; no global package changes."""
import argparse
import os
from pathlib import Path
import sys
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/"src"))
from swe_study.common import setup_environment,local_path,write_json
setup_environment()
from swe_study.audit import Run
from swe_study.storage import StorageGuard
p=argparse.ArgumentParser();p.add_argument("--name",choices=["train","validation"],default="train")
p.add_argument("--system-site-packages",action="store_true");p.add_argument("--install",action="store_true")
a=p.parse_args()
if a.name=="validation" and not a.system_site_packages: p.error("validation reuse requires --system-site-packages")
with Run("bootstrap",vars(a)) as run:
    guard=StorageGuard();guard.check(12*10**9 if a.install else 256*1024**2)
    env=local_path(".envs/"+a.name);python=env/("Scripts/python.exe" if os.name=="nt" else "bin/python")
    if python.exists(): run.event("existing_environment_preserved",path=str(env))
    else:
        if env.exists() and any(env.iterdir()): raise ValueError("Existing environment directory is nonempty but has no Python for this platform; it was preserved")
        command=[sys.executable,"-m","venv","--copies",str(env)]
        if a.system_site_packages: command.append("--system-site-packages")
        run.command(command,timeout=120,guard=guard)
    if a.install: run.command([python,"-m","pip","install","-e",str(root)+"[ml,test]"],timeout=3600,guard=guard)
    frozen=run.command([python,"-m","pip","freeze","--all"],guard=guard)
    target=root/"manifests"/(a.name+"_packages.freeze.txt")
    target.write_text(Path(frozen["stdout_path"]).read_text(encoding="utf-8"),encoding="utf-8")
    write_json(run.path/"environment.json",{"path":str(env),"system_site_packages":a.system_site_packages,"installed":a.install,"freeze_file":str(target)})
    run.command([python,"scripts/study.py","preflight"],timeout=150,guard=guard)
