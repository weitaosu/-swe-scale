#!/usr/bin/env python
"""Logged local training environment creation; no global package changes."""
import argparse
import importlib.util
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
p.add_argument("--constraints",default="configs/constraints_train.txt")
a=p.parse_args()
if a.name=="validation" and not a.system_site_packages: p.error("validation reuse requires --system-site-packages")
with Run("bootstrap",vars(a)) as run:
    guard=StorageGuard();guard.check(12*10**9 if a.install else 256*1024**2)
    env=local_path(".envs/"+a.name);python=env/("Scripts/python.exe" if os.name=="nt" else "bin/python")
    has_ensurepip=importlib.util.find_spec("ensurepip") is not None
    if not has_ensurepip and importlib.util.find_spec("pip") is None:
        raise RuntimeError("This Python needs pip to bootstrap a local venv without ensurepip; no system packages were changed")
    if python.exists(): run.event("existing_environment_preserved",path=str(env))
    else:
        if env.exists() and any(env.iterdir()): raise ValueError("Existing environment directory is nonempty but has no Python for this platform; it was preserved")
        command=[sys.executable,"-m","venv","--copies",str(env)]
        if not has_ensurepip: command.append("--without-pip")
        if a.system_site_packages: command.append("--system-site-packages")
        run.command(command,timeout=120,guard=guard)
    if not has_ensurepip:
        # Base pip >=22.3 can provision pip into a clean project venv without apt/global installs.
        run.command([sys.executable,"-m","pip","--python",python,"install","--no-cache-dir","pip==24.0"],timeout=300,guard=guard)
    if a.install:
        run.command([python,"-m","pip","install","--no-cache-dir","--constraint",local_path(a.constraints),"-e",str(root)+"[ml,test]"],timeout=3600,guard=guard)
    frozen=run.command([python,"-m","pip","freeze","--all"],guard=guard)
    target=root/"manifests"/(a.name+"_packages.freeze.txt")
    target.write_text(Path(frozen["stdout_path"]).read_text(encoding="utf-8"),encoding="utf-8")
    write_json(run.path/"environment.json",{"path":str(env),"system_site_packages":a.system_site_packages,"installed":a.install,"freeze_file":str(target),"constraints":a.constraints if a.install else None,"ensurepip_available":has_ensurepip})
    run.command([python,"scripts/study.py","preflight"],timeout=150,guard=guard)
