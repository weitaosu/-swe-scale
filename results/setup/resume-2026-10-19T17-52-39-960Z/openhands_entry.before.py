#!/usr/bin/env python
"""Narrow local-data adapter around the pinned SWE-Gym OpenHands evaluator.
No project directory is mounted into the agent container. This wrapper must be
validated against the pinned checkout and official grader before benchmark use.
"""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/"src"))
from swe_study.common import setup_environment
setup_environment()
parser=argparse.ArgumentParser()
parser.add_argument("--tasks",required=True);parser.add_argument("--output",required=True)
parser.add_argument("--max-steps",type=int,default=30);parser.add_argument("--timeout",type=int,default=1800)
args=parser.parse_args()
vendor=root/"vendor/SWE-Gym-OpenHands"
sys.path.insert(0,str(vendor));os.chdir(vendor)
os.environ.update(USE_HINT_TEXT="false",USE_INSTANCE_IMAGE="true",RUN_WITH_BROWSING="false")
import datasets
rows=[json.loads(line) for line in Path(args.tasks).read_text(encoding="utf-8").splitlines() if line.strip()]
if any(r.get("patch") or r.get("test_patch") or r.get("hints_text") for r in rows): raise ValueError("Agent task view contains gold or hint material")
original_load=datasets.load_dataset

def local_load(name,*pos,**kw):
    if name=="cs6158-local-development": return datasets.Dataset.from_list(rows)
    return original_load(name,*pos,**kw)
datasets.load_dataset=local_load
import evaluation.utils.shared as shared
original_run=shared.run_evaluation

def bounded_run(*pos,**kw):
    kw["timeout_seconds"]=args.timeout
    return original_run(*pos,**kw)
shared.run_evaluation=bounded_run
sys.argv=[str(vendor/"evaluation/swe_bench/run_infer.py"),"--agent-cls","CodeActAgent","--llm-config","llm.cs6158","--max-iterations",str(args.max_steps),"--eval-num-workers","1","--eval-n-limit",str(len(rows)),"--eval-note","cs6158-local","--eval-output-dir",args.output,"--dataset","cs6158-local-development","--split","dev"]
runpy.run_path(sys.argv[0],run_name="__main__")
