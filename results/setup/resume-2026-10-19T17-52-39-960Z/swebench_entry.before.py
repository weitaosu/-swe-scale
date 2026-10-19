#!/usr/bin/env python
"""Pinned official grader, enriched with local task-repository test metadata.
Pulls only selected prebuilt images through the official harness. No local build
or remote service is requested. Validate controls before reporting any score.
"""
import argparse
import json
from pathlib import Path
import runpy
import subprocess
import sys
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/"src"))
from swe_study.common import setup_environment,read_json,write_json,local_path
from swe_study.evaluation import merge_grading_metadata
setup_environment()
parser=argparse.ArgumentParser();parser.add_argument("--dataset_name",required=True)
args,forward=parser.parse_known_args()
manifest=read_json("manifests/source_repositories.json")["repositories"]
for name in ["SWE-bench","swe-bench-tasks"]:
    entry=manifest[name];repo=local_path(entry["path"])
    head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=repo,text=True).strip()
    if entry.get("commit")!=head: raise ValueError(f"{name} checkout differs from its pinned manifest")
    subprocess.run(["git","diff","--quiet","HEAD"],cwd=repo,check=True)
sys.path.insert(0,str(root/"vendor/SWE-bench"))
from swebench.task.repo import load_task_repo
original=read_json(args.dataset_name);tasks=load_task_repo(root/"vendor/swe-bench-tasks",[row["instance_id"] for row in original])
by_id={row["instance_id"]:row for row in tasks}
combined=[merge_grading_metadata(row,by_id[row["instance_id"]]) for row in original]
enriched=local_path(Path(args.dataset_name).with_name("grading_dataset_enriched.json"))
write_json(enriched,combined)
sys.argv=["swebench.harness.run_evaluation","--dataset_name",str(enriched),*forward]
runpy.run_module("swebench.harness.run_evaluation",run_name="__main__")
