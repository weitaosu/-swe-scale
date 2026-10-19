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
import os
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
# Label only this grader run's containers for supervised cleanup.
import docker.models.containers
run_id=forward[forward.index('--run_id')+1] if '--run_id' in forward else None
original_create=docker.models.containers.ContainerCollection.create
def scoped_create(self,*pos,**kw):
    name=kw.get('name','')
    if run_id and name.startswith('sweb.eval.') and name.endswith('.'+run_id):
        labels=dict(kw.get('labels') or {})
        labels.update({'cs6158.project':'local-swe-study','cs6158.condition_id':os.environ.get('CS6158_CONDITION_ID',run_id)})
        kw['labels']=labels
    return original_create(self,*pos,**kw)
docker.models.containers.ContainerCollection.create=scoped_create
from swebench.task.repo import load_task_repo
original=read_json(args.dataset_name);tasks=load_task_repo(root/"vendor/swe-bench-tasks",[row["instance_id"] for row in original])
by_id={row["instance_id"]:row for row in tasks}
combined=[merge_grading_metadata(row,by_id[row["instance_id"]]) for row in original]
override_path=root/"manifests/image_overrides.json"
if override_path.exists():
    overrides=read_json(override_path)["overrides"]
    for row in combined:
        repair=overrides.get(row["instance_id"])
        if repair:
            if repair["base_image"]!=row["image"]: raise ValueError("Image repair base differs from pinned task metadata")
            image_id=subprocess.check_output(["docker","image","inspect","--format","{{.Id}}",repair["image"]],text=True).strip()
            if image_id!=repair["image_id"]: raise ValueError("Repaired local image identity changed")
            row["image"]=repair["image"]
    write_json(local_path(Path(args.dataset_name).with_name("applied_image_overrides.json")),overrides)
enriched=local_path(Path(args.dataset_name).with_name("grading_dataset_enriched.json"))
write_json(enriched,combined)
sys.argv=["swebench.harness.run_evaluation","--dataset_name",str(enriched),*forward]
runpy.run_module("swebench.harness.run_evaluation",run_name="__main__")
