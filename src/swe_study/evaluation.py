from __future__ import annotations
import json
import os
import subprocess
import sys
from pathlib import Path
from .common import ROOT, local_path, read_json, read_jsonl, write_json, write_jsonl
from .data import records
from .storage import StorageGuard

PUBLIC_FIELDS={"instance_id","repo","version","base_commit","environment_setup_commit","problem_statement","created_at"}


def development_task_ids(smoke=False):
    ids=read_json("manifests/dev_tasks.json")["instance_ids"]
    if not ids or len(ids)!=len(set(ids)): raise ValueError("Empty or duplicate frozen development task IDs")
    return ids[:1] if smoke else ids


def public_instance(row):
    result={key:row.get(key) for key in PUBLIC_FIELDS if key in row}
    # The fork expects these columns, but the agent gets no gold patches or evaluation test identities.
    result.update(patch="",test_patch="",hints_text="",PASS_TO_PASS="[]",FAIL_TO_PASS="[]")
    return result


def export_predictions(run,input_path,output_path,model_name,expected_ids=None):
    converted=[];seen=set()
    for _,row in read_jsonl(input_path):
        instance_id=row.get("instance_id")
        if not instance_id or instance_id in seen: raise ValueError("Missing or duplicate prediction instance ID")
        if expected_ids is not None and instance_id not in expected_ids: raise ValueError("Prediction is outside the frozen task manifest")
        patch=row.get("model_patch")
        if patch is None: patch=(row.get("test_result") or {}).get("git_patch","")
        if not isinstance(patch,str): raise ValueError("Patch must be a string")
        converted.append({"instance_id":instance_id,"model_name_or_path":model_name,"model_patch":patch});seen.add(instance_id)
    if expected_ids is not None:
        for missing in sorted(set(expected_ids)-seen):
            converted.append({"instance_id":missing,"model_name_or_path":model_name,"model_patch":""})
            run.event("missing_prediction_scored_empty",instance_id=missing)
    write_jsonl(output_path,converted);run.event("predictions_exported",count=len(converted),output=str(output_path))
    return converted


def merge_grading_metadata(original,task):
    for field in ["instance_id","repo","base_commit"]:
        if original.get(field)!=task.get(field): raise ValueError(f"Task repository and dataset disagree on {field}")
    for field in ["patch","test_patch","problem_statement"]:
        normalize=(lambda value:" ".join((value or "").split())) if field=="problem_statement" else (lambda value:(value or "").replace("\r\n","\n").strip())
        if normalize(original.get(field))!=normalize(task.get(field)):
            raise ValueError(f"Task repository and dataset disagree on {field}; benchmark identity must be reviewed")
    required=["image","eval_script","eval_type","log_parser","FAIL_TO_PASS","PASS_TO_PASS"]
    if any(field not in task for field in required): raise ValueError("Task repository lacks grading metadata")
    return {**original,**{field:task[field] for field in required},**({"image_assets":task["image_assets"]} if task.get("image_assets") else {})}


def validate_grader_report(report,wanted,control=None):
    if report.get("total_instances")!=len(wanted) or set(report.get("submitted_ids",[]))!=set(wanted):
        raise ValueError("Official grader denominator does not match the frozen task list")
    if report.get("error_instances",0) or report.get("incomplete_ids"):
        raise ValueError("Official grader reported execution errors or incomplete instances")
    resolved=set(report.get("resolved_ids",[]))
    if not resolved<=set(wanted): raise ValueError("Unexpected resolved task ID")
    if control=="gold" and resolved!=set(wanted): raise ValueError("Gold reference control did not resolve the selected task")
    if control=="blank" and (resolved or set(report.get("empty_patch_ids",[]))!=set(wanted)):
        raise ValueError("Blank-patch control behaved unexpectedly")
    return {"passed":True,"control":control,"total_instances":len(wanted),"resolved_instances":len(resolved),"resolved_ids":sorted(resolved)}


def grade(run,predictions=None,control=None,python=None,instance_id=None,smoke=False):
    if control not in {None,"gold","blank"}: raise ValueError("Unknown grader control")
    task_ids=development_task_ids()
    if smoke and (control or instance_id): raise ValueError("Smoke grading cannot select a control or another task")
    if instance_id and (not control or instance_id not in task_ids): raise ValueError("Control task must belong to the frozen development manifest")
    wanted={instance_id} if instance_id else set(task_ids[:1] if control or smoke else task_ids)
    run.event("grading_scope",scope="single_first_frozen_task_smoke" if smoke else "control" if control else "frozen_five_tasks",task_ids=sorted(wanted))
    dataset=[row for row,_ in records("data/raw/swebench/dev") if row["instance_id"] in wanted]
    if {r["instance_id"] for r in dataset}!=wanted: raise ValueError("Development data does not cover the frozen task list")
    dataset_path=run.path/"grading_dataset.json";write_json(dataset_path,dataset)
    if control:
        predictions=run.path/f"{control}_predictions.jsonl"
        write_jsonl(predictions,[{"instance_id":row["instance_id"],"model_name_or_path":f"grader-control-{control}","model_patch":row["patch"] if control=="gold" else ""} for row in dataset])
    if not predictions: raise ValueError("Provide predictions or a grader control")
    from .docker_storage import require_storage
    require_storage()
    import yaml
    overrides=read_json("manifests/image_overrides.json").get("overrides",{}) if (ROOT/"manifests/image_overrides.json").exists() else {}
    image_names=[overrides.get(task_id,{}).get("image") or yaml.safe_load((ROOT/"vendor/swe-bench-tasks/tasks"/task_id/"task.yaml").read_text())["image"] for task_id in sorted(wanted)]
    cache_probe=subprocess.run(["docker","image","inspect",*image_names],capture_output=True,text=True,timeout=30)
    cached=cache_probe.returncode==0
    reserve=2*10**9 if cached else 15*10**9
    write_json(run.path/"image_cache_check.json",{"images":image_names,"all_cached":cached,"reserve_bytes":reserve,"returncode":cache_probe.returncode,"stderr":cache_probe.stderr})
    run.event("grader_storage_reservation",all_images_cached=cached,reserve_bytes=reserve)
    guard=StorageGuard();guard.check(reserve)
    # This entry point remains available in the official harness; local full metadata is grader-only.
    command=[str(local_path(python)) if python else sys.executable,str(ROOT/"scripts/swebench_entry.py"),"--dataset_name",str(dataset_path),"--predictions_path",str(local_path(predictions)),"--max_workers","1","--run_id",run.id,"--instance_ids",*sorted(wanted)]
    run.command(command,cwd=run.path,timeout=7200,guard=guard)
    reports=[]
    for candidate in run.path.rglob("*.json"):
        try: payload=json.loads(candidate.read_text(encoding="utf-8"))
        except (ValueError,UnicodeDecodeError): continue
        if isinstance(payload,dict) and "resolved_ids" in payload and "total_instances" in payload: reports.append((candidate,payload))
    if len(reports)!=1: raise ValueError("Expected exactly one official grading summary")
    summary=validate_grader_report(reports[0][1],wanted,control)
    summary["scope"]="single_first_frozen_task_smoke" if smoke else "control" if control else "frozen_five_tasks"
    write_json(run.path/"grade_validation.json",{**summary,"official_report":str(reports[0][0])})
    run.event("official_grader_finished",control=control,predictions=str(predictions),summary=summary)


def rollout(run,model_name,python=None,smoke=False):
    if os.name=="nt": raise RuntimeError("Run the OpenHands Docker scaffold from Linux/WSL2. Native Windows data/training commands remain supported.")
    vendor=local_path("vendor/SWE-Gym-OpenHands")
    if not (vendor/"evaluation/swe_bench/run_infer.py").exists(): raise FileNotFoundError("Pinned OpenHands checkout has not been downloaded")
    import yaml
    cfg=yaml.safe_load(local_path("configs/agent.yaml").read_text(encoding="utf-8"))
    ids=development_task_ids(smoke)
    run.event("rollout_scope",scope="single_first_frozen_task_smoke" if smoke else "frozen_five_tasks",task_ids=ids)
    rows=[public_instance(r) for r,_ in records("data/raw/swebench/dev") if r["instance_id"] in set(ids)]
    if {r["instance_id"] for r in rows}!=set(ids): raise ValueError("Development metadata missing")
    public_path=run.path/"agent_tasks.jsonl";write_jsonl(public_path,rows)
    # Dedicated project configs are written only if no conflicting file is present.
    config='[llm.cs6158]\nmodel = '+json.dumps('openai/'+model_name)+'\nbase_url = '+json.dumps(cfg['base_url'])+'\napi_key = "local-only"\ntemperature = 0.0\nlog_completions = true\nmax_input_tokens = 6144\nmax_output_tokens = 2048\n'
    target=run.path/"config.toml"
    if target.exists() and target.read_text()!=config: raise ValueError("Existing OpenHands config differs and was preserved")
    target.write_text(config,encoding="utf-8")
    guard=StorageGuard();guard.check(2*10**9 if smoke else 15*10**9)
    if smoke:
        repairs=read_json("manifests/image_overrides.json")["overrides"]
        image=repairs[ids[0]].get("runtime_image")
        if not image or subprocess.run(["docker","image","inspect",image],capture_output=True,timeout=30).returncode: raise ValueError("Smoke mode requires a verified, fully cached runtime image")
    command=[str(local_path(python)) if python else sys.executable,str(ROOT/"scripts/openhands_entry.py"),"--tasks",str(public_path),"--output",str(run.path/"openhands"),"--max-steps",str(cfg["max_steps"]),"--timeout",str(cfg["task_timeout_seconds"]),"--run-id",run.id,"--llm-config-file",str(target)]
    if smoke:
        for flag,value in [("--max-steps",cfg["smoke_max_steps"]),("--timeout",cfg["smoke_task_timeout_seconds"])]:
            command[command.index(flag)+1]=str(value)
        run.event("smoke_limits",max_steps=cfg["smoke_max_steps"],task_timeout_seconds=cfg["smoke_task_timeout_seconds"])
    prefix='openhands-runtime-cs6158-'+run.id
    try:
        run.command(command,cwd=vendor,timeout=cfg["task_timeout_seconds"]*len(ids)+300,guard=guard)
    finally:
        remaining=run.command(['docker','ps','-a','--filter','name='+prefix,'--format','{{json .}}'],timeout=20)
        for line in Path(remaining['stdout_path']).read_text().splitlines():
            item=json.loads(line)
            if item['Names'].startswith(prefix): run.command(['docker','rm','--force',item['ID']],timeout=30)
    run.event("rollout_finished",task_ids=ids)
