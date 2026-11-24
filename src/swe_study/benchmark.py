"""Frozen development/final selection and resumable paired benchmark records."""
import json
from pathlib import Path
from .common import ROOT, canonical_hash, local_path, read_json, sha256_file, write_json, write_jsonl
from .data import records
from .evaluation import public_instance, validate_grader_report


def suite_rows(suite):
    if suite not in {"development","final"}: raise ValueError("Unknown benchmark suite")
    name="dev_tasks.json" if suite=="development" else "eval_100.json"
    ids=read_json("manifests/"+name)["instance_ids"]
    expected=5 if suite=="development" else 100
    if len(ids)!=expected or len(set(ids))!=expected: raise ValueError("Frozen suite denominator changed")
    source="dev" if suite=="development" else "verified"
    wanted=set(ids);by_id={}
    for row,_ in records("data/raw/swebench/"+source):
        if row["instance_id"] in wanted:
            if row["instance_id"] in by_id: raise ValueError("Duplicate task dataset row")
            by_id[row["instance_id"]]=row
    if set(by_id)!=wanted: raise ValueError("Frozen task metadata is incomplete")
    return [by_id[key] for key in ids]


def protocol_identity(adapter):
    import yaml
    metadata=read_json(local_path(adapter)/"experiment.json")
    cfg=yaml.safe_load(local_path("configs/agent.yaml").read_text())
    return {"adapter":str(local_path(adapter).relative_to(ROOT)),"model":metadata["config"]["model"],
            "adapter_weights_sha256":sha256_file(local_path(adapter)/"adapter_model.safetensors"),
            "agent_config":cfg,"source_repositories":read_json("manifests/source_repositories.json"),
            "frozen_development_ids":read_json("manifests/dev_tasks.json")["instance_ids"],
            "frozen_final_ids":read_json("manifests/eval_100.json")["instance_ids"],
            "runtime_overlay_sha256":sha256_file(local_path("src/swe_study/runtime.py")),
            "model_manifest_sha256":sha256_file(local_path("manifests/model_files.json")),
            "dataset_manifest_sha256":sha256_file(local_path("manifests/dataset_files.json")),
            "agent_environment_sha256":sha256_file(local_path("manifests/agent_packages.freeze.txt")),
            "grader_environment_sha256":sha256_file(local_path("manifests/eval_packages.freeze.txt")),
            "server_sha256":sha256_file(local_path("src/swe_study/serving.py")),
            "container_environment_sha256":sha256_file(local_path("cloud/nersc/container_env.sh")),
            "container_network_sha256":sha256_file(local_path("cloud/nersc/containers-network.toml")),
            "container_options_sha256":sha256_file(local_path("src/swe_study/container_options.py")),
            "container_cli_sha256":sha256_file(local_path("cloud/nersc/podman-bin/docker")),
            "agent_entry_sha256":sha256_file(local_path("scripts/openhands_entry.py")),
            "grader_entry_sha256":sha256_file(local_path("scripts/swebench_entry.py")),
            "attempt_policy":"one recorded agent episode per condition/task; no automatic retries",
            "paired_model_precision":"NF4 frozen weights, FP32 unquantized layers, BF16 autocast",
            "metrics":"resolved/frozen denominator; errors and incomplete IDs reported separately"}


def initialize_study(state,adapter,suite):
    state=local_path(state);identity=protocol_identity(adapter)
    path=state/"protocol.json"
    if path.exists() and read_json(path)!=identity: raise ValueError("Study protocol changed; existing records preserved")
    if not path.exists(): write_json(path,identity)
    rows=suite_rows(suite);folder=state/suite;folder.mkdir(parents=True,exist_ok=True)
    budgets=folder/"task_budgets.jsonl"
    content="".join(json.dumps({"instance_id":r["instance_id"],"problem_statement":r["problem_statement"]},ensure_ascii=False)+"\n" for r in rows)
    if budgets.exists() and budgets.read_text()!=content: raise ValueError("Frozen task budget file changed")
    if not budgets.exists(): budgets.write_text(content)
    if suite=="final":
        dev=study_summary(state,"development",identity["frozen_development_ids"])
        if not dev["complete"] or any(dev[condition]["agent_errors"] for condition in ("base","adapter")):
            raise ValueError("Complete the frozen-five paired development run and inspect its errors before final evaluation")
    return identity,rows,budgets


def record_path(state,suite,task_id,condition=None):
    return local_path(state)/suite/"tasks"/task_id/(condition+".json" if condition else "prepared.json")


def study_summary(state,suite,ids):
    result={"suite":suite,"denominator":len(ids),"complete":True}
    for condition in ("base","adapter"):
        resolved=[];incomplete=[];errors=[];completed=[]
        for key in ids:
            path=record_path(state,suite,key,condition)
            row=read_json(path) if path.exists() else {}
            if row.get("status")!="graded": incomplete.append(key)
            else:
                completed.append(key)
                if row["resolved"]: resolved.append(key)
                if row.get("agent_error"): errors.append({"instance_id":key,"error":row["agent_error"]})
        result[condition]={"completed":len(completed),"resolved":len(resolved),"resolved_ids":resolved,
                           "incomplete_ids":incomplete,"agent_errors":errors,
                           "resolved_percent_of_frozen_tasks":100*len(resolved)/len(ids)}
        result["complete"] &= not incomplete
    if result["complete"]:
        result["adapter_minus_base_percentage_points"]=result["adapter"]["resolved_percent_of_frozen_tasks"]-result["base"]["resolved_percent_of_frozen_tasks"]
    else:
        result["note"]="Partial run. Do not report these counts as a completed benchmark result."
    return result


def grade_one(run,row,prediction,control=None):
    """Use the pinned official task-repository grader, keeping gold grader-only."""
    dataset=run.path/"grading_dataset.json";write_json(dataset,[row])
    predictions=run.path/"predictions.jsonl";write_jsonl(predictions,[prediction])
    command=[str(ROOT/".envs/eval/bin/python"),str(ROOT/"scripts/swebench_entry.py"),
             "--dataset_name",str(dataset),"--predictions_path",str(predictions),
             "--max_workers","1","--run_id",run.id,"--instance_ids",row["instance_id"]]
    from .storage import StorageGuard
    run.command(command,cwd=run.path,guard=StorageGuard(),timeout=3600)
    reports=[]
    for candidate in run.path.rglob("*.json"):
        try: value=json.loads(candidate.read_text())
        except (ValueError,UnicodeDecodeError): continue
        if isinstance(value,dict) and "resolved_ids" in value and "total_instances" in value:
            reports.append((candidate,value))
    if len(reports)!=1: raise ValueError("Expected one official grader summary")
    summary=validate_grader_report(reports[0][1],{row["instance_id"]},control)
    write_json(run.path/"grade_validation.json",summary)
    return summary
