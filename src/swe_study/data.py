from __future__ import annotations
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import re
import unicodedata
from .common import ROOT, canonical_hash, local_path, read_json, write_json, write_jsonl


def normalize_text(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(text or ""))).strip()


def text_content(content):
    if content is None: return ""
    if isinstance(content,str): return content
    if isinstance(content,list):
        if any(x.get("type") not in {"text",None} for x in content if isinstance(x,dict)):
            raise ValueError("Non-text message requires an explicit conversion policy")
        return "\n".join(x.get("text","") if isinstance(x,dict) else str(x) for x in content)
    raise ValueError("Unsupported message content")


def canonical_messages(raw):
    if isinstance(raw,str): raw=json.loads(raw)
    if not isinstance(raw,list) or not raw: raise ValueError("Empty or malformed messages")
    messages=[]; pending={}
    for index,entry in enumerate(raw):
        role=entry.get("role")
        role="tool" if role=="function" else role
        if role not in {"system","user","assistant","tool"}: raise ValueError(f"Unsupported role {role}")
        message={"role":role,"content":text_content(entry.get("content"))}
        calls=entry.get("tool_calls") or []
        if isinstance(calls,str): calls=json.loads(calls)
        if entry.get("function_call"):
            if calls: raise ValueError("Both legacy and modern function calls present")
            calls=[{"id":f"legacy_{index}","type":"function","function":entry["function_call"]}]
        if calls:
            if pending: raise ValueError("Unresolved tool call before another assistant action")
            if role!="assistant": raise ValueError("Tool call attached to a non-assistant message")
            normalized=[]
            for ci,call in enumerate(calls):
                function=dict(call.get("function") or {})
                if not function.get("name"): raise ValueError("Missing tool name")
                arguments=function.get("arguments",{})
                if isinstance(arguments,str): json.loads(arguments)
                else: arguments=json.dumps(arguments,ensure_ascii=False)
                function["arguments"]=arguments
                call_id=call.get("id") or f"call_{index}_{ci}"
                if call_id in pending: raise ValueError("Duplicate unresolved tool call ID")
                pending[call_id]=function["name"]
                normalized.append({"id":call_id,"type":"function","function":function})
            message["tool_calls"]=normalized
        if role=="tool":
            call_id=entry.get("tool_call_id")
            if not call_id:
                candidates=[k for k,v in pending.items() if not entry.get("name") or v==entry["name"]]
                if len(candidates)!=1: raise ValueError("Ambiguous tool result identity")
                call_id=candidates[0]
            if call_id not in pending: raise ValueError("Tool result has no pending call")
            message["tool_call_id"]=call_id
            pending.pop(call_id)
        elif pending and role in {"user","assistant"} and not calls:
            raise ValueError("Unresolved tool call before the next conversation turn")
        messages.append(message)
    if pending: raise ValueError("Unresolved tool calls at trajectory end")
    if not any(m["role"]=="assistant" for m in messages): raise ValueError("No assistant target")
    return messages


def records(folder):
    folder=local_path(folder)
    paths=sorted(folder.rglob("*.parquet"))+sorted(folder.rglob("*.jsonl"))
    if not paths: raise FileNotFoundError(f"No local Parquet/JSONL dataset payloads in {folder}")
    for path in paths:
        if path.suffix==".parquet":
            import pyarrow.parquet as pq
            batches=pq.ParquetFile(path).iter_batches(batch_size=128)
            row_index=0
            for batch in batches:
                for row in batch.to_pylist():
                    yield row,{"file":path.relative_to(ROOT).as_posix(),"row":row_index}
                    row_index+=1
        else:
            for row_index,line in enumerate(path.read_text(encoding="utf-8").splitlines()):
                if line.strip(): yield json.loads(line),{"file":path.relative_to(ROOT).as_posix(),"row":row_index}


def clean_tool_schema(value):
    """Remove nullable struct-union placeholders from Parquet tool definitions."""
    if isinstance(value, dict):
        return {k:clean_tool_schema(v) for k,v in value.items() if v is not None or k in {"default","const"}}
    if isinstance(value, list): return [clean_tool_schema(v) for v in value]
    return value


def first_issue(messages):
    return next((normalize_text(text_content(m.get("content"))) for m in messages if m.get("role")=="user"),"")


def prepare_rows(sft_rows,sampled_rows,task_rows,verified_rows,dev_ids=(),seed=6158):
    tasks={r["instance_id"]:r for r in task_rows}
    denied_ids={r["instance_id"] for r in verified_rows}|set(dev_ids)
    denied_issues={normalize_text(r.get("problem_statement")) for r in verified_rows if r.get("problem_statement")}
    denied_patches={normalize_text(r.get("patch")) for r in verified_rows if len(normalize_text(r.get("patch")))>=80}
    samples=defaultdict(list)
    for row,origin in sampled_rows:
        if row.get("resolved") is not True: continue
        messages=row.get("messages",[])
        if isinstance(messages,str): messages=json.loads(messages)
        samples[first_issue(messages)].append((row,origin))
    accepted=[]; excluded=[]; seen=set()
    for sft,origin in sft_rows:
        try:
            messages=canonical_messages(sft.get("messages"))
            issue=first_issue(messages)
            if not issue: raise ValueError("Missing user issue context")
            candidates=samples.get(issue,[])
            candidate_ids={r["instance_id"] for r,_ in candidates}
            match_kind="exact_normalized_first_user_message"
            if not candidate_ids:
                candidate_ids={key for key,row in tasks.items() if len(normalize_text(row.get("problem_statement")))>=40 and normalize_text(row.get("problem_statement")) in issue}
                match_kind="unique_task_problem_contained_in_first_user_message"
            if len(candidate_ids)!=1: raise ValueError("Task provenance missing or ambiguous")
            instance_id=next(iter(candidate_ids))
            task=tasks.get(instance_id)
            if not task: raise ValueError("Task metadata missing")
            if instance_id in denied_ids: raise ValueError("Task overlaps benchmark or held-out development IDs")
            if normalize_text(task.get("problem_statement")) in denied_issues: raise ValueError("Issue fingerprint overlaps Verified")
            if normalize_text(task.get("patch")) in denied_patches: raise ValueError("Patch fingerprint overlaps Verified")
            # Bind tool definitions only if matching source records agree. Do not guess a run ID.
            tools=sft.get("tools")
            if isinstance(tools,str): tools=json.loads(tools)
            if not tools:
                possibilities={canonical_hash(clean_tool_schema(r.get("tools") or [])):clean_tool_schema(r.get("tools") or []) for r,_ in candidates if r["instance_id"]==instance_id}
                if len(possibilities)!=1: raise ValueError("Tool schema provenance missing or ambiguous")
                tools=next(iter(possibilities.values()))
            tools=clean_tool_schema(tools)
            if not tools: raise ValueError("No verified tool definitions")
            names={t.get("function",t).get("name") for t in tools}
            if not names or None in names: raise ValueError("Malformed tool definitions")
            for message in messages:
                for call in message.get("tool_calls",[]):
                    if call["function"]["name"] not in names: raise ValueError("Action is absent from source tool schema")
                for name in re.findall(r"<function=([A-Za-z0-9_]+)>",message["content"]):
                    if message["role"]=="assistant" and name not in names: raise ValueError("Text action is absent from source tool schema")
            digest=canonical_hash({"messages":messages,"tools":tools})
            if digest in seen: raise ValueError("Duplicate trajectory")
            seen.add(digest)
            accepted.append({"instance_id":instance_id,"repo":task.get("repo"),"trajectory_id":digest,"messages":messages,"tools":tools,"tool_prompt_mode":("embedded" if all(name in " ".join(m["content"] for m in messages if m["role"]=="system") for name in names if name) else "native"),
                "provenance":{"sft":origin,"match_kind":match_kind,"candidate_successful_raw_rows":[o for r,o in candidates if r["instance_id"]==instance_id],"schema_normalization":"remove null optional Parquet struct fields; preserve default/const nulls","success_basis":"published_successful_SFT_split","task_level_match_only":True}})
        except (ValueError,TypeError,KeyError,json.JSONDecodeError) as e:
            excluded.append({"source":origin,"reason":str(e)})
    unique_ids=sorted({r["instance_id"] for r in accepted})
    if len(unique_ids)<2: raise ValueError("Fewer than two traceable tasks remain; cannot create disjoint train/validation data")
    rng=random.Random(seed);rng.shuffle(unique_ids)
    val_count=max(1,round(len(unique_ids)*0.1))
    val_ids=set(unique_ids[:val_count])
    train=[r for r in accepted if r["instance_id"] not in val_ids]
    validation=[r for r in accepted if r["instance_id"] in val_ids]
    return train,validation,excluded


def prepare(run):
    sft=list(records("data/raw/swegym/sft"));sampled=list(records("data/raw/swegym/sampled"))
    task_rows=[r for r,_ in records("data/raw/swegym/tasks")]
    verified=[r for r,_ in records("data/raw/swebench/verified")]
    dev=read_json("manifests/dev_tasks.json")["instance_ids"]
    train,val,excluded=prepare_rows(sft,sampled,task_rows,verified,dev)
    write_jsonl("data/processed/swegym/train.jsonl",train)
    write_jsonl("data/processed/swegym/validation.jsonl",val)
    write_jsonl("data/metadata/provenance.jsonl",[{"instance_id":r["instance_id"],"trajectory_id":r["trajectory_id"],**r["provenance"]} for r in train+val])
    write_jsonl("data/metadata/excluded.jsonl",excluded)
    report={"source_trajectories":len(sft),"retained_trajectories":len(train)+len(val),"train_trajectories":len(train),"validation_trajectories":len(val),
      "train_tasks":len({r["instance_id"] for r in train}),"validation_tasks":len({r["instance_id"] for r in val}),"excluded":len(excluded),
      "exclusion_reasons":dict(Counter(r["reason"] for r in excluded)),"seed":6158,
      "limitations":["Task-level provenance matching does not identify an exact teacher run when several successful raw runs share the same issue.","Public identity/fingerprint screening cannot establish absence of benchmark exposure in pretrained weights."]}
    write_json(run.path/"data_audit.json",report);write_json("results/data_audit/latest.json",report)
    run.event("data_prepared",**report);print(json.dumps(report,indent=2))


def proportional_sample(rows,count,seed):
    groups=defaultdict(list)
    for row in rows: groups[row["repo"]].append(row["instance_id"])
    if len({x for ids in groups.values() for x in ids})!=len(rows): raise ValueError("Duplicate instance IDs")
    if not 0<count<=len(rows): raise ValueError("Invalid requested task count")
    quotas={repo:count*len(ids)/len(rows) for repo,ids in groups.items()}
    allocations={repo:math.floor(q) for repo,q in quotas.items()}
    order=sorted(groups,key=lambda repo:(-(quotas[repo]-allocations[repo]),repo))
    for repo in order[:count-sum(allocations.values())]: allocations[repo]+=1
    selected=[]
    for repo in sorted(groups):
        ids=sorted(groups[repo]);random.Random(f"{seed}:{repo}").shuffle(ids);selected+=ids[:allocations[repo]]
    random.Random(seed).shuffle(selected)
    return selected,allocations


def select_tasks(run):
    verified=[r for r,_ in records("data/raw/swebench/verified")]
    dev=[r for r,_ in records("data/raw/swebench/dev")]
    final,coverage=proportional_sample(verified,100,6158)
    denied={r["instance_id"] for r in verified}
    denied_issues={normalize_text(r.get("problem_statement")) for r in verified}
    candidates=[r for r in dev if r["instance_id"] not in denied and normalize_text(r.get("problem_statement")) not in denied_issues]
    development,_=proportional_sample(candidates,5,6158)
    for filename,data in [("eval_100.json",{"instance_ids":final,"repository_counts":coverage,"seed":6158}),("dev_tasks.json",{"instance_ids":development,"seed":6158,"source":"SWE-bench dev","disjoint_from_all_Verified":True})]:
        target=local_path(Path("manifests")/filename)
        if target.exists() and read_json(target)!=data: raise ValueError(f"Frozen task manifest differs; existing file preserved: {target}")
        write_json(target,data)
    write_jsonl("data/metadata/dev_task_budgets.jsonl",[{"instance_id":r["instance_id"],"problem_statement":r["problem_statement"]} for r in dev if r["instance_id"] in set(development)])
    run.event("tasks_selected",final_count=len(final),dev_count=len(development),coverage=coverage)
