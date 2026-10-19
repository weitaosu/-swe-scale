#!/usr/bin/env python
"""Exercise HTTP serialization in process using ONLY the random tiny fixture."""
import json
import sys
from pathlib import Path
from unittest.mock import patch
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/"src"))
from swe_study.common import setup_environment,write_json,write_jsonl
setup_environment()
from swe_study.audit import Run
from swe_study.serving import serve
from fastapi.testclient import TestClient

task_file=root/".cache/tmp/fixture_server_tasks.jsonl"
write_jsonl(task_file,[{"instance_id":f"synthetic-task-{i}","problem_statement":f"task_issue_{i}"} for i in range(2)])
with Run("synthetic_server",{"synthetic_fixture":True,"transport":"in-process ASGI TestClient; no listening socket"}) as run:
    def exercise(app,**kwargs):
        with TestClient(app) as client:
            health=client.get("/health");assert health.status_code==200
            model=client.get("/v1/models").json()["data"][0]["id"]
            body={"model":model,"messages":[{"role":"user","content":"task_issue_0"}],"max_tokens":8,"temperature":0}
            assert client.post("/v1/chat/completions",json={**body,"model":"wrong"}).status_code==404
            assert client.post("/v1/chat/completions",json={**body,"n":2}).status_code==400
            assert client.post("/v1/chat/completions",json={**body,"messages":[{"role":"user","content":"unknown"}]}).status_code==400
            assert client.post("/v1/chat/completions",json={**body,"messages":[{"role":"user","content":"task_issue_1 "+"token_1 "*200}]}).status_code==413
            first=client.post("/v1/chat/completions",json=body);assert first.status_code==200,first.text
            total=first.json()["usage"]["completion_tokens"];assert 0<total<=2
            while total<2:
                response=client.post("/v1/chat/completions",json=body);assert response.status_code==200
                total+=response.json()["usage"]["completion_tokens"]
            assert client.post("/v1/chat/completions",json=body).status_code==429
            streamed=client.post("/v1/chat/completions",json={**body,"messages":[{"role":"user","content":"task_issue_1"}],"stream":True})
            assert streamed.status_code==200 and "data: [DONE]" in streamed.text
            report={"synthetic_fixture":True,"passed":True,"adapter":"checkpoints/pilot/synthetic-tiny-qwen","transport":"in-process ASGI; real GPU model generation","checks":["model list","health","wrong model 404","n limit 400","unknown task 400","context overflow 413","adapter generation","persistent per-task token ledger","exhausted task 429","SSE completion"]}
            write_json(run.path/"server_validation.json",report);run.event("server_validation_passed",**report);print(json.dumps(report,indent=2))
    with patch("uvicorn.run",side_effect=exercise):
        serve(run,".cache/tmp/tiny-qwen-fixture","checkpoints/pilot/synthetic-tiny-qwen",max_context=128,max_new_tokens=8,task_file=task_file,task_token_budget=2)
