import json
from types import SimpleNamespace
import pytest
from swe_study.assets import selected_files
from swe_study.common import ROOT,local_path,canonical_hash
from swe_study.data import canonical_messages,prepare_rows,proportional_sample
from swe_study.evaluation import public_instance,export_predictions
from swe_study.serving import parse_tool_output
from swe_study.tokenization import encode_assistant_only,SupervisedCollator
from swe_study.audit import Run,redact


class CharacterTokenizer:
    """Deterministic character offsets make boundary-mask assertions exact."""
    def apply_chat_template(self,messages,tokenize=False,add_generation_prompt=False,**kwargs):
        s="".join("<"+m["role"]+">"+m.get("content","")+"</end>" for m in messages)
        return s+("<assistant>" if add_generation_prompt else "")
    def __call__(self,text,**kwargs):
        return {"input_ids":[ord(x) for x in text],"offset_mapping":[(i,i+1) for i in range(len(text))]}


def test_only_assistant_and_end_marker_receive_labels():
    row={"messages":[{"role":"system","content":"PRIVATE_SYSTEM"},{"role":"user","content":"USER_ISSUE"},{"role":"assistant","content":"ACTION"},{"role":"user","content":"OBSERVATION"},{"role":"assistant","content":"FINAL"}]}
    encoded=encode_assistant_only(CharacterTokenizer(),row,4096)
    text="".join(chr(t) for t in encoded["labels"] if t!=-100)
    assert text=="ACTION</end>FINAL</end>"
    assert "PRIVATE_SYSTEM" not in text and "OBSERVATION" not in text


def test_overlength_is_rejected_without_truncation():
    with pytest.raises(ValueError,match="over_context"):
        encode_assistant_only(CharacterTokenizer(),{"messages":[{"role":"user","content":"x"*20},{"role":"assistant","content":"solution"}]},10)


def test_template_prefix_instability_is_rejected():
    class Broken(CharacterTokenizer):
        def apply_chat_template(self,messages,**kwargs):
            return super().apply_chat_template(messages,**kwargs)+(str(len(messages)))
    with pytest.raises(ValueError,match="prefix-stable"):
        encode_assistant_only(Broken(),{"messages":[{"role":"user","content":"question"},{"role":"assistant","content":"answer"}]},4096)


def test_collator_masks_padding_even_when_padding_equals_eos():
    batch=SupervisedCollator(5)([{"input_ids":[3,5],"attention_mask":[1,1],"labels":[-100,5]},{"input_ids":[5],"attention_mask":[1],"labels":[5]}])
    assert batch["labels"].tolist()==[[-100,5],[5,-100]]
    assert batch["attention_mask"].tolist()==[[1,1],[1,0]]


def test_unmatched_tool_result_rejected():
    with pytest.raises(ValueError,match="pending call"):
        canonical_messages([{"role":"user","content":"q"},{"role":"tool","content":"result","tool_call_id":"missing"},{"role":"assistant","content":"a"}])


def test_tool_argument_json_and_pairing_preserved():
    messages=canonical_messages([{"role":"user","content":"q"},{"role":"assistant","content":None,"tool_calls":[{"id":"c1","function":{"name":"bash","arguments":{"command":"echo hello"}}}]},{"role":"tool","tool_call_id":"c1","content":"hello"},{"role":"assistant","content":"done"}])
    assert json.loads(messages[1]["tool_calls"][0]["function"]["arguments"])["command"]=="echo hello"
    assert messages[2]["tool_call_id"]==messages[1]["tool_calls"][0]["id"]


def test_unresolved_tool_call_rejected():
    with pytest.raises(ValueError,match="Unresolved"):
        canonical_messages([{"role":"user","content":"q"},{"role":"assistant","tool_calls":[{"id":"c1","function":{"name":"bash","arguments":"{}"}}]}])


def corpus():
    sft=[];sampled=[];tasks=[]
    tools=[{"type":"function","function":{"name":"bash","description":"Execute shell commands","parameters":{"type":"object"}}}]
    for i in range(12):
        issue="Issue "+str(i)+": "+"A unique problem requiring a repository correction. "*2
        messages=[{"role":"system","content":"system"},{"role":"user","content":issue},{"role":"assistant","content":"fix "+str(i)}]
        sft.append(({"messages":messages},{"file":"synthetic_sft","row":i}))
        sampled.append(({"instance_id":f"repo__case-{i}","resolved":True,"messages":messages,"tools":tools},{"file":"synthetic_raw","row":i}))
        tasks.append({"instance_id":f"repo__case-{i}","problem_statement":issue,"repo":"owner/repo","patch":"patch "+str(i)})
    return sft,sampled,tasks


def test_overlap_removed_before_disjoint_task_split():
    sft,sampled,tasks=corpus()
    train,val,excluded=prepare_rows(sft,sampled,tasks,[tasks[0]],dev_ids=[tasks[1]["instance_id"]])
    ids={r["instance_id"] for r in train+val}
    assert tasks[0]["instance_id"] not in ids and tasks[1]["instance_id"] not in ids
    assert len(excluded)==2
    assert not {r["instance_id"] for r in train}&{r["instance_id"] for r in val}
    assert (train,val,excluded)==prepare_rows(sft,sampled,tasks,[tasks[0]],dev_ids=[tasks[1]["instance_id"]])


def test_duplicate_trajectory_removed_and_ambiguous_task_quarantined():
    sft,sampled,tasks=corpus();sft.append(sft[2]);sampled.append(({**sampled[3][0],"instance_id":"other__task-999"},{"file":"raw","row":100}))
    train,val,excluded=prepare_rows(sft,sampled,tasks,[])
    assert any("Duplicate" in r["reason"] for r in excluded)
    assert any("ambiguous" in r["reason"] for r in excluded)


def test_benchmark_selection_is_deterministic_and_proportional():
    rows=[{"repo":"large","instance_id":f"l-{i}"} for i in range(80)]+[{"repo":"small","instance_id":f"s-{i}"} for i in range(20)]
    selected,coverage=proportional_sample(rows,25,6158)
    assert coverage=={"large":20,"small":5}
    assert len(set(selected))==25
    assert (selected,coverage)==proportional_sample(list(reversed(rows)),25,6158)


def test_dataset_download_filters_only_the_intended_split():
    entries=[SimpleNamespace(rfilename=n) for n in ["README.md","data/dev-00000.parquet","data/test-00000.parquet"]]
    assert [x.rfilename for x in selected_files(entries,"dataset","dev")]==["README.md","data/dev-00000.parquet"]
    with pytest.raises(ValueError): selected_files(entries,"dataset","train")


def test_agent_view_contains_no_gold_or_hidden_tests():
    result=public_instance({"instance_id":"task","repo":"r","problem_statement":"issue","patch":"SECRET_FIX","test_patch":"SECRET_TEST","FAIL_TO_PASS":["secret_test"],"hints_text":"secret_hint"})
    serialized=json.dumps(result)
    assert "SECRET" not in serialized and "secret_test" not in serialized and "secret_hint" not in serialized
    assert result["problem_statement"]=="issue"


def test_native_and_xml_actions_parse_without_execution():
    content,calls=parse_tool_output('thinking <tool_call>{"name":"bash","arguments":{"command":"echo hello"}}</tool_call>')
    assert content=="thinking" and json.loads(calls[0]["function"]["arguments"])["command"]=="echo hello"
    _,calls=parse_tool_output('<function=edit><parameter=insert_line>3</parameter><parameter=new_str>hello</parameter></function>')
    assert json.loads(calls[0]["function"]["arguments"])=={"insert_line":3,"new_str":"hello"}


def test_missing_predictions_keep_fixed_denominator(tmp_path):
    source=tmp_path/"input.jsonl";out=tmp_path/"output.jsonl"
    source.write_text(json.dumps({"instance_id":"one","test_result":{"git_patch":"diff"}})+"\n")
    with Run("unit_export") as run:
        rows=export_predictions(run,source,out,"model",["one","two"])
    assert len(rows)==2 and rows[1]["model_patch"]==""


def test_duplicate_predictions_are_rejected(tmp_path):
    source=tmp_path/"duplicates.jsonl";out=tmp_path/"out.jsonl"
    source.write_text((json.dumps({"instance_id":"one","model_patch":""})+"\n")*2)
    with Run("unit_export") as run:
        with pytest.raises(ValueError,match="duplicate"): export_predictions(run,source,out,"m")


def test_output_path_cannot_escape_project():
    with pytest.raises(ValueError): local_path(ROOT/".."/"outside.json")


def test_audit_redacts_credentials_but_preserves_token_metrics():
    assert redact({"api_key":"secret","processed_tokens":100,"nested":{"authorization":"Bearer secret"}})=={"api_key":"[REDACTED]","processed_tokens":100,"nested":{"authorization":"[REDACTED]"}}


def test_nested_parquet_records_have_traceable_row_indices(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from swe_study.data import records
    folder=tmp_path/"parquet_source";folder.mkdir()
    pq.write_table(pa.Table.from_pylist([{"instance_id":"one","messages":[{"role":"user","content":"issue"}]},{"instance_id":"two","messages":[{"role":"assistant","content":"answer"}]}]),folder/"data.parquet")
    loaded=list(records(folder))
    assert [r[0]["instance_id"] for r in loaded]==["one","two"]
    assert [r[1]["row"] for r in loaded]==[0,1]
    assert loaded[0][0]["messages"][0]["role"]=="user"


def test_another_action_cannot_precede_a_pending_tool_result():
    with pytest.raises(ValueError,match="Unresolved"):
        canonical_messages([{"role":"user","content":"q"},{"role":"assistant","tool_calls":[{"id":"one","function":{"name":"bash","arguments":"{}"}}]},{"role":"assistant","tool_calls":[{"id":"two","function":{"name":"bash","arguments":"{}"}}]}])


def test_storage_guard_stops_before_free_floor(monkeypatch):
    from swe_study.storage import StorageGuard,StorageLimit
    import swe_study.storage as storage
    guard=StorageGuard();guard.baseline={"initial_free_bytes":100e9,"min_free_bytes":25e9,"max_incremental_bytes":70e9}
    monkeypatch.setattr(storage.shutil,"disk_usage",lambda _:SimpleNamespace(free=30e9))
    with pytest.raises(StorageLimit,match="free-space"): guard.check(6e9)


def test_storage_guard_stops_before_incremental_cap(monkeypatch):
    from swe_study.storage import StorageGuard,StorageLimit
    import swe_study.storage as storage
    guard=StorageGuard();guard.baseline={"initial_free_bytes":120e9,"min_free_bytes":25e9,"max_incremental_bytes":70e9}
    monkeypatch.setattr(storage.shutil,"disk_usage",lambda _:SimpleNamespace(free=50e9))
    with pytest.raises(StorageLimit,match="70 GB"): guard.check(1)


def test_task_token_budget_is_shared_across_requests_and_cannot_reset(tmp_path):
    from swe_study.serving import TaskBudget
    task_file=tmp_path/"tasks.jsonl";ledger=tmp_path/"ledger.json"
    task_file.write_text(json.dumps({"instance_id":"one","problem_statement":"Fix the broken widget computation"})+"\n")
    budget=TaskBudget(task_file,10,ledger)
    key=budget.identify([{"role":"user","content":"Task: Fix the broken widget computation"}])
    budget.record(key,7);assert budget.remaining(key)==3
    budget.record(key,3);assert budget.remaining(key)==0
    with pytest.raises(ValueError,match="exceed"): budget.record(key,1)
    with pytest.raises(ValueError,match="exactly one"): budget.identify([{"role":"user","content":"unrecognized issue"}])
    assert json.loads(ledger.read_text())["completion_tokens_by_task"]["one"]==10


def test_streamed_tool_calls_include_index_and_separate_finish_chunk():
    from swe_study.serving import stream_chunks
    response={"id":"chat-one","created":1,"model":"local","choices":[{"message":{"role":"assistant","content":None,"tool_calls":[{"id":"c1","type":"function","function":{"name":"bash","arguments":"{}"}}]},"finish_reason":"tool_calls"}],"usage":{"completion_tokens":3}}
    chunks=list(stream_chunks(response));first=json.loads(chunks[0][6:]);last=json.loads(chunks[1][6:])
    assert first["choices"][0]["delta"]["tool_calls"][0]["index"]==0
    assert first["choices"][0]["finish_reason"] is None
    assert last["choices"][0]["finish_reason"]=="tool_calls" and chunks[-1]=="data: [DONE]\n\n"


def test_docker_requires_physical_storage_baseline():
    from swe_study.docker_storage import require_storage
    if not (ROOT/"manifests/docker_storage.json").exists():
        with pytest.raises(RuntimeError,match="physical Docker storage baseline"): require_storage()


def test_failed_child_command_always_has_command_record():
    import sys
    with Run("unit_command") as run:
        with pytest.raises(RuntimeError,match="Command failed"):
            run.command([sys.executable,"-c","import sys; print('expected failure'); sys.exit(7)"])
        records=list(run.path.glob("command-*.json"))
        assert len(records)==1 and json.loads(records[0].read_text())["returncode"]==7
        assert (run.path/"source_snapshot.zip").exists()


@pytest.mark.parametrize("same_volume",[True,False])
def test_docker_volume_growth_is_combined_without_double_counting(tmp_path,monkeypatch,same_volume):
    import swe_study.storage as storage
    guard=storage.StorageGuard();guard.baseline={"initial_free_bytes":150e9,"min_free_bytes":25e9,"max_incremental_bytes":70e9}
    marker=tmp_path/"docker_storage.json";marker.write_text("{}")
    monkeypatch.setattr(storage,"local_path",lambda _:marker)
    monkeypatch.setattr(storage,"read_json",lambda _:{"physical_storage_path":str(tmp_path),"initial_physical_free_bytes":100e9,"same_volume_as_project":same_volume})
    monkeypatch.setattr(storage.shutil,"disk_usage",lambda p:SimpleNamespace(free=100e9 if p==ROOT else 70e9))
    if same_volume: assert guard.check()["conservative_volume_growth_bytes"]==50e9
    else:
        with pytest.raises(storage.StorageLimit,match="70 GB"): guard.check()


def test_continuation_adapters_match_training_configs():
    import yaml
    expected={"qlora_pilot.yaml":"checkpoints/pilot/qwen-1.5b","qlora_overfit.yaml":"checkpoints/pilot/qwen-1.5b-overfit","qlora_probe_3b.yaml":"checkpoints/pilot/qwen-3b-probe","qlora_probe_7b.yaml":"checkpoints/pilot/qwen-7b-probe"}
    for name,output in expected.items(): assert yaml.safe_load((ROOT/"configs"/name).read_text())["output"]==output


def test_grader_metadata_must_match_the_frozen_dataset():
    from swe_study.evaluation import merge_grading_metadata
    original={"instance_id":"one","repo":"owner/repo","base_commit":"abc","problem_statement":"Fix widget","patch":"diff gold","test_patch":"diff tests"}
    task={**original,"image":"official/image:tag","eval_script":"pytest","eval_type":"pass_and_fail","log_parser":"parse_log","FAIL_TO_PASS":["test_widget"],"PASS_TO_PASS":[]}
    merged=merge_grading_metadata(original,task);assert merged["image"]=="official/image:tag"
    with pytest.raises(ValueError,match="base_commit"): merge_grading_metadata(original,{**task,"base_commit":"wrong"})
    with pytest.raises(ValueError,match="patch"): merge_grading_metadata(original,{**task,"patch":"different fix"})


def test_grader_controls_require_correct_outcomes_and_complete_execution():
    from swe_study.evaluation import validate_grader_report
    gold={"total_instances":1,"submitted_ids":["one"],"resolved_ids":["one"],"error_instances":0,"incomplete_ids":[]}
    assert validate_grader_report(gold,{"one"},"gold")["passed"]
    blank={**gold,"resolved_ids":[],"empty_patch_ids":["one"]}
    assert validate_grader_report(blank,{"one"},"blank")["passed"]
    with pytest.raises(ValueError,match="Gold reference"): validate_grader_report(blank,{"one"},"gold")
    with pytest.raises(ValueError,match="errors"): validate_grader_report({**gold,"error_instances":1},{"one"},"gold")
