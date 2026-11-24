from contextlib import contextmanager, nullcontext
import json
from types import SimpleNamespace
import pytest
import torch
from fastapi.testclient import TestClient
from swe_study.benchmark import study_summary, initialize_study
from swe_study.serving import TaskBudget, serve


def test_failed_gold_control_surfaces_official_logs_without_changing_result(tmp_path,monkeypatch,capsys):
    import swe_study.benchmark as benchmark
    import swe_study.storage as storage
    official=tmp_path/'logs/evaluation/run/gold-control/example__repo-1'
    official.mkdir(parents=True)
    (official/'run_instance.log').write_text('Traceback: container startup failed: example diagnostic\n')
    report={'total_instances':1,'submitted_ids':['example__repo-1'],'resolved_ids':[],
            'error_instances':1,'error_ids':['example__repo-1'],'incomplete_ids':[]}
    (official.parent.parent/'results.json').write_text(json.dumps(report))
    commands=[];events=[]
    run=SimpleNamespace(path=tmp_path,id='run',command=lambda *args,**kwargs:commands.append(args),
                        event=lambda *args,**kwargs:events.append(args))
    monkeypatch.setattr(storage,'StorageGuard',lambda:None)
    with pytest.raises(ValueError,match='Official grader reported execution errors'):
        benchmark.grade_one(run,{'instance_id':'example__repo-1'},
                            {'instance_id':'example__repo-1','model_patch':'gold'},'gold')
    details=json.loads((tmp_path/'grader_failure.json').read_text())
    assert details['official_summary']==report
    assert 'example diagnostic' in capsys.readouterr().out
    assert len(commands)==1 and events==[('official_grader_failed',)]
    assert json.loads((official.parent.parent/'results.json').read_text())==report
    assert not (tmp_path/'grade_validation.json').exists()


@pytest.mark.parametrize('changed_patch',[False,True])
def test_preparation_loads_split_task_files_and_preserves_identity_check(tmp_path,monkeypatch,changed_patch):
    import importlib.util
    import yaml
    import swe_study.evaluation as evaluation
    import swe_study.docker_storage as storage
    # Use the pinned official loader with an actual split-file task layout.
    source_root=evaluation.ROOT
    monkeypatch.syspath_prepend(str(source_root/'vendor/SWE-bench'))
    key='example__repo-1';folder=tmp_path/'vendor/swe-bench-tasks/tasks'/key
    folder.mkdir(parents=True)
    original={'instance_id':key,'repo':'example/repo','base_commit':'abc',
              'problem_statement':'Fix widget','patch':'diff gold\n','test_patch':'diff tests\n'}
    metadata={k:original[k] for k in ('instance_id','repo','base_commit')}
    metadata.update(image='official/image:tag',eval_type='pass_and_fail',log_parser='parse_log')
    (folder/'task.yaml').write_text(yaml.safe_dump(metadata))
    for name,text in [('gold.patch','different fix\n' if changed_patch else original['patch']),
                      ('test.patch',original['test_patch']),('problem_statement.md',original['problem_statement']),
                      ('eval.sh','pytest\n'),('tests.json',json.dumps({'FAIL_TO_PASS':['test_widget'],'PASS_TO_PASS':[]}))]:
        (folder/name).write_text(text)
    monkeypatch.setattr(evaluation,'ROOT',tmp_path)
    spec=importlib.util.spec_from_file_location('cloud_benchmark_split_task_test',source_root/'scripts/cloud_benchmark.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module,'ROOT',tmp_path)
    monkeypatch.setattr(module,'record_path',lambda *args:tmp_path/'state/not_prepared.json')
    monkeypatch.setattr(storage,'require_storage',lambda:None)
    monkeypatch.setattr(module,'StorageGuard',lambda:None)
    class ReachedImagePreparation(Exception):pass
    def images_list(**kwargs):raise ReachedImagePreparation()
    closed=[]
    client=SimpleNamespace(images=SimpleNamespace(list=images_list),close=lambda:closed.append(True))
    import sys
    monkeypatch.setitem(sys.modules,'docker',SimpleNamespace(from_env=lambda **kwargs:client))
    if changed_patch:
        with pytest.raises(ValueError,match='disagree on patch'):
            module.prepare(None,original,'unused','development')
    else:
        with pytest.raises(ReachedImagePreparation):
            module.prepare(None,original,'unused','development')
    assert closed==[True]


def test_summary_keeps_missing_tasks_in_the_denominator(tmp_path,monkeypatch):
    import swe_study.benchmark as module
    monkeypatch.setattr(module,'record_path',lambda state,suite,key,condition=None:tmp_path/(key+'-'+str(condition)+'.json'))
    (tmp_path/'one-base.json').write_text(json.dumps({'status':'graded','resolved':True}))
    (tmp_path/'one-adapter.json').write_text(json.dumps({'status':'predicted'}))
    result=study_summary('unused','development',['one','two'])
    assert not result['complete'] and result['denominator']==2
    assert result['base']['resolved_percent_of_frozen_tasks']==50
    assert result['adapter']['incomplete_ids']==['one','two']
    assert 'adapter_minus_base_percentage_points' not in result


@pytest.mark.parametrize('error,valid',[('Agent got stuck in a loop',True),('API error: 413',False)])
def test_model_loop_is_a_recorded_failure_but_context_error_blocks_comparison(tmp_path,monkeypatch,error,valid):
    import swe_study.benchmark as module
    monkeypatch.setattr(module,'record_path',lambda state,suite,key,condition=None:tmp_path/(key+'-'+str(condition)+'.json'))
    (tmp_path/'one-base.json').write_text(json.dumps({'status':'graded','resolved':False,'agent_error':error}))
    (tmp_path/'one-adapter.json').write_text(json.dumps({'status':'graded','resolved':False}))
    result=module.study_summary('unused','development',['one'])
    assert result['complete'] and result['valid_comparison'] is valid
    assert result['base']['agent_errors'][0]['error']==error and result['base']['completed']==1
    assert bool(result['base']['model_failures']) is valid
    assert bool(result['base']['blocking_agent_errors']) is not valid
    assert ('adapter_minus_base_percentage_points' in result) is valid


def test_protocol_drift_and_final_without_development_are_rejected(tmp_path,monkeypatch):
    import swe_study.benchmark as module
    identity={'frozen_development_ids':['d'+str(i) for i in range(5)],
              'frozen_final_ids':['f'+str(i) for i in range(100)],'adapter_weights_sha256':'first'}
    monkeypatch.setattr(module,'protocol_identity',lambda adapter:dict(identity))
    monkeypatch.setattr(module,'suite_rows',lambda suite:[{'instance_id':key,'problem_statement':'Issue '+key}
                         for key in identity['frozen_development_ids' if suite=='development' else 'frozen_final_ids']])
    state=tmp_path/'state'
    initialize_study(state,'adapter','development')
    with pytest.raises(ValueError,match='frozen-five'):
        initialize_study(state,'adapter','final')
    identity['adapter_weights_sha256']='changed'
    with pytest.raises(ValueError,match='protocol changed'):
        initialize_study(state,'adapter','development')


def test_token_ledgers_survive_restart_and_reject_changed_task_set(tmp_path):
    tasks=tmp_path/'tasks.jsonl';tasks.write_text(json.dumps({'instance_id':'one','problem_statement':'Fix widget'})+'\n')
    ledger=tmp_path/'ledger.json'
    budget=TaskBudget(tasks,10,ledger);budget.record('one',7)
    assert TaskBudget(tasks,10,ledger).remaining('one')==3
    with pytest.raises(ValueError,match='different task set or limit'):
        TaskBudget(tasks,11,ledger)
    tasks.write_text(json.dumps({'instance_id':'two','problem_statement':'Fix another widget'})+'\n')
    with pytest.raises(ValueError,match='different task set or limit'):
        TaskBudget(tasks,10,ledger)


def test_paired_server_switches_adapter_and_has_separate_persistent_budgets(tmp_path,monkeypatch):
    import swe_study.training as training
    import swe_study.storage as storage
    import uvicorn
    class Batch(dict):
        def to(self,device): return self
    class Tokenizer:
        pad_token_id=0
        def apply_chat_template(self,*args,**kwargs): return 'prompt'
        def __call__(self,*args,**kwargs): return Batch(input_ids=torch.tensor([[3,4]]))
        def decode(self,tokens,**kwargs): return str(tokens.tolist())
    class Model:
        config=SimpleNamespace(use_cache=False)
        disabled=False
        def eval(self): return self
        @contextmanager
        def disable_adapter(self):
            self.disabled=True
            try: yield
            finally: self.disabled=False
        def generate(self,**kwargs):
            return torch.tensor([[3,4,5 if self.disabled else 6]])
    model=Model();monkeypatch.setattr(training,'load_model',lambda *args:(model,Tokenizer()))
    monkeypatch.setattr(storage,'StorageGuard',lambda:SimpleNamespace(check=lambda:None))
    for name in ['reset_peak_memory_stats','synchronize']:
        monkeypatch.setattr(torch.cuda,name,lambda:None)
    for name in ['max_memory_allocated','max_memory_reserved']:
        monkeypatch.setattr(torch.cuda,name,lambda:0)
    monkeypatch.setattr(torch,'autocast',lambda *args,**kwargs:nullcontext())
    apps=[];monkeypatch.setattr(uvicorn,'run',lambda app,**kwargs:apps.append(app))
    adapter=tmp_path/'adapter';adapter.mkdir();(adapter/'adapter_model.safetensors').write_bytes(b'fixture')
    tasks=tmp_path/'tasks.jsonl';tasks.write_text(json.dumps({'instance_id':'one','problem_statement':'Fix widget'})+'\n')
    run=SimpleNamespace(path=tmp_path,event=lambda *args,**kwargs:None)
    serve(run,'models/Qwen2.5-Coder-3B-Instruct',str(adapter),max_context=32,max_new_tokens=4,
          task_file=tasks,task_token_budget=8,paired=True,budget_ledger_dir=tmp_path/'ledgers')
    with TestClient(apps[0]) as client:
        aliases=client.get('/health').json()['models']
        body={'messages':[{'role':'user','content':'Fix widget'}],'max_tokens':4,'temperature':0}
        assert client.post('/v1/chat/completions',json={**body,'model':aliases[0]}).json()['choices'][0]['message']['content']=='[5]'
        assert client.post('/v1/chat/completions',json={**body,'model':aliases[1]}).json()['choices'][0]['message']['content']=='[6]'
        assert not model.disabled
        assert client.post('/v1/chat/completions',json={**body,'model':aliases[0],'temperature':.5}).status_code==400
        assert client.post('/v1/chat/completions',json={**body,'model':'wrong'}).status_code==404
    for condition in ['base','adapter']:
        assert json.loads((tmp_path/'ledgers'/(condition+'_task_tokens.json')).read_text())['completion_tokens_by_task']['one']==1


@pytest.mark.parametrize('policy,status',[('reject',413),('history-window-v1',200)])
def test_context_policy_uses_exact_prompt_budget_without_resetting_paired_ledgers(tmp_path,monkeypatch,policy,status):
    import swe_study.training as training
    import swe_study.storage as storage
    import uvicorn
    class Batch(dict):
        def to(self,device):return self
    class Tokenizer:
        pad_token_id=0
        def encode(self,text,**kwargs):return list(text.encode())
        def decode(self,ids,**kwargs):return bytes(ids.tolist() if hasattr(ids,'tolist') else ids).decode(errors='replace')
        def apply_chat_template(self,rows,tools=None,**kwargs):return json.dumps({'messages':rows,'tools':tools})
        def __call__(self,prompt,**kwargs):return Batch(input_ids=torch.tensor([self.encode(prompt)]))
    lengths=[]
    class Model:
        config=SimpleNamespace(use_cache=False)
        def eval(self):return self
        def disable_adapter(self):return nullcontext()
        def generate(self,input_ids,**kwargs):
            lengths.append(len(input_ids[0]));assert len(input_ids[0])+kwargs['max_new_tokens']<=1024
            return torch.cat([input_ids,torch.tensor([[120]])],dim=1)
    monkeypatch.setattr(training,'load_model',lambda *a:(Model(),Tokenizer()))
    monkeypatch.setattr(storage,'StorageGuard',lambda:SimpleNamespace(check=lambda:None))
    for name in ('reset_peak_memory_stats','synchronize'):monkeypatch.setattr(torch.cuda,name,lambda:None)
    for name in ('max_memory_allocated','max_memory_reserved'):monkeypatch.setattr(torch.cuda,name,lambda:0)
    monkeypatch.setattr(torch,'autocast',lambda *a,**k:nullcontext())
    apps=[];events=[];monkeypatch.setattr(uvicorn,'run',lambda app,**kw:apps.append(app))
    adapter=tmp_path/'adapter';adapter.mkdir();(adapter/'adapter_model.safetensors').write_bytes(b'fixture')
    tasks=tmp_path/'tasks.jsonl';tasks.write_text(json.dumps({'instance_id':'one','problem_statement':'Fix widget'})+'\n')
    run=SimpleNamespace(path=tmp_path,event=lambda name,**kw:events.append((name,kw)))
    serve(run,'models/Qwen2.5-Coder-3B-Instruct',str(adapter),max_context=1024,max_new_tokens=16,
          task_file=tasks,paired=True,budget_ledger_dir=tmp_path/'ledgers',context_policy=policy)
    messages=[{'role':'system','content':'instructions'},{'role':'user','content':'Fix widget'},
              {'role':'assistant','content':'inspect file'},{'role':'user','content':'BEGIN '+('x'*4000)+' END'}]
    with TestClient(apps[0]) as client:
        health=client.get('/health').json();assert health['context_policy']==policy
        for alias in health['models']:
            response=client.post('/v1/chat/completions',json={'model':alias,'messages':messages,'temperature':0,'max_tokens':16})
            assert response.status_code==status
    if status==200:
        assert len(lengths)==2
        windows=[record for name,record in events if name=='context_window']
        assert len(windows)==2 and all(w['changed'] and w['effective_messages'] for w in windows)
        for condition in ('base','adapter'):
            assert json.loads((tmp_path/'ledgers'/(condition+'_task_tokens.json')).read_text())['completion_tokens_by_task']['one']==1
    else:assert not lengths and not list((tmp_path/'ledgers').glob('*_task_tokens.json'))


def test_podman_identity_rejects_unscoped_graph_store(monkeypatch):
    import os
    import swe_study.docker_storage as module
    monkeypatch.setenv('SWE_STUDY_CONTAINER_ENGINE','podman-hpc')
    monkeypatch.setenv('DOCKER_HOST','unix:///tmp/scoped.sock')
    info={'store':{'graphRoot':'/tmp/'+str(os.getuid())+'_swe_study_podman/graph'},'version':{'Version':'5.4.0'}}
    monkeypatch.setattr(module,'_json_command',lambda args:info)
    result=module.docker_identity()
    assert result['backend']=='podman-hpc' and result['hostname']
    info['store']['graphRoot']='/tmp/unrelated-existing-store'
    with pytest.raises(RuntimeError,match='explicitly scoped'):
        module.docker_identity()
