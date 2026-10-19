#!/usr/bin/env python3
"""Validate one local model API and optionally run the frozen development workflow."""
import argparse,json,os,re,signal,socket,subprocess,sys,time,urllib.request,urllib.error,urllib.parse
from pathlib import Path
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from swe_study.common import setup_environment,read_json,read_jsonl,write_json,sha256_file,assert_local_port_available
setup_environment()
from swe_study.audit import Run
from swe_study.storage import StorageGuard
from swe_study.docker_storage import require_storage
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--adapter');p.add_argument('--label',required=True);p.add_argument('--rollout',action='store_true');p.add_argument('--condition-id');p.add_argument('--smoke',action='store_true');a=p.parse_args()
if a.smoke and not a.rollout: p.error('--smoke requires --rollout')
mode_args=['--smoke'] if a.smoke else []
run=Run('development_condition',vars(a));owner=a.condition_id or run.id
if not re.fullmatch('[A-Za-z0-9-]+',owner): raise ValueError('Invalid condition ID')
os.environ['CS6158_CONDITION_ID']=owner;run.meta['condition_id']=owner
server=None
with run:
    guard=StorageGuard();guard.check(2*10**9);require_storage()
    import yaml
    config=yaml.safe_load((root/'configs/agent.yaml').read_text());base=config['base_url'].rstrip('/');port=urllib.parse.urlparse(base).port or 80
    if urllib.parse.urlparse(base).hostname!='127.0.0.1': raise ValueError('Expected loopback model API')
    assert_local_port_available(port)
    model_name='cs6158-'+Path(a.model).name
    command=[str(root/'.envs/train/bin/python'),str(root/'scripts/study.py'),'serve','--model',a.model,'--port',str(port)]
    if a.adapter: command.extend(['--adapter',a.adapter])
    def request(path,body=None):
        data=json.dumps(body).encode() if body is not None else None
        req=urllib.request.Request(base.removesuffix('/v1')+path,data=data,headers={'Content-Type':'application/json'})
        try:
            with urllib.request.urlopen(req,timeout=180) as response: return response.status,json.load(response)
        except urllib.error.HTTPError as exc: return exc.code,json.loads(exc.read())
    def child_run(path,*args,timeout=2400): return run.command([root/'.envs/train/bin/python',path,*args],guard=guard,timeout=timeout)
    def child_path(command_record,stage):
        prefix=str(root/'results'/stage)+'/'
        candidates=[Path(line[len('Run log: '):]) for line in Path(command_record['stdout_path']).read_text().splitlines() if line.startswith('Run log: '+prefix)]
        if not candidates: raise ValueError('Missing audited child run path for '+stage)
        target=candidates[0].resolve()
        if not target.is_relative_to(root/'results'/stage): raise ValueError('Child run escaped project stage')
        return target
    try:
        with (run.path/'server.stdout.log').open('w') as out,(run.path/'server.stderr.log').open('w') as err:
            # Inherit this supervised job's process group so outer cancellation also stops the server.
            server=subprocess.Popen(command,cwd=root,stdout=out,stderr=err,text=True)
            write_json(run.path/'server_process.json',{'args':command,'pid':server.pid,'condition_id':owner})
            run.event('server_process_started',pid=server.pid,args=command)
            started=time.monotonic();health=None
            while time.monotonic()-started<900:
                guard.check()
                if server.poll() is not None: raise RuntimeError('Model server exited before readiness; inspect server logs')
                try:
                    status,payload=request('/health')
                    if status==200 and payload.get('model')==model_name and payload.get('adapter')==a.adapter: health=payload;break
                except (urllib.error.URLError,TimeoutError): pass
                time.sleep(1)
            if not health: raise TimeoutError('Model server did not become ready')
            issue=next(read_jsonl('data/metadata/dev_task_budgets.jsonl'))[1]['problem_statement']
            body={'model':model_name,'messages':[{'role':'user','content':[{'type':'text','text':issue+'\nReply briefly for the local interface check.'}]}],'max_tokens':1,'temperature':0,'stop':['</function']}
            checks=[]
            for title,changed,expected in [('wrong_model',{**body,'model':'wrong'},404),('multiple_choices',{**body,'n':2},400),('unknown_task',{**body,'messages':[{'role':'user','content':'unregistered interface task'}]},400),('context_limit',{**body,'messages':[{'role':'user','content':issue+' '+('x '*10000)}]},413),('generation',body,200)]:
                status,response=request('/v1/chat/completions',changed)
                run.event('interface_response',check=title,status=status,response=response)
                if status!=expected: raise ValueError(title+' failed: '+str(status))
                checks.append(title)
            write_json(run.path/'interface_validation.json',{'passed':True,'health':health,'checks':checks,'preflight_completion_tokens':response['usage']['completion_tokens'],'preflight_charged_to_first_task_budget':True})
            if a.rollout:
                rollout=child_run(root/'scripts/study.py','rollout','--model-name',model_name,'--python','.envs/agent/bin/python',*mode_args,timeout=9500)
                rollout_path=child_path(rollout,'rollout');outputs=list((rollout_path/'openhands').rglob('output.jsonl'))
                if len(outputs)!=1: raise ValueError('Expected one OpenHands prediction artifact')
                predictions=run.path/'predictions.jsonl'
                child_run(root/'scripts/study.py','export','--input',str(outputs[0]),'--output',str(predictions),'--model-name',a.label,*mode_args)
                grading=child_run(root/'scripts/study.py','grade','--predictions',str(predictions),'--python','.envs/eval/bin/python',*mode_args,timeout=7200)
                grade_path=child_path(grading,'grade')
                rows=[r for _,r in read_jsonl(outputs[0])];wanted=set(read_json('manifests/dev_tasks.json')['instance_ids'][:1] if a.smoke else read_json('manifests/dev_tasks.json')['instance_ids'])
                report={'scope':'single_first_frozen_task_smoke' if a.smoke else 'frozen_five_tasks','interface_passed':True,'task_ids':[r['instance_id'] for r in rows],'complete_task_coverage':{r['instance_id'] for r in rows}==wanted,'errors':[{'instance_id':r['instance_id'],'error':r['error']} for r in rows if r.get('error')],'rollout':str(rollout_path),'grader':str(grade_path),'predictions':str(predictions),'grade':read_json(grade_path/'grade_validation.json')}
                write_json(run.path/'development_validation.json',report)
                if not report['complete_task_coverage']: raise ValueError('Development task coverage incomplete')
    finally:
        if server is not None:
            if server.poll() is None:
                server.send_signal(signal.SIGINT)
                try: server.wait(15)
                except subprocess.TimeoutExpired:
                    server.terminate()
                    try: server.wait(5)
                    except subprocess.TimeoutExpired: server.kill();server.wait()
            run.event('server_process_stopped',pid=server.pid,returncode=server.returncode)
        listing=run.command(['docker','ps','-a','--filter','label=cs6158.condition_id='+owner,'--format','{{json .}}'],timeout=20)
        for line in Path(listing['stdout_path']).read_text().splitlines():
            item=json.loads(line)
            if not item['Names'].startswith(('openhands-runtime-cs6158-','sweb.eval.')): raise ValueError('Unexpected labelled container preserved')
            run.command(['docker','logs',item['ID']],timeout=20)
            run.command(['docker','rm','--force',item['ID']],timeout=30)
