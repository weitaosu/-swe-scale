"""Recovery checks for Podman compatibility and precisely scoped API shutdown."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import tomllib

import pytest

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('in_process',[False,True])
def test_live_progress_correlates_existing_agent_logs_and_keeps_real_denominators(tmp_path,monkeypatch,in_process):
    module=load_script('benchmark_progress');monkeypatch.setattr(module,'ROOT',tmp_path)
    def save(path,record):
        path=tmp_path/path;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(record));return path
    save('study/protocol.json',{'frozen_development_ids':['one','two'],'agent_config':{'max_steps':30}})
    save('study/development/tasks/one/prepared.json',{'passed':True})
    parent='results/cloud_benchmark_predict/parent'
    save(parent+'/run.json',{'pid':31,'status':'running','started_at':'2026-11-24T01:00:00+00:00',
                            'config':{'state':'study','suite':'development'}})
    save('study/development/tasks/one/base.json',{'status':'predicted','agent_error':'tool failure'})
    save('study/development/tasks/one/adapter.json',{'status':'started','run':parent})
    for name,pid,step in [('ours',31,7),('other-study',32,29)]:
        agent='results/benchmark_agent/'+name
        save(agent+'/run.json',{'pid':31 if in_process else pid+100,'parent_pid':999 if in_process else pid,
                               'status':'running','started_at':'2026-11-24T01:01:00+00:00',
                               'config':{'instance_id':'one','condition':'adapter','suite':'development'}})
        log=tmp_path/agent/'openhands/dataset/CodeActAgent/model/infer_logs/instance_one.log'
        log.parent.mkdir(parents=True);log.write_text('LEVEL 0 LOCAL STEP 1 GLOBAL STEP '+str(step)+'\n')
    (tmp_path/parent/'stdout.log').write_text('Run log: '+str(tmp_path/'results/benchmark_agent/ours')+'\n')
    save('results/cloud_benchmark_grade/grade-parent/run.json',{'pid':41,'status':'running',
         'started_at':'2026-11-24T01:00:00+00:00','config':{'state':'study','suite':'development'}})
    save('results/benchmark_grade/grader/run.json',{'pid':41 if in_process else 141,'parent_pid':999 if in_process else 41,'status':'running',
         'started_at':'2026-11-24T01:02:00+00:00','config':{'instance_id':'one','condition':'base'}})
    (tmp_path/'results/cloud_benchmark_grade/grade-parent/stdout.log').write_text(
        'Run log: '+str(tmp_path/'results/benchmark_grade/grader')+'\n')
    before={p:p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    report=module.snapshot('study')
    assert report['episode_count']==4 and report['counts']=={
        'prepared':1,'predictions':1,'graded':0,'interrupted':0,'agent_errors':1}
    assert report['episodes'][1]['agent']['reported_step']==7
    assert report['episodes'][1]['agent']['run'].endswith('/ours')
    assert report['episodes'][0]['grader']['recorded_status']=='running'
    assert module.snapshot('study',max_tasks=1)['episode_count']==2
    rendered=module.render(report)
    assert '7/30 (step budget, not completion %)' in rendered
    assert 'grading running (recorded)' in rendered and 'no ETA' in rendered
    assert before=={p:p.read_bytes() for p in before}


def test_progress_does_not_invent_steps_or_mark_interrupted_attempts_complete(tmp_path,monkeypatch):
    module=load_script('benchmark_progress');monkeypatch.setattr(module,'ROOT',tmp_path)
    state=tmp_path/'study';state.mkdir()
    (state/'protocol.json').write_text(json.dumps({'frozen_development_ids':['one']}))
    task=state/'development/tasks/one';task.mkdir(parents=True)
    (task/'base.json').write_text(json.dumps({'status':'interrupted'}))
    (task/'adapter.json').write_text('{partial')
    report=module.snapshot('study')
    assert report['counts']['predictions']==report['counts']['graded']==0
    assert report['counts']['interrupted']==1
    assert all(row['agent'] is None for row in report['episodes'])
    assert 'Reported step' not in module.render(report)
    assert module.snapshot('missing')['error']


def test_runtime_image_identity_accepts_full_podman_and_api_forms_only():
    from swe_study.runtime import canonical_image_id,require_image_identity
    digest='95d37f7b50011ef65bad32c6ed900f716cdd2a3c588070b65e5eb7702997efdd'
    assert canonical_image_id(digest)==canonical_image_id('sha256:'+digest)
    require_image_identity(digest,'sha256:'+digest,'Prepared runtime image identity changed')
    require_image_identity('sha256:'+digest,digest,'Prepared runtime image identity changed')
    with pytest.raises(ValueError,match='Prepared runtime image identity changed'):
        require_image_identity('0'*64,'sha256:'+digest,'Prepared runtime image identity changed')
    for invalid in (digest[:12],'sha256:'+digest[:12],'other:'+digest,'','latest',None):
        with pytest.raises(ValueError,match='complete SHA-256'):
            require_image_identity(invalid,'sha256:'+digest,'Image changed')


@pytest.mark.parametrize('problem',[None,'protocol','image','gold','runtime','cache','attempt'])
def test_preparation_reuse_is_model_independent_and_preserves_shared_state(tmp_path,monkeypatch,problem):
    module=load_script('reuse_task_preparation');monkeypatch.setattr(module,'ROOT',tmp_path)
    monkeypatch.setattr(module,'local_path',lambda p:(tmp_path/p).resolve())
    monkeypatch.setattr(module,'read_json',lambda p:json.loads((tmp_path/p).read_text()))
    def save(path,value):
        path=tmp_path/path;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
    monkeypatch.setattr(module,'write_json',save)
    monkeypatch.setattr(module,'record_path',lambda state,suite,key,condition=None:
                        tmp_path/state/suite/'tasks'/key/((condition or 'prepared')+'.json'))
    source=tmp_path/'source';dest=tmp_path/'dest';key='example__repo-1'
    identity={'adapter':'three','model':'3B','adapter_weights_sha256':'three','runtime_overlay_sha256':'same'}
    save(source/'protocol.json',{**identity,'adapter':'one','model':'1.5B','adapter_weights_sha256':'one',
                                'runtime_overlay_sha256':'different' if problem=='protocol' else 'same'})
    for control in ('gold','blank'):
        save(tmp_path/control/'grade_validation.json',{'passed':problem!='gold' or control!='gold','control':control,
                'total_instances':1,'resolved_ids':[key] if control=='gold' else []})
    runtime={'passed':problem!='runtime','instance_id':key,'base_image':'base','runtime_image':'runtime',
             'runtime_image_id':'sha256:'+'a'*64}
    save(tmp_path/'runtime-check/runtime.json',runtime)
    save(tmp_path/'manifests/runtime_cache.json',{key:{**runtime,'runtime_image':'other' if problem=='cache' else 'runtime'}})
    item={**runtime,'passed':True,'base_image_id':'sha256:'+'b'*64,'gold_passed':True,
          'gold_control':'gold','blank_control':'blank','runtime_run':'runtime-check','new_image_ids':['owned-by-source']}
    old=module.record_path(source,'development',key);save(old,item)
    if problem=='attempt':save(module.record_path(dest,'development',key,'base'),{'status':'started'})
    events=[];run=SimpleNamespace(path=tmp_path/'review',event=lambda *a,**k:events.append((a,k)))
    client=SimpleNamespace(images=SimpleNamespace(get=lambda name:SimpleNamespace(
        id='sha256:'+('b'*64 if name=='base' else 'c'*64 if problem=='image' else 'a'*64))))
    before={p:p.read_bytes() for root in (source,tmp_path/'manifests') for p in root.rglob('*') if p.is_file()}
    target=module.record_path(dest,'development',key)
    if problem:
        with pytest.raises(ValueError):module.reuse(run,client,source,dest,'development',key,identity)
        assert not target.exists()
    else:
        module.reuse(run,client,source,dest,'development',key,identity)
        copied=json.loads(target.read_text())
        assert copied['new_image_ids']==[] and copied['runtime_image_id']==item['runtime_image_id']
        assert copied['reused_preparation']['source_sha256']==module.sha256_file(old)
        assert not module.record_path(dest,'development',key,'base').exists()
        assert len(events)==1
    assert before=={p:p.read_bytes() for p in before}


@pytest.mark.parametrize('problem',[None,'tokens','changed_image','agent_output','wrong_trace','source_drift'])
def test_prestart_review_preserves_records_and_rejects_used_or_changed_runs(tmp_path,monkeypatch,problem):
    import hashlib,zipfile
    module=load_script('review_prestart_failure')
    monkeypatch.setattr(module,'ROOT',tmp_path)
    monkeypatch.setattr(module,'local_path',lambda p:(tmp_path/p).resolve())
    monkeypatch.setattr(module,'read_json',lambda p:json.loads((tmp_path/p).read_text()))
    state=tmp_path/'state';key='example__repo-1';task=state/'development/tasks'/key
    task.mkdir(parents=True);agent=tmp_path/'results/benchmark_agent/failed';agent.mkdir(parents=True)
    def save(path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
    source='configure_task_images(rows,args.output,args.run_id)\nrunpy.run_path("evaluator")\n'
    entry=tmp_path/'scripts/openhands_entry.py';entry.parent.mkdir();entry.write_text(source)
    old_runtime=b'original runtime source\n'
    with zipfile.ZipFile(agent/'source_snapshot.zip','w') as archive:
        archive.writestr('scripts/openhands_entry.py',source)
        archive.writestr('src/swe_study/runtime.py',old_runtime)
    save(agent/'run.json',{'stage':'benchmark_agent','status':'failed'})
    (agent/'command-001.stderr.log').write_text('File "/project/scripts/openhands_entry.py", line 1, in <module>\n'
        '    configure_task_images(rows,args.output,args.run_id)\nValueError: Prepared runtime image identity changed\n')
    save(task/'base.json',{'status':'interrupted','agent_run':str(agent)})
    save(state/'protocol.json',{'runtime_overlay_sha256':hashlib.sha256(old_runtime).hexdigest()})
    digest='a'*64
    save(task/'prepared.json',{'base_image':'base','runtime_image':'runtime',
                              'base_image_id':'sha256:'+'b'*64,'runtime_image_id':'sha256:'+digest})
    save(tmp_path/'manifests/runtime_cache.json',{key:{'runtime_image':'runtime','runtime_image_id':digest}})
    if problem=='tokens':save(state/'development/token_ledgers/base_task_tokens.json',{'completion_tokens_by_task':{key:1}})
    if problem=='agent_output':(agent/'output.jsonl').write_text('{}\n')
    if problem=='wrong_trace':(agent/'command-001.stderr.log').write_text('Some other failure')
    if problem=='source_drift':entry.write_text(source+'# changed\n')
    image=lambda name:SimpleNamespace(id='sha256:'+('b'*64 if name=='base' else 'c'*64 if problem=='changed_image' else digest))
    client=SimpleNamespace(images=SimpleNamespace(get=image))
    before={p:p.read_bytes() for root in (state,agent) for p in root.rglob('*') if p.is_file()}
    if problem:
        with pytest.raises(ValueError):module.review(state,agent,key,client)
    else:
        report=module.review(state,agent,key,client)
        assert report['passed'] and report['records_modified'] is False
    assert before=={p:p.read_bytes() for p in before}


def load_script(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def process(proc,pid,uid,args):
    path=proc/str(pid);path.mkdir()
    (path/'status').write_text(f'Name:\tpodman\nUid:\t{uid}\t{uid}\t{uid}\t{uid}\n')
    (path/'cmdline').write_bytes(b'\0'.join(arg.encode() for arg in args)+b'\0')
    return path


def test_stop_matches_only_exact_owned_project_api_service(tmp_path):
    module=load_script('stop_container_service')
    endpoint='unix:///tmp/123_swe_study_podman/docker.sock'
    process(tmp_path,10,123,['/usr/bin/podman','system','service','--time=0',endpoint])
    process(tmp_path,11,123,['/usr/bin/python3','/usr/bin/podman-hpc','system','service',endpoint])
    process(tmp_path,12,456,['podman','system','service',endpoint])
    process(tmp_path,13,123,['podman','system','service','unix:///tmp/unrelated.sock'])
    process(tmp_path,14,123,['podman','run','echo','system','service',endpoint])
    process(tmp_path,15,123,['grep','podman','system','service',endpoint])
    process(tmp_path,16,123,['python3','unrelated.py','system','service',endpoint])
    process(tmp_path,17,123,['podman','system','--root','/tmp/project','--runroot','/tmp/run','service','--time=0',endpoint])
    process(tmp_path,18,123,['podman','system','--root','/tmp/project','prune','service',endpoint])
    # podman run's command can contain arbitrary words: never treat it as an API service.
    assert module.service_pids(123,endpoint,tmp_path)==[10,11,17]


def test_failed_container_start_still_removes_probe_and_closes_client(tmp_path,monkeypatch):
    module=load_script('check_container_api')
    removed=[];closed=[]
    def fail():raise RuntimeError('simulated port-forwarding failure')
    container=SimpleNamespace(id='probe',start=fail,remove=lambda **kw:removed.append(kw),
                              reload=lambda:None,attrs={'HostConfig':{'NetworkMode':'slirp4netns'}})
    client=SimpleNamespace(ping=lambda:True,images=SimpleNamespace(get=lambda image:SimpleNamespace(id='image')),
                           containers=SimpleNamespace(create=lambda *args,**kwargs:container),
                           close=lambda:closed.append(True))
    monkeypatch.setitem(sys.modules,'docker',SimpleNamespace(from_env=lambda **kwargs:client))
    class Run:
        id='probe';path=tmp_path
        def __init__(self,*args):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def command(self,*args,**kwargs):pass
    monkeypatch.setattr(module,'Run',Run)
    monkeypatch.setattr(module,'setup_environment',lambda:None)
    monkeypatch.delenv('SWE_STUDY_CONTAINER_ENGINE',raising=False)
    monkeypatch.setattr(module,'record_storage',lambda run:None)
    monkeypatch.setattr(module,'require_storage',lambda:None)
    monkeypatch.setattr(module,'StorageGuard',lambda:SimpleNamespace(check=lambda *args:None))
    monkeypatch.setattr(sys,'argv',['check_container_api.py'])
    with pytest.raises(RuntimeError,match='simulated port-forwarding'):
        module.main()
    assert removed==[{'force':True}] and closed==[True]


@pytest.mark.parametrize('job_id,host',[('', 'nid001008'),('123','login23')])
def test_podman_api_probe_rejects_login_and_unallocated_shell(monkeypatch,job_id,host):
    module=load_script('check_container_api')
    monkeypatch.setenv('SWE_STUDY_CONTAINER_ENGINE','podman-hpc')
    monkeypatch.setenv('SLURM_JOB_ID',job_id)
    monkeypatch.setattr(module.socket,'gethostname',lambda:host)
    with pytest.raises(RuntimeError,match='srun before this container check'):
        module.require_compute_allocation()


@pytest.mark.parametrize('job_id,host',[('', 'nid001008'),('123','login23')])
@pytest.mark.parametrize('script',['container_service.sh','recover_containers.sh'])
def test_api_service_rejects_login_or_unallocated_shell(tmp_path,job_id,host,script):
    service=tmp_path/'cloud/nersc'/script;service.parent.mkdir(parents=True)
    service.write_text((ROOT/'cloud/nersc'/script).read_text())
    (service.parent/'container_env.sh').write_text('# isolated fixture\n')
    bins=tmp_path/'bin';bins.mkdir()
    hostname=bins/'hostname';hostname.write_text('#!/bin/sh\necho '+host+'\n');hostname.chmod(0o755)
    env=dict(os.environ,PATH=str(bins)+':'+os.environ['PATH'],SLURM_JOB_ID=job_id)
    result=subprocess.run(['bash',str(service)],env=env,text=True,capture_output=True)
    assert result.returncode!=0 and 'compute' in result.stderr


def test_network_override_preserves_site_settings_and_selects_userspace_network():
    config=tomllib.loads((ROOT/'cloud/nersc/containers-network.toml').read_text())
    assert config=={'containers':{'netns':'slirp4netns'},
                    'network':{'default_rootless_network_cmd':'slirp4netns'}}


@pytest.mark.parametrize('condition',['empty_setup','working_tree','index','changed_head','missing_base'])
def test_checkout_accepts_setup_commit_but_rejects_changed_source(tmp_path,condition):
    module=load_script('prepare_task_runtime')
    def git(*args):
        return subprocess.run(['git','-c','user.name=Fixture','-c','user.email=fixture@example.invalid',*args],
                              cwd=tmp_path,check=True,text=True,capture_output=True).stdout.strip()
    git('init','-q');(tmp_path/'source.py').write_text('original\n');git('add','source.py');git('commit','-qm','base')
    base=git('rev-parse','HEAD');git('commit','--allow-empty','-qm','SWE-bench setup')
    assert git('rev-parse','HEAD')!=base
    if condition in {'working_tree','index','changed_head'}:
        (tmp_path/'source.py').write_text('changed\n')
    if condition in {'index','changed_head'}:git('add','source.py')
    if condition=='index':(tmp_path/'source.py').write_text('original\n')
    if condition=='changed_head':git('commit','-qm','changed source')
    if condition=='missing_base':base='0'*40
    result=subprocess.run(['bash','-c',module.checkout_command(base,tmp_path)],text=True,capture_output=True)
    assert (result.returncode==0)==(condition=='empty_setup')
    assert ('SWE_STUDY_CHECKOUT_OK' in result.stdout)==(condition=='empty_setup')


def test_podman_builder_uses_engine_identity_and_initializes_required_state(monkeypatch):
    from swe_study.runtime import configure_builder_backend
    class Builder:
        def __init__(self,client):raise RuntimeError('Docker version guard')
    logger=lambda **kwargs:SimpleNamespace(**kwargs)
    original=Builder.__init__
    monkeypatch.delenv('SWE_STUDY_CONTAINER_ENGINE',raising=False)
    configure_builder_backend(Builder,logger)
    assert Builder.__init__ is original
    monkeypatch.setenv('SWE_STUDY_CONTAINER_ENGINE','podman-hpc')
    version={'Version':'5.8.2','Components':[{'Name':'Podman Engine','Version':'5.8.2'}]}
    client=SimpleNamespace(version=lambda:version)
    configure_builder_backend(Builder,logger)
    instance=Builder(client)
    assert instance.docker_client is client and instance.rolling_logger.max_lines==10
    assert version['Version']=='5.8.2'
    with pytest.raises(RuntimeError,match='does not identify a Podman engine'):
        Builder(SimpleNamespace(version=lambda:{'Version':'5.8.2','Components':[{'Name':'Docker Engine'}]}))


@pytest.mark.parametrize('args,expected_args',[
    (['build','--tag','cs6158/probe:test','.'],['build','--tag','docker.io/cs6158/probe:test','.']),
    (['image','build','.'],['image','build','.']),
    (['build','--tag=project:test','-tghcr.io/owner/probe:test','.'],
     ['build','--tag=docker.io/library/project:test','-tghcr.io/owner/probe:test','.'])])
def test_build_shim_overrides_project_tmp_only_for_build_child(tmp_path,args,expected_args):
    bins=tmp_path/'bin';bins.mkdir()
    for name in ('mkdir','chmod'):
        path=bins/name;path.write_text('#!/bin/sh\nexit 0\n');path.chmod(0o755)
    engine=bins/'podman-hpc'
    engine.write_text('#!/bin/sh\nprintf "%s\\n" "$TMPDIR" "$TEMP" "$TMP" "$@"\n');engine.chmod(0o755)
    original=str(tmp_path/'project-temp')
    env=dict(os.environ,PATH=str(bins)+':'+os.environ['PATH'],SWE_STUDY_CONTAINER_ENGINE='podman-hpc',
             TMPDIR=original,TEMP=original,TMP=original)
    shim=ROOT/'cloud/nersc/podman-bin/docker'
    result=subprocess.run(['bash',str(shim),*args],env=env,text=True,capture_output=True,check=True)
    expected='/tmp/'+str(os.getuid())+'_swe_study_podman/build-tmp'
    assert result.stdout.splitlines()==[expected]*3+expected_args
    query=subprocess.run(['bash',str(shim),'image','inspect','project:test'],env=env,text=True,capture_output=True,check=True)
    assert query.stdout.splitlines()==[original]*3+['image','inspect','project:test']


def test_build_probe_executes_uncached_run_and_removes_only_its_image(tmp_path):
    module=load_script('check_container_api');commands=[];removed=[]
    built=SimpleNamespace(id='sha256:probe')
    client=SimpleNamespace(images=SimpleNamespace(get=lambda name:built,
        remove=lambda image,**kwargs:removed.append((image,kwargs))))
    run=SimpleNamespace(path=tmp_path,id='unique-probe',command=lambda args,**kwargs:commands.append(args))
    report=module.check_container_build(run,client,SimpleNamespace(id='sha256:base'),None)
    assert report['passed'] and report['base_image_id']=='sha256:base'
    assert commands[0][:5]==['docker','build','--no-cache','--tag','cs6158/build-probe:unique-probe']
    assert 'RUN python -c' in (tmp_path/'build_probe/Dockerfile').read_text()
    assert removed==[('sha256:probe',{'force':False})]


def test_archive_upload_replaces_host_ownership_and_preserves_bytes_and_mode(tmp_path):
    import io,tarfile
    from swe_study.container_archives import copy_to_container,container_owner
    member=tarfile.TarInfo('patch.diff');member.uid=member.gid=115408;member.mode=0o755
    member.pax_headers={'uid':'115408','gid':'115408','uname':'ws535','gname':'users','mtime':'123.5'}
    normalized=container_owner(member)
    assert (normalized.uid,normalized.gid,normalized.uname,normalized.gname)==(0,0,'root','root')
    assert normalized.pax_headers=={'mtime':'123.5'} and normalized.mode==0o755
    source=tmp_path/'patch.diff';content=b'diff --git a/file b/file\n+binary\x00payload\n'
    source.write_bytes(content);source.chmod(0o755)
    uploads=[];commands=[]
    container=SimpleNamespace(exec_run=lambda command:(commands.append(command) or SimpleNamespace(exit_code=0)),
                              put_archive=lambda path,data:(uploads.append((path,data)) or True))
    copy_to_container(container,source,'/tmp/upload/patch.diff')
    assert commands==[['mkdir','-p','/tmp/upload']] and uploads[0][0]=='/tmp/upload'
    with tarfile.open(fileobj=io.BytesIO(uploads[0][1])) as archive:
        entry=archive.getmembers()[0]
        assert entry.name=='patch.diff' and entry.mode==(source.stat().st_mode&0o7777) and entry.uid==entry.gid==0
        assert archive.extractfile(entry).read()==content
    assert source.read_bytes()==content


def test_archive_upload_failure_is_not_reported_as_success(tmp_path):
    from swe_study.container_archives import copy_to_container
    source=tmp_path/'patch';source.write_text('reference patch')
    container=SimpleNamespace(exec_run=lambda command:SimpleNamespace(exit_code=0),put_archive=lambda *args:False)
    with pytest.raises(RuntimeError,match='archive upload failed'):
        copy_to_container(container,source,'/tmp/patch')


@pytest.mark.parametrize('image,expected',[
    ('swebench/task:latest','docker.io/swebench/task:latest'),
    ('python:3.12-slim','docker.io/library/python:3.12-slim'),
    ('swebench/task@sha256:abc','docker.io/swebench/task@sha256:abc'),
    ('docker.io/swebench/task:latest','docker.io/swebench/task:latest'),
    ('ghcr.io/owner/task:tag','ghcr.io/owner/task:tag'),
    ('registry.example:5000/owner/task:tag','registry.example:5000/owner/task:tag'),
    ('localhost/task:tag','localhost/task:tag'),
    ('docker://docker.io/swebench/task:tag','docker://docker.io/swebench/task:tag'),
    ('sha256:abc','sha256:abc'),
])
def test_podman_pull_uses_docker_hub_for_unqualified_names(monkeypatch,image,expected):
    from swe_study.container_options import pull_reference
    monkeypatch.setenv('SWE_STUDY_CONTAINER_ENGINE','podman-hpc')
    assert pull_reference(image)==expected
    monkeypatch.delenv('SWE_STUDY_CONTAINER_ENGINE')
    assert pull_reference(image)==image


def test_explicit_podman_network_preserves_docker_and_rejects_host_network(monkeypatch):
    from swe_study.container_options import container_options
    original={'ports':{'8765/tcp':('127.0.0.1',None)},'network_mode':None}
    monkeypatch.setenv('SWE_STUDY_CONTAINER_ENGINE','podman-hpc')
    assert container_options(original)['network_mode']=='slirp4netns'
    assert original['network_mode'] is None
    for mode in ['host','container:unrelated','custom-network']:
        with pytest.raises(ValueError,match='require slirp4netns'):
            container_options({'network_mode':mode})
    monkeypatch.delenv('SWE_STUDY_CONTAINER_ENGINE')
    assert container_options(original)==original
    assert container_options({'network_mode':'custom-network'})=={'network_mode':'custom-network'}


def test_stop_refuses_success_when_unknown_api_is_still_alive(tmp_path,monkeypatch):
    module=load_script('stop_container_service')
    class Run:
        path=tmp_path
        def __init__(self,*args):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
    monkeypatch.setattr(module,'Run',Run)
    monkeypatch.setattr(module,'setup_environment',lambda:None)
    monkeypatch.setattr(module,'service_pids',lambda *args:[])
    monkeypatch.setattr(module,'endpoint_responds',lambda *args:True)
    monkeypatch.setattr(module.os,'kill',lambda *args:pytest.fail('An unidentified process must be preserved'))
    monkeypatch.delenv('DOCKER_HOST',raising=False)
    monkeypatch.setattr(sys,'argv',['stop_container_service.py','--stop'])
    with pytest.raises(RuntimeError,match='process was not identified'):
        module.main()
