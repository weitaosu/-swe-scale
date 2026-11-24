"""Recovery checks for Podman compatibility and precisely scoped API shutdown."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import tomllib

import pytest

ROOT=Path(__file__).resolve().parents[1]


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
