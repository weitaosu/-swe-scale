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
