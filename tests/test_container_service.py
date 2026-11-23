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
    # podman run's command can contain arbitrary words: never treat it as an API service.
    assert module.service_pids(123,endpoint,tmp_path)==[10,11]


def test_failed_container_start_still_removes_probe_and_closes_client(tmp_path,monkeypatch):
    module=load_script('check_container_api')
    removed=[];closed=[]
    def fail():raise RuntimeError('simulated port-forwarding failure')
    container=SimpleNamespace(start=fail,remove=lambda **kw:removed.append(kw))
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
    monkeypatch.setattr(module,'record_storage',lambda run:None)
    monkeypatch.setattr(module,'require_storage',lambda:None)
    monkeypatch.setattr(module,'StorageGuard',lambda:SimpleNamespace(check=lambda *args:None))
    monkeypatch.setattr(sys,'argv',['check_container_api.py'])
    with pytest.raises(RuntimeError,match='simulated port-forwarding'):
        module.main()
    assert removed==[{'force':True}] and closed==[True]


@pytest.mark.parametrize('job_id,host',[('', 'nid001008'),('123','login23')])
def test_api_service_rejects_login_or_unallocated_shell(tmp_path,job_id,host):
    service=tmp_path/'cloud/nersc/container_service.sh';service.parent.mkdir(parents=True)
    service.write_text((ROOT/'cloud/nersc/container_service.sh').read_text())
    (service.parent/'container_env.sh').write_text('# isolated fixture\n')
    bins=tmp_path/'bin';bins.mkdir()
    hostname=bins/'hostname';hostname.write_text('#!/bin/sh\necho '+host+'\n');hostname.chmod(0o755)
    env=dict(os.environ,PATH=str(bins)+':'+os.environ['PATH'],SLURM_JOB_ID=job_id)
    result=subprocess.run(['bash',str(service)],env=env,text=True,capture_output=True)
    assert result.returncode!=0 and 'Slurm compute allocation' in result.stderr


def test_network_override_preserves_site_settings_and_selects_userspace_network():
    config=tomllib.loads((ROOT/'cloud/nersc/containers-network.toml').read_text())
    assert config=={'containers':{'netns':'slirp4netns'}}
