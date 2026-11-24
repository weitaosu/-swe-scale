"""Dependency repair must preserve evidence and pass reference controls."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1]


def load_repair():
    spec=importlib.util.spec_from_file_location('repair_task_environment',ROOT/'scripts/repair_task_environment.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('numpy_version,pytest_version',[
    (None,None),('>=1.26',None),(None,'7.4.4; true'),(None,'7.4.*')])
def test_repair_rejects_missing_or_unpinned_dependencies(numpy_version,pytest_version):
    with pytest.raises(ValueError):load_repair().dependency_pins(numpy_version,pytest_version)


@pytest.mark.parametrize('failure',[None,'gold','blank','invalid_gold_report'])
def test_repair_is_not_accepted_without_passing_reference_controls(tmp_path,monkeypatch,failure):
    module=load_repair();task='pydicom__pydicom-1048'
    run_path=tmp_path/'repair';run_path.mkdir()
    evidence=tmp_path/'failed-control';evidence.mkdir();(evidence/'test_output.txt').write_text('missing file_out')
    task_dir=tmp_path/'vendor/swe-bench-tasks/tasks'/task;task_dir.mkdir(parents=True)
    (task_dir/'task.yaml').write_text('image: swebench/original:latest\n')
    manifest=tmp_path/'manifests/image_overrides.json';manifest.parent.mkdir()
    manifest.write_text(json.dumps({'overrides':{}}))
    monkeypatch.setattr(module,'ROOT',tmp_path)
    monkeypatch.setattr(module,'setup_environment',lambda:None)
    monkeypatch.setattr(module,'require_storage',lambda:None)
    monkeypatch.setattr(module,'local_path',lambda path:tmp_path/path)
    monkeypatch.setattr(module,'StorageGuard',lambda:SimpleNamespace(check=lambda *args:None))
    def read(path):
        if str(path)=='manifests/dev_tasks.json':return {'instance_ids':[task]}
        return json.loads((tmp_path/path).read_text())
    def write(path,value):
        target=tmp_path/path;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(json.dumps(value))
    monkeypatch.setattr(module,'read_json',read);monkeypatch.setattr(module,'write_json',write)
    probes=[];closed=[]
    def probe(client,image_id,run,label):
        probes.append((image_id,label))
        return {'python':'3.8','packages':{'numpy':'1.24.4','pytest':'7.4.4' if label=='after' else '8.3.5'}}
    monkeypatch.setattr(module,'probe_versions',probe)
    monkeypatch.setitem(sys.modules,'docker',SimpleNamespace(from_env=lambda **kwargs:SimpleNamespace(close=lambda:closed.append(True))))
    commands=[]
    class Run:
        path=run_path;id='repair-fixture'
        def __init__(self,*args):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def command(self,args,**kwargs):
            args=[str(value) for value in args];commands.append(args)
            output=run_path/('command-'+str(len(commands))+'.stdout.log')
            if args[:3]==['docker','image','inspect']:
                image='sha256:source' if args[3]=='swebench/original:latest' else 'sha256:repaired'
                output.write_text(json.dumps([{'Id':image,'RootFS':{'Layers':['original']}}]))
            elif 'grade' in args:
                control=args[args.index('--control')+1]
                if failure==control:raise RuntimeError('Official control failed')
                folder=tmp_path/'results/grade'/control;folder.mkdir(parents=True)
                report={'passed':True,'control':control,'total_instances':1,
                        'resolved_ids':[task] if control=='gold' else []}
                if failure=='invalid_gold_report' and control=='gold':report['resolved_ids']=[]
                write(folder/'grade_validation.json',report)
                output.write_text('Run log: '+str(folder)+'\n')
            else:output.write_text('Build succeeded\n')
            return {'stdout_path':str(output)}
    monkeypatch.setattr(module,'Run',Run)
    argv=['--instance-id',task,'--pytest-version','7.4.4','--evidence','failed-control']
    if failure:
        with pytest.raises((RuntimeError,ValueError)):module.main(argv)
    else:module.main(argv)
    entry=json.loads(manifest.read_text())['overrides'][task]
    assert entry['dependency_pins']=={'pytest':'7.4.4'}
    assert entry['status']==('candidate_pending_controls' if failure else 'gold_and_blank_controls_passed')
    assert (run_path/'repair_validation.json').exists()==(failure is None)
    assert probes==[('sha256:source','before'),('sha256:repaired','after')] and closed==[True]
    dockerfile=(run_path/'build/Dockerfile').read_text()
    assert dockerfile.startswith('FROM sha256:source\n') and 'pytest==7.4.4' in dockerfile and 'numpy==' not in dockerfile
    assert all(command[0]!='pip' for command in commands)  # Never downgrade a host environment.


def test_dependency_probe_removes_only_its_container_even_when_start_fails(tmp_path):
    module=load_repair();created=[];removed=[]
    def start():raise RuntimeError('Container start failed')
    container=SimpleNamespace(start=start,remove=lambda **kwargs:removed.append(kwargs))
    def create(*args,**kwargs):
        created.append((args,kwargs));return container
    client=SimpleNamespace(containers=SimpleNamespace(create=create))
    with pytest.raises(RuntimeError,match='start failed'):
        module.probe_versions(client,'sha256:source',SimpleNamespace(path=tmp_path,id='unique'),'before')
    assert created[0][1]['network_mode']=='none'
    assert created[0][1]['name']=='cs6158-repair-before-unique'
    assert 'volumes' not in created[0][1] and removed==[{'force':True}]


def test_existing_numpy_repair_option_remains_supported():
    assert load_repair().dependency_pins('1.26.4')=={'numpy':'1.26.4'}
