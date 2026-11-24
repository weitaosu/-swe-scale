#!/usr/bin/env python3
"""Create and control-test a dependency-only repair for a frozen development task."""
from pathlib import Path
import argparse,os,re,sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from swe_study.common import local_path,setup_environment,read_json,write_json,sha256_file
from swe_study.audit import Run
from swe_study.storage import StorageGuard
from swe_study.docker_storage import require_storage
def dependency_pins(numpy_version=None,pytest_version=None):
    pins={}
    for name,version in [('numpy',numpy_version),('pytest',pytest_version)]:
        if version is not None:
            if not re.fullmatch(r'[0-9]+(?:[.][0-9]+)+',version):
                raise ValueError('Expected exact '+name+' version')
            pins[name]=version
    if not pins: raise ValueError('Select --numpy-version or --pytest-version')
    return pins


def probe_versions(client,image_id,run,label):
    """Use the working Docker API, without networking or host mounts."""
    code=("import sys,json,importlib.metadata as m; "
          "print(json.dumps({'python':sys.version,'packages':"
          "{name:m.version(name) for name in ['pytest','numpy']}}))")
    container=client.containers.create(
        image_id,entrypoint=['/opt/miniconda3/envs/testbed/bin/python'],
        command=['-c',code],network_mode='none',user='0:0',
        name='cs6158-repair-'+label+'-'+run.id,
        labels={'cs6158.project':'local-swe-study'},detach=True)
    try:
        container.start();status=container.wait(timeout=120)
        output=container.logs().decode(errors='replace')
        (run.path/(label+'-versions.log')).write_text(output)
        if status['StatusCode']!=0: raise RuntimeError('Task dependency probe failed: '+output)
        report=json.loads(output);write_json(run.path/(label+'-versions.json'),report)
        return report
    finally: container.remove(force=True)


def validate_control_result(result,control,instance_id):
    # CLI prints its one run both on entry and after successful completion.
    paths={Path(line[len('Run log: '):]).resolve()
           for line in Path(result['stdout_path']).read_text().splitlines()
           if line.startswith('Run log: '+str(ROOT/'results/grade')+'/')}
    if len(paths)!=1: raise ValueError('Control did not record exactly one distinct run')
    validation=paths.pop()/'grade_validation.json';report=read_json(validation)
    if report.get('passed') is not True or report.get('control')!=control or report.get('total_instances')!=1:
        raise ValueError('Repair control was not validated')
    if report.get('resolved_ids')!=([instance_id] if control=='gold' else []):
        raise ValueError('Repair control resolved IDs disagree')
    return str(validation)


def resume_candidate(args,pins,manifest,source):
    """Recover only the known post-control bookkeeping failure, preserving evidence."""
    folder=local_path(args.resume_repair)
    meta=read_json(folder/'run.json')
    if (meta.get('stage')!='task_environment_repair' or meta.get('status')!='failed'
        or meta.get('error',{}).get('message')!='Control did not record exactly one run'):
        raise ValueError('Resume accepts only the recorded duplicate-run-path failure')
    fields=('instance_id','numpy_version','pytest_version','runtime_parent','evidence')
    if any(meta['config'].get(key)!=getattr(args,key) for key in fields):
        raise ValueError('Resume repair arguments differ from the failed repair')
    entry=json.loads(json.dumps(manifest['overrides'].get(args.instance_id,{})))
    if (entry.get('status')!='candidate_pending_controls' or entry.get('dependency_pins')!=pins
        or local_path(entry.get('run',''))!=folder or entry.get('base_image_id')!=source['Id']):
        raise ValueError('Pending repair image provenance differs from the failed repair')
    results={};command_evidence=[]
    for path in sorted(folder.glob('command-*.json')):
        result=read_json(path);command=result.get('args',[])
        for control in ('gold','blank'):
            expected=list(map(str,[ROOT/'.envs/train/bin/python',ROOT/'scripts/study.py','grade',
                                  '--control',control,'--instance-id',args.instance_id,
                                  '--python','.envs/eval/bin/python']))
            if command!=expected:continue
            if result.get('returncode')!=0 or result.get('error'):
                raise ValueError('Failed control commands cannot be reused')
            if control in results:raise ValueError('Ambiguous repeated control commands preserved')
            # Verify the report now, even when its stdout repeats the same path.
            validate_control_result(result,control,args.instance_id)
            results[control]=result
            command_evidence.append({'path':str(path),'sha256':sha256_file(path)})
    if not results:raise ValueError('No completed control command found in the failed repair')
    return entry,results,command_evidence


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--instance-id',required=True)
    p.add_argument('--numpy-version');p.add_argument('--pytest-version')
    p.add_argument('--runtime-parent');p.add_argument('--evidence',required=True)
    p.add_argument('--resume-repair',help='Reuse an image/control from the recorded duplicate-run-path failure')
    a=p.parse_args(argv);pins=dependency_pins(a.numpy_version,a.pytest_version)
    setup_environment()
    with Run('task_environment_repair',vars(a)) as run:
        import docker,yaml
        if a.instance_id not in read_json('manifests/dev_tasks.json')['instance_ids']:
            raise ValueError('Task outside frozen development set')
        evidence=local_path(a.evidence)
        if not evidence.is_dir() or not list(evidence.rglob('test_output.txt')):
            raise ValueError('Preserve the failed control test_output.txt as repair evidence')
        require_storage();guard=StorageGuard();guard.check(2*10**9)
        task=yaml.safe_load((ROOT/'vendor/swe-bench-tasks/tasks'/a.instance_id/'task.yaml').read_text())
        original=task['image']
        record=run.command(['docker','image','inspect',original],timeout=20)
        source=json.loads(Path(record['stdout_path']).read_text())[0]
        manifest=read_json('manifests/image_overrides.json')
        if a.resume_repair:
            entry,reused_results,command_evidence=resume_candidate(a,pins,manifest,source)
            record=run.command(['docker','image','inspect',entry['image']],timeout=20)
            current=json.loads(Path(record['stdout_path']).read_text())[0]
            if current['Id']!=entry['image_id']:
                raise ValueError('Repaired image identity changed; pending repair preserved')
            client=docker.from_env(timeout=120)
            try:after=probe_versions(client,current['Id'],run,'resumed')
            finally:client.close()
            if any(after['packages'].get(name)!=version for name,version in pins.items()):
                raise ValueError('Pending repair no longer has the requested dependency versions')
            write_json(run.path/'previous_image_override.json',entry)
            entry.update(run=str(run.path),resumed_from=str(local_path(a.resume_repair)),
                         versions_after_resume=after,resumed_control_commands=command_evidence)
            manifest['overrides'][a.instance_id]=entry
            write_json('manifests/image_overrides.json',manifest)
            finish_controls(run,a,entry,manifest,guard,reused_results)
            return
        parent=a.runtime_parent or original;parent_info=source
        if a.runtime_parent:
            if not parent.startswith('cs6158/runtime:'): raise ValueError('Runtime parent must be a project image')
            record=run.command(['docker','image','inspect',parent],timeout=20)
            parent_info=json.loads(Path(record['stdout_path']).read_text())[0]
            layers=source['RootFS']['Layers']
            if parent_info['RootFS']['Layers'][:len(layers)]!=layers:
                raise ValueError('Runtime parent does not descend from the exact original task image')
        suffix=''.join('-'+name+'-'+version for name,version in pins.items())
        tag=('cs6158/runtime:repair-' if a.runtime_parent else 'cs6158/')+a.instance_id.lower()+suffix
        folder=run.path/'build';folder.mkdir()
        requirements=' '.join(name+'=='+version for name,version in pins.items())
        # Pin the inspected parent ID so a tag change cannot change the build input.
        (folder/'Dockerfile').write_text('FROM '+parent_info['Id']+'\nRUN /opt/miniconda3/envs/testbed/bin/python -m pip install --no-cache-dir '+requirements+'\n')
        client=docker.from_env(timeout=120)
        try:
            before=probe_versions(client,parent_info['Id'],run,'before')
            os.environ['DOCKER_BUILDKIT']='0'
            run.command(['docker','build','--tag',tag,str(folder)],guard=guard,timeout=600)
            record=run.command(['docker','image','inspect',tag],timeout=20)
            info=json.loads(Path(record['stdout_path']).read_text())[0]
            after=probe_versions(client,info['Id'],run,'after')
        finally: client.close()
        if any(after['packages'].get(name)!=version for name,version in pins.items()):
            raise ValueError('Repaired task image did not retain the requested dependency versions')
        previous=manifest['overrides'].get(a.instance_id)
        if previous is not None: write_json(run.path/'previous_image_override.json',previous)
        entry={'base_image':original,'base_image_id':source['Id'],
               'base_repo_digests':source.get('RepoDigests'),'parent_image':parent,
               'parent_image_id':parent_info['Id'],'image':tag,'image_id':info['Id'],
               'dependency_pins':pins,'versions_before':before,'versions_after':after,
               'reason':'Historical task dependency compatibility; testbed pins: '+requirements,
               'scope':'Dependency-only image layer; task source, tests and scoring unchanged',
               'status':'candidate_pending_controls','failed_control':a.evidence,'run':str(run.path)}
        if a.runtime_parent: entry.update(runtime_image=tag,runtime_image_id=info['Id'])
        manifest['overrides'][a.instance_id]=entry;write_json('manifests/image_overrides.json',manifest)
        finish_controls(run,a,entry,manifest,guard)


def finish_controls(run,args,entry,manifest,guard,reused_results=None):
    controls={};reused_results=reused_results or {}
    for control in ['gold','blank']:
        result=reused_results.get(control)
        if result is None:
            result=run.command([ROOT/'.envs/train/bin/python',ROOT/'scripts/study.py','grade',
                                '--control',control,'--instance-id',args.instance_id,
                                '--python','.envs/eval/bin/python'],guard=guard,timeout=1200)
        else:print('Reusing validated '+control+' control from '+str(args.resume_repair))
        controls[control]=validate_control_result(result,control,args.instance_id)
        entry['controls']=dict(controls)
        write_json('manifests/image_overrides.json',manifest)
    controls['blank_note']='Official harness short-circuits empty predictions; this checks denominator/empty handling, not an executed test-negative control.'
    entry.update(status='gold_and_blank_controls_passed',controls=controls)
    write_json('manifests/image_overrides.json',manifest);write_json(run.path/'repair_validation.json',entry)
    print(json.dumps(entry,indent=2))


if __name__=='__main__':main()
