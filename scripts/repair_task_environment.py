#!/usr/bin/env python3
"""Create and control-test a dependency-only repair for a frozen development task."""
from pathlib import Path
import argparse,os,re,sys,json
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from swe_study.common import setup_environment,read_json,write_json
setup_environment()
from swe_study.audit import Run
from swe_study.storage import StorageGuard
from swe_study.docker_storage import require_storage
p=argparse.ArgumentParser();p.add_argument('--instance-id',required=True);p.add_argument('--numpy-version',required=True);p.add_argument('--runtime-parent');p.add_argument('--evidence',required=True);a=p.parse_args()
with Run('task_environment_repair',vars(a)) as run:
    import yaml
    if a.instance_id not in read_json('manifests/dev_tasks.json')['instance_ids']: raise ValueError('Task outside frozen development set')
    if not re.fullmatch(r'[0-9]+(?:[.][0-9]+)+',a.numpy_version): raise ValueError('Expected exact NumPy version')
    require_storage();guard=StorageGuard();guard.check(2*10**9)
    task=yaml.safe_load((root/'vendor/swe-bench-tasks/tasks'/a.instance_id/'task.yaml').read_text());original=task['image']
    info_record=run.command(['docker','image','inspect',original],timeout=20)
    source=json.loads(Path(info_record['stdout_path']).read_text())[0]
    parent=a.runtime_parent or original
    if a.runtime_parent:
        if not parent.startswith('cs6158/runtime:'): raise ValueError('Runtime parent must be a project image')
        record=run.command(['docker','image','inspect',parent],timeout=20)
        parent_info=json.loads(Path(record['stdout_path']).read_text())[0]
        layers=source['RootFS']['Layers']
        if parent_info['RootFS']['Layers'][:len(layers)]!=layers: raise ValueError('Runtime parent does not descend from the exact original task image')
    else: parent_info=source
    tag=('cs6158/runtime:repair-' if a.runtime_parent else 'cs6158/')+a.instance_id.lower()+'-numpy-'+a.numpy_version
    folder=run.path/'build';folder.mkdir()
    (folder/'Dockerfile').write_text('FROM '+parent+'\nRUN /opt/miniconda3/envs/testbed/bin/python -m pip install --no-cache-dir numpy=='+a.numpy_version+'\n')
    os.environ['DOCKER_BUILDKIT']='0'
    run.command(['docker','build','--tag',tag,str(folder)],guard=guard,timeout=600)
    record=run.command(['docker','image','inspect',tag],timeout=20)
    info=json.loads(Path(record['stdout_path']).read_text())[0]
    manifest=read_json('manifests/image_overrides.json')
    entry={'base_image':original,'base_image_id':source['Id'],'base_repo_digests':source.get('RepoDigests'),'parent_image':parent,'parent_image_id':parent_info['Id'],'image':tag,'image_id':info['Id'],'reason':'Historical task cannot collect under NumPy 2; only testbed NumPy is pinned to '+a.numpy_version,'status':'candidate_pending_controls','failed_control':a.evidence,'run':str(run.path)}
    if a.runtime_parent: entry.update(runtime_image=tag,runtime_image_id=info['Id'])
    manifest['overrides'][a.instance_id]=entry;write_json('manifests/image_overrides.json',manifest)
    controls={}
    for control in ['gold','blank']:
        result=run.command([root/'.envs/train/bin/python',root/'scripts/study.py','grade','--control',control,'--instance-id',a.instance_id,'--python','.envs/eval/bin/python'],guard=guard,timeout=1200)
        paths=[Path(line[len('Run log: '):]) for line in Path(result['stdout_path']).read_text().splitlines() if line.startswith('Run log: '+str(root/'results/grade')+'/')]
        if not paths: raise ValueError('Control did not record its run')
        controls[control]=str(paths[0]/'grade_validation.json')
    controls['blank_note']='Official harness short-circuits empty predictions; this checks denominator/empty handling, not an executed test-negative control.'
    entry.update(status='gold_and_blank_controls_passed',controls=controls)
    write_json('manifests/image_overrides.json',manifest);write_json(run.path/'repair_validation.json',entry)
