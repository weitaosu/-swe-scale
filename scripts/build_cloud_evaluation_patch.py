#!/usr/bin/env python3
"""Build a self-contained, checked installer and explicit Git delivery file list."""
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from swe_study.audit import Run
from swe_study.common import write_json

NAMES=[
    'scripts/cloud.py','src/swe_study/docker_storage.py',
    'src/swe_study/runtime.py','src/swe_study/serving.py','src/swe_study/storage.py',
    'src/swe_study/memory_head.py','src/swe_study/benchmark.py','src/swe_study/validation.py','src/swe_study/container_options.py','src/swe_study/container_archives.py','src/swe_study/evaluation.py',
    'scripts/check_container_api.py','scripts/stop_container_service.py','scripts/cloud_benchmark.py','scripts/cloud_validation.py',
    'scripts/prepare_task_runtime.py','scripts/preserve_cloud_results.py','scripts/validate_cloud_evaluation.py','scripts/validate_distributed_validation.py','scripts/swebench_entry.py','scripts/validate_task_metadata.py','scripts/validate_podman_builder.py',
    'tests/test_cloud_benchmark.py','tests/test_cloud_validation.py','tests/test_container_service.py',
    'cloud/nersc/container_env.sh','cloud/nersc/container_service.sh','cloud/nersc/recover_containers.sh','cloud/nersc/containers-network.toml','cloud/nersc/podman-bin/docker',
    'cloud/EVALUATION_NERSC.md']


def sha(text):return hashlib.sha256(text.encode()).hexdigest()


def main():
    with Run('cloud_evaluation_patch_build') as run:
        target=ROOT/'deliverables/nersc_evaluation_patch.py'
        previous=runpy.run_path(str(target))['FILES'] if target.exists() else {}
        v3=runpy.run_path(str(ROOT/'deliverables/nersc_training_memory_patch_v3.py'))['FILES']
        payload={}
        for name in NAMES:
            allowed=set(previous.get(name,{}).get('allowed_before_sha256',[]))
            if name in previous:allowed.add(sha(previous[name]['text']))
            if name in v3:allowed.add(sha(v3[name]['text']))
            probe=subprocess.run(['git','show','HEAD:'+name],cwd=ROOT,capture_output=True,text=True)
            if probe.returncode==0:allowed.add(sha(probe.stdout))
            payload[name]={'text':(ROOT/name).read_text(),'allowed_before_sha256':sorted(allowed)}
        installer='''#!/usr/bin/env python3
"""Install cloud evaluation helpers; back up known sources and preserve unknown edits."""
import argparse,datetime,hashlib,json,os,tempfile
from pathlib import Path
FILES = json.loads(PAYLOAD)
def sha(text):return hashlib.sha256(text.encode()).hexdigest()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path)
    args=parser.parse_args();root=args.root.resolve()
    if not (root/'pyproject.toml').is_file() or not (root/'cloud/nersc/env.sh').is_file():
        raise SystemExit('Choose the extracted swe-study-a100 project root.')
    changes=[]
    for name,item in FILES.items():
        target=(root/name).resolve()
        if not target.is_relative_to(root):raise SystemExit('Target escaped project root')
        if name.endswith('.py'):compile(item['text'],name,'exec')
        old=target.read_text() if target.exists() else None
        if old==item['text']:continue
        if old is not None and sha(old) not in item['allowed_before_sha256']:
            raise SystemExit('Unfamiliar source preserved: '+name)
        if old is None and name in {'scripts/cloud.py','src/swe_study/docker_storage.py','src/swe_study/runtime.py','src/swe_study/serving.py','src/swe_study/storage.py'}:
            raise SystemExit('Required source missing: '+name)
        changes.append((name,target,old,item['text']))
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+str(os.getpid())
    backup=root/'.cache/evaluation-patch-backups'/stamp
    for name,target,old,text in changes:
        if old is not None:
            saved=backup/name;saved.parent.mkdir(parents=True,exist_ok=True);saved.write_text(old)
        target.parent.mkdir(parents=True,exist_ok=True)
        fd,temporary=tempfile.mkstemp(prefix=target.name+'.',suffix='.tmp',dir=target.parent)
        try:
            with os.fdopen(fd,'w',encoding='utf-8') as handle:handle.write(text)
            Path(temporary).replace(target)
        finally:Path(temporary).unlink(missing_ok=True)
        print('Installed',name)
    for name in FILES:
        if name.endswith('.sh') or name=='cloud/nersc/podman-bin/docker':os.chmod(root/name,0o755)
    log=root/'logs'/('evaluation-patch-'+stamp+'.json');log.parent.mkdir(parents=True,exist_ok=True)
    log.write_text(json.dumps({'changed':[name for name,*_ in changes],'backup':str(backup),
                              'installed_sha256':{name:sha(item['text']) for name,item in FILES.items()}},indent=2))
    print('Evaluation helpers installed. Start with cloud/EVALUATION_NERSC.md.')
    print('Installation log:',log)
if __name__=='__main__':main()
'''.replace('PAYLOAD',repr(json.dumps(payload)))
        compile(installer,'nersc_evaluation_patch.py','exec');target.write_text(installer)
        delivery=NAMES+['scripts/package_cloud.py','scripts/build_cloud_evaluation_patch.py','scripts/validate_evaluation_installer.py',
                        'deliverables/nersc_evaluation_patch.py','deliverables/nersc_evaluation_files.txt']
        (ROOT/'deliverables/nersc_evaluation_files.txt').write_text('\n'.join(delivery)+'\n')
        report={'installer':str(target.relative_to(ROOT)),'sha256':sha(installer),'bytes':target.stat().st_size,
                'files':NAMES,'note':'Container compatibility and real 3B results require NERSC execution'}
        write_json(run.path/'build.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
