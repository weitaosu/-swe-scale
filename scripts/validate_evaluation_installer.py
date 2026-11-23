#!/usr/bin/env python3
"""Audit installer source guards, backups, idempotence and shell syntax."""
import json
from pathlib import Path
import runpy
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from swe_study.audit import Run
from swe_study.common import setup_environment,write_json


def main():
    setup_environment()
    with Run('evaluation_installer_validation') as run:
        files=runpy.run_path(str(ROOT/'deliverables/nersc_evaluation_patch.py'))['FILES']
        project=ROOT/'.cache/tmp'/('evaluation-installer-'+run.id)
        (project/'cloud/nersc').mkdir(parents=True)
        (project/'cloud/nersc/env.sh').write_text('# fixture\n')
        (project/'pyproject.toml').write_text('# fixture\n')
        before={}
        for name,item in files.items():
            if name=='src/swe_study/memory_head.py':continue  # v2-to-evaluation upgrade creates it.
            if not item['allowed_before_sha256']:continue
            probe=subprocess.run(['git','show','HEAD:'+name],cwd=ROOT,capture_output=True,text=True)
            if probe.returncode:continue  # Newly added files may have prior installer hashes but no Git HEAD yet.
            before[name]=probe.stdout
            target=project/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(probe.stdout)
        command=[sys.executable,'deliverables/nersc_evaluation_patch.py','--root',str(project)]
        run.command(command,timeout=60)
        for name,item in files.items():assert (project/name).read_text()==item['text']
        for name,text in before.items():
            if text!=files[name]['text']:
                saved=list((project/'.cache/evaluation-patch-backups').glob('*/'+name))
                assert len(saved)==1 and saved[0].read_text()==text
        run.command(command,timeout=60)
        logs=[json.loads(p.read_text()) for p in (project/'logs').glob('evaluation-patch-*.json')]
        assert any(not log['changed'] for log in logs)
        early='scripts/cloud.py';(project/early).write_text(before[early])
        (project/'cloud/EVALUATION_NERSC.md').write_text('Unfamiliar operator edit\n')
        snapshot={name:(project/name).read_bytes() for name in files}
        try:run.command(command,timeout=60)
        except RuntimeError:pass
        else:raise AssertionError('Unknown source was not preserved')
        assert snapshot=={name:(project/name).read_bytes() for name in files}
        for name in ['cloud/nersc/container_env.sh','cloud/nersc/container_service.sh','cloud/nersc/podman-bin/docker']:
            run.command(['bash','-n',name],timeout=30)
            assert (project/name).stat().st_mode&0o111
        for name in (ROOT/'deliverables/nersc_evaluation_files.txt').read_text().splitlines():
            assert (ROOT/name).is_file(),name
        report={'passed':True,'backs_up_known_sources':True,'idempotent':True,
                'unknown_sources_rejected_before_writes':True,'shell_syntax_and_executable_modes':True,
                'v2_upgrade_without_head_module':'passed'}
        write_json(run.path/'validation.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
