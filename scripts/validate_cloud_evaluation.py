#!/usr/bin/env python3
"""Audit CPU protocol tests and optional one-GPU NF4 paired validation fixture."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from swe_study.audit import Run
from swe_study.common import setup_environment,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-fixture',type=Path)
    args=parser.parse_args();setup_environment()
    with Run('cloud_evaluation_validation',{'gpu_fixture':str(args.gpu_fixture) if args.gpu_fixture else None}) as run:
        run.command([sys.executable,'-m','pytest','tests/test_cloud_validation.py','tests/test_cloud_benchmark.py','tests/test_container_service.py','tests/test_context_window.py',
                     '-q','--capture=sys','-p','no:cacheprovider','--basetemp',ROOT/'.cache/tmp'/('pytest-cloud-eval-'+run.id)],timeout=300)
        report={'passed':True,'cpu_validation':'passed','synthetic_fixture':True,
                'limitation':'No NERSC Podman service or real 43-history model result is claimed'}
        if args.gpu_fixture:
            source=args.gpu_fixture.resolve()
            if not source.is_relative_to(ROOT/'.cache/tmp'): raise ValueError('Use a project-local isolated fixture')
            adapter=source/'checkpoints/step-resume';metadata=json.loads((adapter/'experiment.json').read_text())
            if not metadata.get('synthetic_fixture'): raise ValueError('This diagnostic accepts random tiny models only')
            spec=importlib.util.spec_from_file_location('package_cloud',ROOT/'scripts/package_cloud.py')
            package=importlib.util.module_from_spec(spec);spec.loader.exec_module(package)
            project=ROOT/'.cache/tmp'/('cloud-evaluation-'+run.id)
            for name,data in package.bundle_files().items():
                target=project/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
            cfg=metadata['config']
            shutil.copytree(source/cfg['model'],project/cfg['model'])
            for field in ('train_file','validation_file'):
                target=project/cfg[field];target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/cfg[field],target)
            shutil.copytree(adapter,project/'checkpoints/step-resume')
            run.command([sys.executable,'scripts/cloud.py','init','--storage-accounting','project'],cwd=project)
            result=run.command([sys.executable,'scripts/cloud_validation.py','--adapter','checkpoints/step-resume','--allow-synthetic'],cwd=project,timeout=600)
            paths=list((project/'results/cloud_validation').glob('*/comparison.json'))
            if len(paths)!=1: raise AssertionError('Missing paired comparison report')
            comparison=json.loads(paths[0].read_text())
            if not comparison['adapter_matches_recorded_loss']: raise AssertionError('Paired adapter loss did not reproduce training')
            report['one_gpu_nf4_paired_evaluation']={'passed':True,'examples':comparison['examples'],
                       'base_loss':comparison['base_validation_loss'],'adapter_loss':comparison['adapter_validation_loss'],
                       'project':str(project.relative_to(ROOT))}
            run.command([sys.executable,'scripts/preserve_cloud_results.py','--adapter','checkpoints/step-resume','--archive'],cwd=project,timeout=300)
            import hashlib,tarfile
            archives=list((project/'results/cloud_preservation').glob('*/preservation.tar'))
            if len(archives)!=1: raise AssertionError('Expected one preservation archive')
            with tarfile.open(archives[0]) as archive:
                manifests=[member for member in archive.getmembers() if member.name.endswith('/preservation_manifest.json')]
                preserved=json.load(archive.extractfile(manifests[0]))
                for item in preserved['files']:
                    stream=archive.extractfile('swe-study-a100/'+item['path'])
                    if hashlib.sha256(stream.read()).hexdigest()!=item['sha256']:
                        raise AssertionError('Preservation archive checksum mismatch')
            report['self_contained_archive_checksums']='passed'
        write_json(run.path/'validation.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
