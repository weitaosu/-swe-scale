#!/usr/bin/env python3
"""Read-only review of the image-ID guard failure before any agent episode."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from swe_study.audit import Run
from swe_study.common import local_path,read_json,setup_environment,write_json
from swe_study.runtime import require_image_identity


def review(state,agent_run,instance_id,client):
    state=local_path(state);agent_run=local_path(agent_run)
    task=state/'development/tasks'/instance_id
    failed=read_json(task/'base.json')
    if (failed.get('status')!='interrupted' or
            local_path(failed.get('agent_run',''))!=agent_run or (task/'adapter.json').exists()):
        raise ValueError('Expected only the interrupted base startup; preserve other episodes for review')
    metadata=read_json(agent_run/'run.json')
    if metadata.get('status')!='failed' or metadata.get('stage')!='benchmark_agent':
        raise ValueError('Expected the failed audited agent child run')
    stderr=(agent_run/'command-001.stderr.log').read_text()
    match=re.search(r'File "[^"\n]*/scripts/openhands_entry.py", line (\d+), in <module>\n\s+configure_task_images\(',stderr)
    if not match or 'Prepared runtime image identity changed' not in stderr:
        raise ValueError('Failure is not the recognized image-ID startup guard')
    snapshot=agent_run/'source_snapshot.zip'
    with zipfile.ZipFile(snapshot) as archive:
        source=archive.read('scripts/openhands_entry.py').decode()
        old_runtime=archive.read('src/swe_study/runtime.py')
    lines=source.splitlines();line=int(match.group(1))
    if ('configure_task_images(' not in lines[line-1] or
            source.index('configure_task_images(rows,')>source.index('runpy.run_path(')):
        raise ValueError('Cannot establish that startup failed before evaluator invocation')
    protocol=read_json(state/'protocol.json')
    if hashlib.sha256(old_runtime).hexdigest()!=protocol['runtime_overlay_sha256']:
        raise ValueError('Failed-run source does not match its frozen runtime protocol')
    if source!=(ROOT/'scripts/openhands_entry.py').read_text():
        raise ValueError('Agent entry changed; requires a separate review')
    if list(agent_run.rglob('output.jsonl')) or list(agent_run.rglob('llm_completions')):
        raise ValueError('Agent output exists; do not treat this as an unused episode')
    for condition in ('base','adapter'):
        ledger=state/'development/token_ledgers'/(condition+'_task_tokens.json')
        if ledger.exists() and read_json(ledger)['completion_tokens_by_task'].get(instance_id,0)!=0:
            raise ValueError('Task has generated tokens; preserve its budget and review the episode')
    prepared=read_json(task/'prepared.json')
    cache=read_json('manifests/runtime_cache.json')[instance_id]
    for field,reference in [('base_image_id',prepared['base_image']),
                            ('runtime_image_id',prepared['runtime_image'])]:
        require_image_identity(client.images.get(reference).id,prepared[field],'Prepared image changed')
    require_image_identity(cache['runtime_image_id'],prepared['runtime_image_id'],'Runtime cache changed')
    require_image_identity(client.images.get(cache['runtime_image']).id,cache['runtime_image_id'],'Cached image changed')
    return {'passed':True,'instance_id':instance_id,'failed_agent_run':str(agent_run.relative_to(ROOT)),
            'failed_record_sha256':hashlib.sha256((task/'base.json').read_bytes()).hexdigest(),
            'scope':'Recognized failure before evaluator invocation; no recorded output or generated task tokens',
            'recovery':'Use a new study state for the corrected runtime; preserve the old state and existing server token ledgers',
            'records_modified':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state',required=True);parser.add_argument('--agent-run',required=True)
    parser.add_argument('--instance-id',default='pvlib__pvlib-python-807')
    args=parser.parse_args();setup_environment()
    with Run('benchmark_prestart_review',vars(args)) as run:
        from swe_study.docker_storage import require_storage
        import docker
        require_storage();client=docker.from_env(timeout=60)
        try: report=review(args.state,args.agent_run,args.instance_id,client)
        finally: client.close()
        write_json(run.path/'review.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
