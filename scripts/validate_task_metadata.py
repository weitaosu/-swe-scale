#!/usr/bin/env python3
"""Check all frozen dataset/task identities without containers, models or GPUs."""
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from swe_study.audit import Run
from swe_study.benchmark import suite_rows
from swe_study.common import read_json,setup_environment,sha256_file,write_json
from swe_study.evaluation import load_grading_task,merge_grading_metadata


def check_task_files(run,repo,rows):
    names=('task.yaml','gold.patch','test.patch','problem_statement.md','tests.json','eval.sh')
    paths=['tasks/'+row['instance_id']+'/'+name for row in rows for name in names]
    requests=''.join('HEAD:'+name+'\n' for name in paths).encode()
    (run.path/'task_blob_requests.txt').write_bytes(requests)
    command=['git','cat-file','--batch']
    run.event('task_blob_check_started',args=command,cwd=str(repo),files=len(paths))
    result=subprocess.run(command,cwd=repo,input=requests,capture_output=True,timeout=120,check=True)
    (run.path/'task_blob_check.stderr.log').write_bytes(result.stderr)
    offset=0
    for name in paths:
        end=result.stdout.index(b'\n',offset)
        header=result.stdout[offset:end].split()
        if len(header)!=3 or header[1]!=b'blob': raise ValueError('Missing pinned task file: '+name)
        size=int(header[2]);start=end+1;expected=result.stdout[start:start+size]
        actual=(repo/name).read_bytes()
        if actual.replace(b'\r\n',b'\n')!=expected.replace(b'\r\n',b'\n'):
            raise ValueError('Task file differs from pinned Git revision: '+name)
        offset=start+size+1
    if offset!=len(result.stdout): raise ValueError('Unexpected Git batch response')
    run.event('task_blob_check_passed',files=len(paths),normalization='CRLF to LF only')


def main():
    setup_environment()
    with Run('task_metadata_validation') as run:
        rows_by_suite={suite:suite_rows(suite) for suite in ('development','final')}
        repositories=read_json('manifests/source_repositories.json')['repositories']
        revisions={}
        for name in ('SWE-bench','swe-bench-tasks'):
            entry=repositories[name];repo=ROOT/entry['path']
            result=run.command(['git','rev-parse','HEAD'],cwd=repo,timeout=30)
            head=Path(result['stdout_path']).read_text().strip()
            if head!=entry['commit']: raise ValueError(name+' differs from its pinned revision')
            if name=='SWE-bench':
                run.command(['git','diff','--quiet','HEAD','--','swebench/__init__.py','swebench/task'],cwd=repo,timeout=120)
            else:
                check_task_files(run,repo,[row for rows in rows_by_suite.values() for row in rows])
            revisions[name]=head
        checked=[];failures=[]
        for suite in ('development','final'):
            for row in rows_by_suite[suite]:
                key=row['instance_id']
                try:
                    task=load_grading_task(key)
                    merged=merge_grading_metadata(row,task)
                    folder=ROOT/'vendor/swe-bench-tasks/tasks'/key
                    checked.append({'suite':suite,'instance_id':key,'image':merged['image'],
                        'files_sha256':{name:sha256_file(folder/name) for name in
                            ('task.yaml','gold.patch','test.patch','problem_statement.md','tests.json','eval.sh')}})
                except (ValueError,KeyError,OSError) as exc:
                    failures.append({'suite':suite,'instance_id':key,'error':str(exc)})
        report={'passed':not failures,'checked':len(checked),
                'development_checked':sum(item['suite']=='development' for item in checked),
                'final_checked':sum(item['suite']=='final' for item in checked),
                'revisions':revisions,'revision_check_scope':'Official loader source and six consumed metadata files per frozen task',
                'failures':failures,'tasks':checked,
                'limitation':'Metadata compatibility only; no gold tests or agent episodes executed'}
        write_json(run.path/'validation.json',report)
        print(json.dumps({key:value for key,value in report.items() if key!='tasks'},indent=2))
        if failures: raise ValueError('Frozen task metadata failed identity checks; see validation.json')


if __name__=='__main__':main()
