#!/usr/bin/env python3
"""Stage CPU preparation, paired GPU-backed agent inference and official CPU grading."""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import urllib.request

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from swe_study.audit import Run
from swe_study.common import local_path,read_json,read_jsonl,setup_environment,write_json,write_jsonl
from swe_study.benchmark import initialize_study,record_path,study_summary,grade_one
from swe_study.evaluation import public_instance,merge_grading_metadata,export_predictions
from swe_study.storage import StorageGuard


@contextmanager
def study_lock(state):
    folder=ROOT/'.cache/benchmark_locks';folder.mkdir(parents=True,exist_ok=True)
    path=folder/(hashlib.sha256(str(local_path(state)).encode()).hexdigest()+'.lock')
    with path.open('a') as handle:
        try: fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('Another process owns this benchmark study') from None
        yield


def child_path(record,stage):
    prefix='Run log: '+str(ROOT/'results'/stage)+'/'
    paths=[Path(line[len('Run log: '):]) for line in Path(record['stdout_path']).read_text().splitlines() if line.startswith(prefix)]
    if len(paths)!=1: raise ValueError('Expected one audited '+stage+' child run')
    return local_path(paths[0])


def containers_for_run(run_id):
    import docker
    client=docker.from_env(timeout=60)
    try:
        for container in client.containers.list(all=True,filters={'label':'cs6158.condition_id='+run_id}):
            if not container.name.startswith(('openhands-runtime-cs6158-','sweb.eval.')):
                raise ValueError('Unexpected labelled container preserved')
            container.remove(force=True)
    finally: client.close()


def prepare(run,row,state,suite):
    from swe_study.docker_storage import require_storage
    require_storage();guard=StorageGuard()
    import docker,yaml
    client=docker.from_env(timeout=120)
    try:
        key=row['instance_id'];path=record_path(state,suite,key)
        previous=read_json(path) if path.exists() else None
        if previous:
            try:
                for image_id in [previous['base_image_id'],previous['runtime_image_id']]: client.images.get(image_id)
                print('Already prepared:',key);return
            except docker.errors.ImageNotFound: pass
        for condition in ('base','adapter'):
            attempt=record_path(state,suite,key,condition)
            if attempt.exists():
                raise ValueError('A task with recorded predictions lost its prepared images; preserve/restore the exact image IDs before continuing')
        task=yaml.safe_load((ROOT/'vendor/swe-bench-tasks/tasks'/key/'task.yaml').read_text())
        merge_grading_metadata(row,task)
        overrides=ROOT/'manifests/image_overrides.json'
        if not overrides.exists(): write_json(overrides,{'overrides':{}})
        repair=read_json(overrides)['overrides'].get(key,{})
        if repair and repair.get('status')!='gold_and_blank_controls_passed':
            raise ValueError('Candidate image repair requires passing controls before reuse')
        image=repair.get('image',task['image'])
        before={item.id for item in client.images.list(all=True)}
        write_json(run.path/'preparation_images_before.json',{'instance_id':key,'image_ids':sorted(before)})
        try: base=client.images.get(image)
        except docker.errors.ImageNotFound:
            guard.check(15*10**9)
            run.command(['docker','pull',image],guard=guard,timeout=1800)
            base=client.images.get(image)
        gold={'instance_id':key,'model_name_or_path':'gold-control','model_patch':row['patch']}
        with Run('benchmark_gold_control',{'instance_id':key,'suite':suite}) as gold_run:
            try: control=grade_one(gold_run,row,gold,'gold')
            finally: containers_for_run(gold_run.id)
        with Run('benchmark_blank_control',{'instance_id':key,'suite':suite}) as blank_run:
            blank={**gold,'model_name_or_path':'blank-control','model_patch':''}
            try: grade_one(blank_run,row,blank,'blank')
            finally: containers_for_run(blank_run.id)
        cache_path=ROOT/'manifests/runtime_cache.json'
        cache=read_json(cache_path) if cache_path.exists() else {}
        # Cache entries are node/engine dependent; an absent image is rebuilt
        # before any predictions and never silently substituted mid-condition.
        cache.pop(key,None);write_json(cache_path,cache)
        public=run.path/'public_task.json';write_json(public,[public_instance(row)])
        result=run.command([ROOT/'.envs/agent/bin/python',ROOT/'scripts/prepare_task_runtime.py',
                            '--task-file',public],guard=guard,timeout=3600)
        runtime_run=child_path(result,'benchmark_runtime_prepare')
        runtime=read_json(runtime_run/'runtime.json')
        cache[key]={k:runtime[k] for k in ('base_image','runtime_image','runtime_image_id')}
        write_json(cache_path,cache)
        after={item.id for item in client.images.list(all=True)}
        item={**runtime,'base_image_id':base.id,'base_repo_digests':base.attrs.get('RepoDigests'),
              'gold_control':str(gold_run.path.relative_to(ROOT)),'gold_passed':control['passed'],
              'blank_control':str(blank_run.path.relative_to(ROOT)),
              'new_image_ids':sorted(after-before),'runtime_run':str(runtime_run.relative_to(ROOT))}
        write_json(path,item)
        print('Prepared:',key)
    finally: client.close()


def predict(run,row,state,suite,identity,base_url):
    key=row['instance_id'];prepared=record_path(state,suite,key)
    if not prepared.exists(): raise ValueError('Prepare this task on CPU before requesting GPU inference: '+key)
    import docker
    client=docker.from_env(timeout=60)
    try:
        entry=read_json(prepared)
        for field in ('base_image_id','runtime_image_id'): client.images.get(entry[field])
    finally: client.close()
    with urllib.request.urlopen(base_url.removesuffix('/v1')+'/health',timeout=30) as response:
        health=json.load(response)
    cfg=identity['agent_config']
    import re
    from swe_study.common import canonical_hash
    tasks={entry['instance_id']:re.sub(r'\s+',' ',entry['problem_statement']).strip()
           for _,entry in read_jsonl(local_path(state)/suite/'task_budgets.jsonl')}
    if (not health.get('paired') or health.get('adapter_weights_sha256')!=identity['adapter_weights_sha256']
            or health.get('quantization')!='nf4' or health.get('task_set_sha256')!=canonical_hash(tasks)
            or health['max_context']!=cfg['max_context_tokens']
            or health.get('task_token_budget')!=cfg['max_generated_tokens_per_task']
            or health.get('max_new_tokens')!=cfg['max_new_tokens']):
        raise ValueError('Paired model server does not match the frozen study protocol')
    import time
    deadline=health.get('allocation_deadline_unix')
    remaining_conditions=sum(not record_path(state,suite,key,condition).exists() for condition in ('base','adapter'))
    if deadline and deadline-time.time()<remaining_conditions*cfg['task_timeout_seconds']+300:
        raise ValueError('GPU allocation has insufficient time for the remaining complete task episodes; obtain a new allocation')
    for condition,suffix in [('base','-base'),('adapter','-lora')]:
        path=record_path(state,suite,key,condition)
        if path.exists():
            saved=read_json(path)
            if saved['status'] in ('predicted','graded'): continue
            raise ValueError('An interrupted attempt is preserved for review; no automatic resampling: '+str(path))
        model=health['model']+suffix
        write_json(path,{'instance_id':key,'condition':condition,'status':'started','run':str(run.path.relative_to(ROOT))})
        with Run('benchmark_agent',{'instance_id':key,'condition':condition,'suite':suite}) as agent_run:
            os.environ['CS6158_CONDITION_ID']=agent_run.id
            public=agent_run.path/'agent_tasks.jsonl';write_jsonl(public,[public_instance(row)])
            config=agent_run.path/'config.toml'
            text='[llm.cs6158]\nmodel = '+json.dumps('openai/'+model)+'\nbase_url = '+json.dumps(base_url)+'\napi_key = "local-only"\ntemperature = 0.0\nlog_completions = true\nmax_input_tokens = '+str(cfg['max_context_tokens']-cfg['max_new_tokens'])+'\nmax_output_tokens = '+str(cfg['max_new_tokens'])+'\n'
            config.write_text(text)
            try:
                agent_run.command([ROOT/'.envs/agent/bin/python',ROOT/'scripts/openhands_entry.py',
                                   '--tasks',public,'--output',agent_run.path/'openhands','--run-id',agent_run.id,
                                   '--llm-config-file',config,'--max-steps',str(cfg['max_steps']),
                                   '--timeout',str(cfg['task_timeout_seconds'])],cwd=ROOT/'vendor/SWE-Gym-OpenHands',
                                   guard=StorageGuard(),timeout=cfg['task_timeout_seconds']+300)
                outputs=list((agent_run.path/'openhands').rglob('output.jsonl'))
                if len(outputs)!=1: raise ValueError('Expected one OpenHands output artifact')
                predictions=agent_run.path/'predictions.jsonl'
                exported=export_predictions(agent_run,outputs[0],predictions,model,{key})
                rows=list(read_jsonl(outputs[0]))
                if len(rows)!=1 or rows[0][1]['instance_id']!=key: raise ValueError('Agent task coverage mismatch')
                write_json(path,{'instance_id':key,'condition':condition,'status':'predicted',
                                 'prediction':exported[0],'predictions_file':str(predictions.relative_to(ROOT)),
                                 'agent_error':rows[0][1].get('error'),'agent_run':str(agent_run.path.relative_to(ROOT)),
                                 'prepared_image_identity':entry})
            except BaseException as exc:
                write_json(path,{'instance_id':key,'condition':condition,'status':'interrupted',
                                 'error':str(exc),'agent_run':str(agent_run.path.relative_to(ROOT))})
                raise
            finally: containers_for_run(agent_run.id)
        os.environ.pop('CS6158_CONDITION_ID',None)


def grade(run,row,state,suite):
    key=row['instance_id']
    import docker
    client=docker.from_env(timeout=60)
    try: client.images.get(read_json(record_path(state,suite,key))['base_image_id'])
    finally: client.close()
    for condition in ('base','adapter'):
        path=record_path(state,suite,key,condition)
        if not path.exists(): raise ValueError('No saved prediction for '+key+' / '+condition)
        item=read_json(path)
        if item['status']=='graded': continue
        if item['status']!='predicted': raise ValueError('Incomplete agent episode requires review')
        with Run('benchmark_grade',{'instance_id':key,'condition':condition,'suite':suite}) as grade_run:
            try: summary=grade_one(grade_run,row,item['prediction'])
            finally: containers_for_run(grade_run.id)
            item.update(status='graded',resolved=key in summary['resolved_ids'],grader=str(grade_run.path.relative_to(ROOT)))
            write_json(path,item)


def cleanup(run,row,state,suite):
    key=row['instance_id'];path=record_path(state,suite,key)
    if not path.exists(): return
    for condition in ('base','adapter'):
        result=record_path(state,suite,key,condition)
        if not result.exists() or read_json(result).get('status')!='graded':
            raise ValueError('Keep task images until both conditions have been graded')
    import docker
    client=docker.from_env(timeout=120)
    try:
        item=read_json(path);removed=[]
        for image_id in sorted(item['new_image_ids'],key=lambda key:key!=item['runtime_image_id']):
            if os.environ.get('SWE_STUDY_CONTAINER_ENGINE')=='podman-hpc':
                try: run.command(['podman-hpc','rmsqi',image_id],timeout=120)
                except RuntimeError as exc: run.event('squashed_image_preserved',image_id=image_id,error=str(exc))
            try:
                client.images.remove(image_id,force=False,noprune=True);removed.append(image_id)
            except docker.errors.ImageNotFound: continue
            except docker.errors.APIError as exc: run.event('image_preserved',image_id=image_id,error=str(exc))
        item['removed_image_ids']=removed;write_json(path,item)
        print('Removed only newly created, unused task images:',removed)
    finally: client.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['plan','prepare','predict','grade','summary','cleanup'])
    parser.add_argument('--suite',choices=['development','final'],default='development')
    parser.add_argument('--state',default='results/benchmark_3b')
    parser.add_argument('--adapter',default='checkpoints/cloud/qwen-3b-32k-1gpu-step-resume-v1')
    parser.add_argument('--instance-id')
    parser.add_argument('--max-tasks',type=int,default=1)
    parser.add_argument('--base-url',default='http://127.0.0.1:8615/v1')
    args=parser.parse_args();setup_environment()
    if args.max_tasks<1: parser.error('--max-tasks must be positive')
    with Run('cloud_benchmark_'+args.command,vars(args)) as run,study_lock(args.state):
        identity,rows,budgets=initialize_study(args.state,args.adapter,args.suite)
        if args.instance_id:
            rows=[row for row in rows if row['instance_id']==args.instance_id]
            if not rows: raise ValueError('Requested task is outside the frozen suite')
        if args.command in ('plan','summary'):
            ids=identity['frozen_development_ids' if args.suite=='development' else 'frozen_final_ids']
            summary=study_summary(args.state,args.suite,ids)
            write_json(local_path(args.state)/args.suite/'summary.json',summary)
            print(json.dumps(summary,indent=2));print('Task budgets:',budgets)
            return
        from swe_study.docker_storage import require_storage
        require_storage();done=0
        for row in rows:
            key=row['instance_id']
            statuses=[read_json(record_path(args.state,args.suite,key,condition)).get('status')
                      if record_path(args.state,args.suite,key,condition).exists() else None
                      for condition in ('base','adapter')]
            if args.command in ('prepare','predict','grade') and statuses==['graded','graded']: continue
            if args.command=='prepare' and record_path(args.state,args.suite,key).exists():
                # prepare() also checks whether the cached image IDs still exist.
                pass
            if args.command=='predict' and all(status in ('predicted','graded') for status in statuses): continue
            if args.command=='grade' and not all(status in ('predicted','graded') for status in statuses): continue
            if args.command=='cleanup' and statuses!=['graded','graded']: continue
            globals()[args.command](run,row,args.state,args.suite,*([identity,args.base_url] if args.command=='predict' else []))
            done+=1
            if done>=args.max_tasks: break
        print('Tasks processed:',done)


if __name__=='__main__':main()
