#!/usr/bin/env python3
"""Reuse verified, model-independent task preparation without changing shared caches."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from swe_study.audit import Run
from swe_study.common import local_path,read_json,setup_environment,sha256_file,write_json
from swe_study.benchmark import initialize_study,record_path
from swe_study.runtime import require_image_identity


def reuse(run,client,source,state,suite,key,identity):
    source=local_path(source);state=local_path(state)
    if source==state: raise ValueError('Choose distinct source and destination studies')
    original_protocol=read_json(source/'protocol.json')
    model_fields={'adapter','model','adapter_weights_sha256'}
    if ({k:v for k,v in original_protocol.items() if k not in model_fields}!=
            {k:v for k,v in identity.items() if k not in model_fields}):
        raise ValueError('Task preparation protocols differ beyond model and adapter')
    old=record_path(source,suite,key);target=record_path(state,suite,key)
    item=read_json(old)
    if item.get('instance_id')!=key or item.get('passed') is not True or item.get('gold_passed') is not True:
        raise ValueError('Source task preparation did not pass')
    for field,control in [('gold_control','gold'),('blank_control','blank')]:
        result=read_json(local_path(item[field])/'grade_validation.json')
        if (result.get('passed') is not True or result.get('control')!=control or result.get('total_instances')!=1
                or result.get('resolved_ids')!=([key] if control=='gold' else [])):
            raise ValueError('Missing or incompatible source '+control+' control')
    runtime=read_json(local_path(item['runtime_run'])/'runtime.json')
    cache=read_json('manifests/runtime_cache.json').get(key)
    if runtime.get('passed') is not True or runtime.get('instance_id')!=key or not cache:
        raise ValueError('Source runtime check or shared cache is unavailable')
    for field in ('base_image','runtime_image'):
        if runtime[field]!=item[field] or cache[field]!=item[field]: raise ValueError('Runtime mapping differs')
    for field,reference in [('base_image_id',item['base_image']),('runtime_image_id',item['runtime_image'])]:
        require_image_identity(client.images.get(reference).id,item[field],'Source image changed or missing')
    for record in (runtime,cache):
        require_image_identity(record['runtime_image_id'],item['runtime_image_id'],'Source runtime identity differs')
    if target.exists():
        current=read_json(target)
        for field in ('base_image_id','runtime_image_id'):
            require_image_identity(current[field],item[field],'Destination preparation differs')
        print('Already prepared:',key);return
    if any(record_path(state,suite,key,c).exists() for c in ('base','adapter')):
        raise ValueError('Destination has an agent attempt without preparation; preserve it for review')
    copied={**item,'new_image_ids':[],
            'reused_preparation':{'source':str(old.relative_to(ROOT)),'source_sha256':sha256_file(old),
                                  'review_run':str(run.path.relative_to(ROOT)),
                                  'scope':'Environment and controls only; no predictions, metrics or token ledgers copied'}}
    write_json(target,copied)
    run.event('task_preparation_reused',instance_id=key,source=str(old),destination=str(target))
    print('Reused prepared environment:',key)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-state',required=True);parser.add_argument('--state',required=True)
    parser.add_argument('--adapter',required=True)
    parser.add_argument('--suite',choices=['development','final'],default='development')
    parser.add_argument('--max-tasks',type=int,default=1)
    args=parser.parse_args();setup_environment()
    if args.max_tasks<1:parser.error('--max-tasks must be positive')
    from cloud_benchmark import study_lock
    from swe_study.docker_storage import require_storage
    import docker
    with Run('benchmark_preparation_reuse',vars(args)) as run,study_lock(args.state):
        identity,rows,_=initialize_study(args.state,args.adapter,args.suite)
        require_storage();client=docker.from_env(timeout=60)
        try:
            for row in rows[:args.max_tasks]:reuse(run,client,args.from_state,args.state,args.suite,row['instance_id'],identity)
        finally:client.close()
        print('Preparation reused; keep shared task images until all borrowing studies finish.')


if __name__=='__main__':main()
