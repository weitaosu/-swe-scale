from pathlib import Path
import sys,json,time,zipfile,re,shutil,os
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'src'))
from swe_study.common import setup_environment,read_json,write_json,sha256_file
setup_environment()
from swe_study.audit import Run,utc
from swe_study.storage import StorageGuard
review=read_json('results/setup/cleanup_review_latest.json');plan_path=root/review['plan'];plan=read_json(plan_path)
with Run('cleanup',{'requested_by':'user','plan':str(plan_path),'plan_sha256':sha256_file(plan_path),'preserve_logs_models_data_checkpoints':True}) as run:
    shutil.copy2(__file__,run.path/'workflow.py');shutil.copy2(plan_path,run.path/'deletion_plan.json');guard=StorageGuard();before=guard.snapshot()
    write_json(root/review['review']/'active_run.json',{'run':str(run.path.relative_to(root))})
    def cmd(args,timeout=120):return run.command(args,timeout=timeout,guard=guard)
    def output(record):return Path(record['stdout_path']).read_text()
    containers_before=output(cmd(['docker','ps','-a','--no-trunc','--format','{{json .}}']))
    (run.path/'containers_before.jsonl').write_text(containers_before)
    images_before=set(output(cmd(['docker','image','ls','--no-trunc','--quiet'])).splitlines())
    cached_before=output(cmd(['docker','--config','.cache/docker_agent','buildx','du','--verbose']))
    cache_ids_before=set(re.findall(r'^ID:\s+(\S+)',cached_before,re.M))
    image=plan['docker_obsolete_image_id'];cache_id=plan['docker_cache_id']
    details=json.loads(output(cmd(['docker','image','inspect',image])))[0]
    tags=details.get('RepoTags') or []
    if details['Id']!=image or set(tags)!={'cs6158/runtime:oh_v0.13.1_r7tyogpmfbg9q3z8','cs6158/runtime:oh_v0.13.1_r7tyogpmfbg9q3z8_y766p9z6jyxsud5c'}:raise ValueError('Obsolete image identity or references changed')
    if output(cmd(['docker','ps','-a','--filter','ancestor='+image,'--format','{{.ID}}'])).strip():raise ValueError('Obsolete image is referenced by a container')
    repairs=read_json('manifests/image_overrides.json')['overrides']
    protected_images={value for entry in repairs.values() for key,value in entry.items() if key in ['image_id','runtime_image_id','parent_image_id','base_image_id']}
    if image in protected_images:raise ValueError('Image is required by validated runtime provenance')
    target=output(cmd(['docker','--config','.cache/docker_agent','buildx','du','--verbose','--filter','id='+cache_id]))
    if re.findall(r'^ID:\s+(\S+)',target,re.M)!=[cache_id] or 'Browsing disabled for this local pilot' not in target or 'Reclaimable:\ttrue' not in target:raise ValueError('Cache target is not the failed project runtime')
    if output(cmd(['docker','--config','.cache/docker_agent','buildx','du','--verbose','--filter','id=cs6158-cleanup-nonexistent-id'])).strip():raise ValueError('Builder cache filtering is not reliable')
    cmd(['docker','history','--no-trunc',image])
    with zipfile.ZipFile(run.path/'temporary_files_recovery.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        for entry in plan['files']:
            if entry['path'].startswith('.cache/tmp/') or entry['path']=='.envs/validation/pyvenv.cfg':
                p=root/entry['path']
                if p.is_file() and not p.is_symlink():archive.write(p,entry['path'])
    write_json(root/review['review']/'archive_ready.json',{'run':str(run.path.relative_to(root)),'archive':str(run.path/'temporary_files_recovery.zip')})
    start=time.monotonic();marker=root/review['review']/'filesystem_cleanup.json'
    while not marker.exists():
        if time.monotonic()-start>900:raise TimeoutError('Native filesystem cleanup did not finish')
        time.sleep(1)
    filesystem=read_json(marker);write_json(run.path/'filesystem_cleanup.json',filesystem)
    if filesystem.get('errors'):raise ValueError('Some filesystem cleanup operations failed')
    cmd(['docker','image','rm',*tags])
    prune=cmd(['docker','--config','.cache/docker_agent','buildx','prune','--force','--filter','id='+cache_id],timeout=300)
    images_after=set(output(cmd(['docker','image','ls','--no-trunc','--quiet'])).splitlines())
    if images_before-images_after!={image}:raise ValueError('Unexpected image removal')
    cached_after=output(cmd(['docker','--config','.cache/docker_agent','buildx','du','--verbose']))
    cache_ids_after=set(re.findall(r'^ID:\s+(\S+)',cached_after,re.M));removed_cache=cache_ids_before-cache_ids_after
    if not removed_cache<={cache_id}:raise ValueError('Unexpected cache records removed')
    cmd(['docker','system','df'])
    containers_after=output(cmd(['docker','ps','-a','--no-trunc','--format','{{json .}}']))
    before_ids={json.loads(line)['ID'] for line in containers_before.splitlines()};after_ids={json.loads(line)['ID'] for line in containers_after.splitlines()}
    if before_ids!=after_ids:raise ValueError('Container set changed during cleanup')
    for env,packages in [('train',['torch','transformers','peft','bitsandbytes']),('agent',['litellm','docker','datasets']),('eval',['docker','datasets'])]:
        code='import importlib.metadata as m,json; print(json.dumps({name:m.version(name) for name in '+repr(packages)+'}))'
        cmd([root/'.envs'/env/'bin/python','-B','-c',code])
    after=guard.snapshot();summary={'passed':True,'checked_at':utc(),'filesystem':filesystem,'removed_docker_image':image,'removed_docker_tags':tags,'removed_cache_ids':sorted(removed_cache),'docker_prune_output':output(prune),'containers_unchanged':True,'validated_images_preserved':sorted(protected_images),'before_storage':before,'after_storage':after,'host_free_delta_bytes':after['free_bytes']-before['free_bytes'],'note':'Docker layer removal releases space inside the WSL filesystem; its VHDX may not return that space to the Windows host without a separate compaction.'}
    write_json(run.path/'verification.json',summary);print(json.dumps(summary,indent=2))
