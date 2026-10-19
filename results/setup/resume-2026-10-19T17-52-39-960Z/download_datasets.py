from pathlib import Path
import json,sys,urllib.request,time,os
from types import SimpleNamespace
root=Path(__file__).resolve().parents[3];sys.path.insert(0,str(root/'src'))
from swe_study.common import setup_environment,read_json,write_json,local_path,sha256_file
setup_environment()
from swe_study.audit import Run
from swe_study.assets import selected_files
from swe_study.storage import StorageGuard
with Run('download',{'kind':'dataset','transport':'stdlib HTTPS; pinned metadata and LFS SHA256 verification'}) as run:
    inventory=read_json('manifests/planned_downloads_4090.json');manifest={'assets':[]};guard=StorageGuard()
    for asset in inventory['datasets']:
        repo=asset['repo'];revision=asset.get('revision') or 'main'
        with urllib.request.urlopen('https://huggingface.co/api/datasets/'+repo+'/revision/'+revision+'?blobs=true',timeout=30) as response: info=json.load(response)
        asset['revision']=info['sha'];write_json('manifests/planned_downloads_4090.json',inventory)
        write_json(run.path/(repo.split('/')[-1]+'_remote.json'),info)
        entry={'repo':repo,'kind':'dataset','split':asset['split'],'target':asset['target'],'revision':info['sha'],'status':'downloading','files':[]}
        manifest['assets'].append(entry);write_json('manifests/dataset_files.json',manifest)
        try:
            files=selected_files([SimpleNamespace(**x) for x in info['siblings']],'dataset',asset['split'])
            guard.check(sum(x.size for x in files)+256*1024**2)
            for item in files:
                target=local_path(Path(asset['target'])/item.rfilename);target.parent.mkdir(parents=True,exist_ok=True)
                started=time.perf_counter();reused=target.exists();expected=(getattr(item,'lfs',None) or {}).get('sha256')
                if not reused:
                    temp=local_path(Path('.cache/tmp')/('download-'+run.id+'-'+Path(item.rfilename).name+'.partial'))
                    url='https://huggingface.co/datasets/'+repo+'/resolve/'+info['sha']+'/'+item.rfilename
                    run.event('file_download_start',url=url,path=str(target),expected_bytes=item.size)
                    with urllib.request.urlopen(url,timeout=60) as response,temp.open('wb') as output:
                        while chunk:=response.read(4*1024**2):
                            guard.check(len(chunk));output.write(chunk)
                    if temp.stat().st_size!=item.size: raise ValueError('Incomplete file '+str(temp))
                    if expected and sha256_file(temp)!=expected: raise ValueError('LFS checksum mismatch '+str(temp))
                    os.replace(temp,target)
                actual=sha256_file(target)
                if target.stat().st_size!=item.size or expected and actual!=expected: raise ValueError('Existing file differs '+str(target))
                record={'path':target.relative_to(root).as_posix(),'bytes':target.stat().st_size,'sha256':actual,'expected_lfs_sha256':expected,'reused':reused,'seconds':time.perf_counter()-started}
                entry['files'].append(record);write_json('manifests/dataset_files.json',manifest);run.event('asset_file_verified',**record)
            entry['status']='verified';write_json('manifests/dataset_files.json',manifest)
            print(repo+' verified',flush=True)
        except Exception as exc:
            entry.update(status='blocked',error_type=type(exc).__name__,error=str(exc));write_json('manifests/dataset_files.json',manifest);raise
