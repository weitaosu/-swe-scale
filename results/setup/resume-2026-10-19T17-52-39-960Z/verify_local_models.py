from pathlib import Path
import json,sys,urllib.request,time
root=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(root/'src'))
from swe_study.common import setup_environment,read_json,write_json,sha256_file,local_path
setup_environment()
from swe_study.audit import Run
from swe_study.assets import selected_files
from types import SimpleNamespace
from swe_study.storage import StorageGuard
with Run('model_integrity',{'source':'user-downloaded local snapshots; pinned remote metadata'}) as run:
    manifest={'assets':[]}
    for asset in read_json('manifests/planned_downloads_4090.json')['models']:
        url='https://huggingface.co/api/models/'+asset['repo']+'/revision/'+asset['revision']+'?blobs=true'
        with urllib.request.urlopen(url,timeout=30) as response: info=json.load(response)
        if info['sha'] != asset['revision']: raise ValueError('Revision mismatch')
        write_json(run.path/(asset['repo'].split('/')[-1]+'_remote.json'),info)
        entry={'repo':asset['repo'],'kind':'model','target':asset['target'],'revision':info['sha'],'status':'verifying','files':[]}
        manifest['assets'].append(entry)
        write_json('manifests/model_files.json',manifest)
        for item in selected_files([SimpleNamespace(**s) for s in info['siblings']],'model'):
            StorageGuard().check()
            p=local_path(Path(asset['target'])/item.rfilename)
            started=time.perf_counter()
            if not p.exists(): raise FileNotFoundError(p)
            actual=sha256_file(p)
            expected=(getattr(item,'lfs',None) or {}).get('sha256')
            if p.stat().st_size!=item.size: raise ValueError('Size mismatch: '+str(p))
            if expected and actual!=expected: raise ValueError('SHA256 mismatch: '+str(p))
            record={'path':p.relative_to(root).as_posix(),'bytes':p.stat().st_size,'sha256':actual,'expected_lfs_sha256':expected,'reused':True,'seconds':time.perf_counter()-started}
            entry['files'].append(record)
            run.event('asset_file_verified',**record)
            write_json('manifests/model_files.json',manifest)
        entry['status']='verified'
        write_json('manifests/model_files.json',manifest)
        print(asset['repo']+' verified, existing files reused',flush=True)
    run.event('models_verified',models=len(manifest['assets']))
    print('Run log: '+str(run.path),flush=True)
