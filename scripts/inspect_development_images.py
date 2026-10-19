#!/usr/bin/env python3
"""Read-only image availability, compressed-size, cache, and storage audit."""
from pathlib import Path
import json,sys,subprocess,urllib.request,urllib.parse
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from swe_study.common import setup_environment,read_json,write_json
setup_environment()
from swe_study.audit import Run
from swe_study.storage import StorageGuard
import yaml
with Run('development_image_audit') as run:
    rows=[]
    for task in read_json('manifests/dev_tasks.json')['instance_ids']:
        image=yaml.safe_load((root/'vendor/swe-bench-tasks/tasks'/task/'task.yaml').read_text())['image']
        probe=subprocess.run(['docker','image','inspect',image],capture_output=True,text=True,timeout=30)
        row={'instance_id':task,'image':image,'cached':probe.returncode==0}
        if row['cached']:
            details=json.loads(probe.stdout)[0];row.update(image_id=details['Id'],expanded_bytes=details['Size'],repo_digests=details.get('RepoDigests',[]))
        else:
            repository,tag=image.rsplit(':',1)
            query=urllib.parse.urlencode({'service':'registry.docker.io','scope':'repository:'+repository+':pull'})
            try:
                with urllib.request.urlopen('https://auth.docker.io/token?'+query,timeout=30) as response: token=json.load(response)['token']
                def fetch(ref):
                    req=urllib.request.Request('https://registry-1.docker.io/v2/'+repository+'/manifests/'+ref,headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.oci.image.index.v1+json, application/vnd.docker.distribution.manifest.list.v2+json, application/vnd.docker.distribution.manifest.v2+json, application/vnd.oci.image.manifest.v1+json'})
                    with urllib.request.urlopen(req,timeout=30) as response:return json.load(response),response.headers.get('Docker-Content-Digest')
                manifest,digest=fetch(tag)
                if 'manifests' in manifest:
                    candidates=[m for m in manifest['manifests'] if m.get('platform',{}).get('architecture')=='amd64' and m.get('platform',{}).get('os')=='linux']
                    if len(candidates)!=1:raise ValueError('Expected exactly one linux/amd64 manifest')
                    manifest,digest=fetch(candidates[0]['digest'])
                row.update(registry_available=True,digest=digest,compressed_layer_bytes=sum(m['size'] for m in manifest['layers']),layers=len(manifest['layers']))
                write_json(run.path/(task+'.manifest.json'),manifest)
            except Exception as exc:row.update(registry_available=False,error_type=type(exc).__name__,error=str(exc))
        rows.append(row);run.event('development_image_inspected',**row)
    write_json(run.path/'images.json',{'images':rows,'storage':StorageGuard().snapshot(),'note':'Compressed layer bytes do not determine expanded or runtime-build disk requirements.'})
    print(json.dumps(rows,indent=2))
