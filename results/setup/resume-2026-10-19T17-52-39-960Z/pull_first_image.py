from pathlib import Path
import sys,json
root=Path(__file__).resolve().parents[3];sys.path.insert(0,str(root/'src'))
from swe_study.common import setup_environment,write_json
setup_environment()
from swe_study.audit import Run
from swe_study.storage import StorageGuard
from swe_study.docker_storage import require_storage
with Run('docker_image',{'instance_id':'pvlib__pvlib-python-807','image':'swebench/sweb.eval.x86_64.pvlib_1776_pvlib-python-807:latest'}) as run:
    require_storage();guard=StorageGuard();guard.check(15*10**9)
    image='swebench/sweb.eval.x86_64.pvlib_1776_pvlib-python-807:latest'
    run.command(['docker','pull',image],timeout=1200,guard=guard)
    result=run.command(['docker','image','inspect',image],timeout=20,guard=guard)
    detail=json.loads(Path(result['stdout_path']).read_text())[0]
    write_json('manifests/development_images.json',{'images':[{'instance_id':'pvlib__pvlib-python-807','image':image,'id':detail['Id'],'repo_digests':detail['RepoDigests'],'expanded_bytes':detail['Size'],'run':str(run.path)}]})
    guard.snapshot()
