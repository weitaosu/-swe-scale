from pathlib import Path
import sys,json,os
root=Path(__file__).resolve().parents[3];sys.path.insert(0,str(root/'src'))
from swe_study.common import setup_environment,write_json
setup_environment()
from swe_study.audit import Run
from swe_study.storage import StorageGuard
from swe_study.docker_storage import require_storage
with Run('docker_dependency_repair',{'task':'pvlib__pvlib-python-807','repair':'numpy==1.26.4; source and tests unchanged'}) as run:
    require_storage();guard=StorageGuard();guard.check(4*10**9)
    base='swebench/sweb.eval.x86_64.pvlib_1776_pvlib-python-807@sha256:28abd49543ca62244c1f0b261f36892cd4024a042f3533b887ca351b24a032d3'
    target='cs6158/pvlib-python-807:numpy-1.26.4'
    folder=run.path/'build';folder.mkdir()
    (folder/'Dockerfile').write_text('FROM '+base+'\nRUN /opt/miniconda3/envs/testbed/bin/python -m pip install --no-cache-dir numpy==1.26.4\n')
    os.environ['DOCKER_BUILDKIT']='0'
    run.command(['docker','build','--tag',target,str(folder)],guard=guard,timeout=600)
    result=run.command(['docker','image','inspect',target],guard=guard,timeout=20)
    info=json.loads(Path(result['stdout_path']).read_text())[0]
    write_json('manifests/image_overrides.json',{'overrides':{'pvlib__pvlib-python-807':{'base_image':'swebench/sweb.eval.x86_64.pvlib_1776_pvlib-python-807:latest','base_digest':base,'image':target,'image_id':info['Id'],'reason':'Published image NumPy 2 removes np.Inf used by historical pvlib; gold control failed during collection. Only dependency pinned to 1.26.4.','status':'candidate_pending_controls','run':str(run.path)}}})
