from pathlib import Path
import sys,json
root=Path(__file__).resolve().parents[3];sys.path.insert(0,str(root/'src'))
from swe_study.common import setup_environment,write_json
setup_environment()
from swe_study.audit import Run
with Run('image_manifest_probe',{'operation':'read-only registry metadata; no pull/build/container'}) as run:
    for image in ['swebench/sweb.eval.x86_64.pvlib_1776_pvlib-python-807:latest','xingyaoww/sweb.eval.x86_64.pvlib_s_pvlib-python-807:latest']:
        try:
            result=run.command(['docker','manifest','inspect',image],timeout=40)
            metadata=json.loads(Path(result['stdout_path']).read_text())
            run.event('manifest_available',image=image,compressed_layer_bytes=sum(x.get('size',0) for x in metadata.get('layers',[])))
        except Exception as exc: run.event('manifest_unavailable',image=image,error=str(exc))
