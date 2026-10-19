#!/usr/bin/env python3
from pathlib import Path
import sys,json
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from swe_study.common import setup_environment,read_json,write_json
setup_environment()
from swe_study.audit import Run
with Run('cached_runtime_registration') as run:
    manifest=read_json('manifests/image_overrides.json');entry=manifest['overrides']['pvlib__pvlib-python-807']
    if entry['status']!='gold_and_blank_controls_passed':raise ValueError('Task image controls missing')
    runtime='cs6158/runtime:oh_v0.13.1_9gwl2n4p6cfl3edz_y766p9z6jyxsud5c'
    result=run.command(['docker','image','inspect',entry['image'],runtime],timeout=30)
    base,image=json.loads(Path(result['stdout_path']).read_text())
    if base['Id']!=entry['image_id'] or image['RootFS']['Layers'][:len(base['RootFS']['Layers'])]!=base['RootFS']['Layers']:raise ValueError('Runtime does not descend from the validated task base')
    entry.update(runtime_image=runtime,runtime_image_id=image['Id'],runtime_registration=str(run.path))
    other=manifest['overrides']['pvlib__pvlib-python-1048'];other['runtime_image_id']=other['image_id']
    other['controls']['blank_note']='Official harness short-circuits empty predictions; this checks denominator/empty handling, not an executed test-negative control.'
    write_json('manifests/image_overrides.json',manifest)
    write_json(run.path/'registration.json',entry)
