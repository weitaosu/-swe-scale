from pathlib import Path
import sys,time,shutil
root=Path(__file__).resolve().parents[3];sys.path.insert(0,str(root/'src'))
from swe_study.common import setup_environment,read_json,write_json,sha256_file
setup_environment()
from swe_study.audit import Run
from swe_study.storage import StorageGuard
source=Path(__file__).with_name('native_inventory.js')
with Run('native_inventory',{'backend':'native Windows filesystem through Node REPL','indexer_sha256':sha256_file(source)}) as run:
    shutil.copy2(source,run.path/'indexer.js');start=time.monotonic();guard=StorageGuard()
    while not (run.path/'native_complete.json').exists():
        guard.check()
        if time.monotonic()-start>900:raise TimeoutError('Native filesystem inventory did not complete')
        time.sleep(1)
    report=read_json(run.path/'native_index.json')
    if report['file_count']!=len(report['files']) or sum(r.get('bytes',0) for r in report['files'])!=report['logical_file_bytes']:raise ValueError('Native index totals do not match entries')
    report['volume_storage']=guard.snapshot();report['execution_evidence']=str(run.path)
    write_json('manifests/artifact_inventory.json',report)
    run.event('inventory_written',file_count=report['file_count'],logical_file_bytes=report['logical_file_bytes'],scan_seconds=report['scan_seconds'])
    print({'file_count':report['file_count'],'logical_file_bytes':report['logical_file_bytes'],'scan_seconds':report['scan_seconds']})
