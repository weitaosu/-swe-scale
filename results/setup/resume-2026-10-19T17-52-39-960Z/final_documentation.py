from pathlib import Path
import sys,json,socket
root=Path(__file__).resolve().parents[3];sys.path.insert(0,str(root/'src'))
from swe_study.common import setup_environment,read_json,write_json,sha256_file
setup_environment()
from swe_study.audit import Run,utc
from swe_study.storage import StorageGuard
with Run('documentation_finalization',{'scope':'final result clarity, native inventory provenance, current storage, cleanup verification'}) as run:
    native=root/'results/native_inventory/20261019T224817-43e7e724'
    source={'recorded_at':utc(),'specification':{'path':str(native/'indexer.js'),'sha256':sha256_file(native/'indexer.js')},'executed_static_function':{'path':str(native/'executed_indexer.js'),'sha256':sha256_file(native/'executed_indexer.js')},'execution':'Static JavaScript through the Node REPL; only native filesystem reads/writes, no WSL subprocess launch. The exact executed function is stored separately from the original specification.'}
    write_json(native/'execution_source.json',source)
    index=read_json('manifests/artifact_inventory.json');index['execution_source']=source;write_json('manifests/artifact_inventory.json',index)
    guard=StorageGuard();storage=guard.snapshot();free=storage['free_bytes']/1e9;growth=storage['conservative_volume_growth_bytes']/1e9;headroom=70-growth
    if headroom>=15:raise ValueError('Capacity changed; reassess uncached image stage before reporting a storage gate')
    doc=root/'docs/implementation_status.md';text=doc.read_text();text=text.replace('| API/tests | Real 1.5B baseline and pilot-adapter HTTP generation/guards passed; 38 Linux regression tests passed | All four real model/adapter APIs and bounded smoke orchestration completed; full five-task comparison pending |','| API/tests | All three model APIs and the pilot-adapter API passed; 38 Linux regression tests passed; four bounded smoke conditions graded | Full five-task comparison |')
    text+='\nAll four smoke episodes reached the five-step limit. The 7B model exported a 1,508-byte patch; the other three conditions exported empty patches. None resolved the task. The refreshed native inventory contains 111,781 entries, about 38.40 GB of logical files, with zero scan errors.\n'
    text+='\nFinal storage check: '+format(free,'.2f')+' GB free, '+format(growth,'.2f')+' GB conservative growth, and '+format(headroom,'.2f')+' GB remaining under the 70 GB cap. The uncached-image reserve is 15 GB. Project API and container cleanup passed.\n';doc.write_text(text)
    constraints=root/'docs/development_environment_constraints.md';text=constraints.read_text();text+='\nFinal check after the four bounded diagnostics: '+format(free,'.2f')+' GB free, '+format(growth,'.2f')+' GB conservative growth, '+format(headroom,'.2f')+' GB cap headroom. The full five-task gates remain. [Cleanup and storage verification](../results/progress_finalization/20261019T224903-f9f33887/finalization.json).\n';constraints.write_text(text)
    with (root/'AGENTS.md').open('a') as file:file.write('\n- For a full index of this Windows-backed project, prefer native Windows filesystem reads for metadata. The completed native inventory scanned 111,781 entries in 3.4 seconds; the WSL walk was cancelled because NTFS metadata traversal was slow. Exact static indexer source and audited driver evidence are in results/native_inventory/20261019T224817-43e7e724/. Compute, model, data processing, network, and Docker work still use Ubuntu.\n')
    status=read_json('manifests/implementation_status.json');status['as_of']=utc();status['blockers'][0]['message']='Current cap headroom is '+format(headroom,'.2f')+' GB; uncached image stages reserve 15 GB before execution.';status['blockers'][0]['latest_evidence']=str((run.path/'verification.json').relative_to(root));status['validated']['native_file_inventory']='results/native_inventory/20261019T224817-43e7e724/run.json';status['validated']['final_documentation']=str((run.path/'verification.json').relative_to(root));write_json('manifests/implementation_status.json',status)
    command=run.command(['docker','ps','-a','--filter','label=cs6158.project=local-swe-study','--format','{{json .}}'],timeout=30)
    if Path(command['stdout_path']).read_text().strip():raise ValueError('Unexpected project container remains')
    with socket.socket() as probe:
        probe.settimeout(2)
        if probe.connect_ex(('127.0.0.1',8615))==0:raise ValueError('Project API port is active')
    write_json(run.path/'verification.json',{'passed':True,'checked_at':utc(),'storage':guard.snapshot(),'uncached_image_stage_still_gated':True,'project_containers':[],'project_api_active':False,'native_inventory_errors':len([row for row in index['files'] if row.get('error')]),'unit_tests_passed':38,'scope':'four single-task five-step diagnostics, no resolved task; full five-task acceptance incomplete'})
    files=[root/'README.md',root/'AGENTS.md',*list((root/'docs').glob('*.md')),root/'manifests/implementation_status.json',root/'manifests/storage_usage.json',root/'manifests/artifact_inventory.json',native/'execution_source.json']
    write_json(run.path/'final_artifact_hashes.json',{'captured_at':utc(),'files':{str(path.relative_to(root)):sha256_file(path) for path in files}})
    print({'passed':True,'free_GB':free,'cap_headroom_GB':headroom,'verification':str(run.path/'verification.json')})
