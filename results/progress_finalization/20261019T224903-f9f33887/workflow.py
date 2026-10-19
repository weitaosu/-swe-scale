from pathlib import Path
import sys,json,time,socket,shutil
root=Path(__file__).resolve().parents[3];sys.path.insert(0,str(root/'src'))
from swe_study.common import setup_environment,read_json,write_json,sha256_file
setup_environment()
from swe_study.audit import Run,utc
from swe_study.storage import StorageGuard
sequence=root/'results/agent_smoke_sequence/20261019T221705-5a995848'
inventory=root/'results/native_inventory/20261019T224817-43e7e724'
with Run('progress_finalization',{'smoke_sequence':str(sequence),'inventory':str(inventory),'milestones_remain_incomplete':True}) as run:
    shutil.copy2(__file__,run.path/'workflow.py');guard=StorageGuard();start=time.monotonic()
    for parent in [sequence,inventory]:
        while read_json(parent/'run.json')['status']=='running':
            guard.check()
            if time.monotonic()-start>2400:raise TimeoutError('Pending validation did not finish')
            time.sleep(3)
        if read_json(parent/'run.json')['status']!='passed':raise ValueError('Required validation stage failed: '+str(parent))
    generator=root/'results/setup/resume-2026-10-19T17-52-39-960Z/summarize_smoke.py';shutil.copy2(generator,run.path/'report_generator.py')
    command=run.command(['/usr/bin/python3',generator,str(sequence.relative_to(root))],guard=guard,timeout=120)
    candidates=[Path(line[len('Run log: '):]) for line in Path(command['stdout_path']).read_text().splitlines() if line.startswith('Run log: '+str(root/'results/smoke_reporting')+'/')]
    if len(candidates)!=1:raise ValueError('Expected one consolidated smoke report')
    metrics=candidates[0]/'metrics.json';summary=read_json(metrics)
    listing=run.command(['docker','ps','-a','--filter','label=cs6158.project=local-swe-study','--format','{{json .}}'],timeout=30)
    containers=Path(listing['stdout_path']).read_text().strip()
    if containers:raise ValueError('Project-labelled container remains; inspect before finalizing')
    with socket.socket() as probe:
        probe.settimeout(2);api_active=probe.connect_ex(('127.0.0.1',8615))==0
    if api_active:raise ValueError('Project API port still active; inspect ownership')
    active=[]
    for stage in (root/'results').iterdir():
        if not stage.is_dir():continue
        for folder in stage.iterdir():
            if not folder.is_dir() or not folder.name.startswith('20261019T') or folder==run.path:continue
            record=folder/'run.json'
            if not record.exists():continue
            meta=read_json(record)
            if meta['status']=='running':active.append({'run':str(folder),'pid':meta.get('pid'),'pid_exists':Path('/proc',str(meta.get('pid'))).exists()})
    if active:raise ValueError('Unexpected active/stale run records: '+json.dumps(active))
    run.command(['nvidia-smi','--query-gpu=name,memory.total,memory.used,utilization.gpu','--format=csv,noheader'],timeout=30)
    storage=guard.snapshot();status=read_json('manifests/implementation_status.json')
    status.update(as_of=utc(),status='independent_checks_complete_full_development_gated',agent_smoke_sequence_status='passed_as_bounded_pipeline_diagnostic',active_project_jobs=[],next_command='Resolve the frozen development image/storage gates before the full five-task baseline/adapter comparison; completed training pilots need no repeat.')
    status['validated'].update(agent_smoke_sequence=str((sequence/'smoke_sequence.json').relative_to(root)),agent_smoke_metrics=str(metrics.relative_to(root)),final_runtime_cleanup=str((run.path/'finalization.json').relative_to(root)))
    status['documentation']['agent_smoke_report']='docs/agent_smoke_report.md'
    status['limitations'][-1]='Full frozen-five baseline/adapter comparisons remain pending; single-task five-step diagnostics do not establish repair performance.'
    status['milestones'][0]['implemented'].append('All three real-model APIs and first-frozen-task agent/export/grader diagnostics')
    status['milestones'][1]['implemented'].append('Pilot adapter served through the real API and first-frozen-task agent/export/grader diagnostic')
    write_json('manifests/implementation_status.json',status)
    doc=root/'docs/implementation_status.md';text=doc.read_text()
    text=text.replace('cached single-task integration diagnostics are continuing.','cached single-task integration diagnostics have completed.')
    text=text.replace('Adapter-to-agent-to-grader development run','Full frozen-five-task adapter comparison').replace('Real model-driven episodes and grading','Full frozen-five-task model comparison')
    text=text.replace('All four real model/adapter APIs passed; bounded agent smoke sequence is running','All four real model/adapter APIs and bounded smoke orchestration completed; full five-task comparison pending')
    text=text.replace('Real API checks and a bounded smoke sequence continue using','Real API checks and a bounded smoke sequence completed using')
    text=text.replace('Baseline 1.5B, its pilot adapter, 3B, and 7B run sequentially','Baseline 1.5B, its pilot adapter, 3B, and 7B ran sequentially')
    text=text.replace('Current interface checks:','Completed interface checks:').replace('Current smoke supervisor:','Completed smoke sequence:')
    text+='\nThe four bounded diagnostics completed through official grading. Their individual episode errors, patch sizes, token counts, timing, and GPU peaks are in the [agent smoke report](agent_smoke_report.md). These results do not complete the frozen-five-task acceptance checks.\n'
    doc.write_text(text)
    readme=root/'README.md';text=readme.read_text();text=text.replace('Single-task smoke checks use --smoke','[Completed single-task diagnostics](docs/agent_smoke_report.md) use --smoke');readme.write_text(text)
    proof={'checked_at':utc(),'all_independent_checks_finished':True,'milestones_complete':False,'smoke_metrics':str(metrics),'project_containers_remaining':[],'project_api_port_active':False,'active_project_jobs':[],'storage':storage,'full_development_gates':status['blockers']}
    write_json(run.path/'finalization.json',proof)
    files=[root/'README.md',root/'AGENTS.md',*list((root/'docs').glob('*.md')),root/'manifests/implementation_status.json',root/'manifests/storage_usage.json',metrics,sequence/'smoke_sequence.json']
    write_json(run.path/'final_artifact_hashes.json',{'captured_at':utc(),'files':{str(path.relative_to(root)):sha256_file(path) for path in files}})
    print(json.dumps(proof,indent=2))
