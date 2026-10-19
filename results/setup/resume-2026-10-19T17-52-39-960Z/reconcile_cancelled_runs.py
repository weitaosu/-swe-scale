from pathlib import Path
import sys,json
root=Path(__file__).resolve().parents[3];sys.path.insert(0,str(root/'src'))
from swe_study.common import setup_environment,read_json,write_json,sha256_file
setup_environment()
from swe_study.audit import Run,utc
with Run('run_state_reconciliation',{'reason':'Controlled environment-gate cancellation terminated descendants before their final metadata flush'}) as run:
    parent=root/'results/development_condition/20261019T215949-17acd5cb/run.json'
    if read_json(parent)['status']!='failed':raise ValueError('Cancellation parent state changed')
    records=[]
    for relative in ['results/rollout/20261019T220137-35d7aef9/run.json','results/runtime_build/20261019T220652-45a8030e/run.json']:
        path=root/relative;meta=read_json(path)
        if meta['status']!='running' or not meta.get('pid') or Path('/proc',str(meta['pid'])).exists():raise ValueError('Cannot prove this recorded process has terminated')
        backup=run.path/(path.parent.parent.name+'-'+path.parent.name+'.original.json');backup.write_bytes(path.read_bytes())
        evidence={'observed_at':utc(),'recorded_pid':meta['pid'],'pid_absent':True,'parent_cancellation':str(parent),'gate':'results/development_environment_gate/20261019T220844-6089e679','original_metadata':str(backup),'original_sha256':sha256_file(backup),'exact_exit_time_and_code_unknown':True}
        meta.update(status='interrupted',reconciliation=evidence,error={'type':'ExternalCancellation','message':'Stopped with the development environment gate; process is absent. Final exit metadata was not flushed.'})
        write_json(path,meta)
        with (path.parent/'events.jsonl').open('a') as output:output.write(json.dumps({'time':utc(),'event':'interrupted_state_reconciled',**evidence})+'\n')
        records.append({'path':str(path),**evidence})
    write_json(run.path/'reconciliation.json',records)
