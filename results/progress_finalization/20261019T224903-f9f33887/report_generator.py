from pathlib import Path
import sys,json,datetime as dt,re
root=Path(__file__).resolve().parents[3];sys.path.insert(0,str(root/'src'))
from swe_study.common import setup_environment,read_json,read_jsonl,write_json
setup_environment()
from swe_study.audit import Run
p=(root/sys.argv[1]).resolve()
if not p.is_relative_to(root/'results/agent_smoke_sequence'):raise ValueError('Unexpected smoke stage')
with Run('smoke_reporting',{'sequence':str(p),'benchmark_score':False}) as run:
    sequence=read_json(p/'smoke_sequence.json');rows=[]
    for check in sequence['checks']:
        evidence=Path(check['evidence']);report=read_json(evidence);condition=evidence.parent;rollout=Path(report['rollout'])
        lines=(condition/'server.stdout.log').read_text().splitlines();paths=[Path(v[len('Run log: '):]) for v in lines if v.startswith('Run log: '+str(root/'results/serve')+'/')]
        if not paths:raise ValueError('Serving evidence missing')
        server=paths[0];events=[e for _,e in read_jsonl(server/'events.jsonl')]
        completions=[e for e in events if e['event']=='completion'];agent=completions[1:]
        output_files=list((rollout/'openhands').rglob('output.jsonl'))
        outputs=[r for _,r in read_jsonl(output_files[0])]
        if len(outputs)!=1:raise ValueError('Smoke denominator changed')
        output=outputs[0];actions={}
        for event in output.get('history',[]):
            if event.get('action') and event.get('source')=='agent':actions[event['action']]=actions.get(event['action'],0)+1
        text=(rollout/'command-001.stderr.log').read_text()
        steps=[int(value) for value in re.findall(r'GLOBAL STEP ([0-9]+)',text)]
        meta=read_json(server/'run.json');ready=next(e for e in events if e['event']=='server_start')
        load=(dt.datetime.fromisoformat(ready['time'])-dt.datetime.fromisoformat(meta['started_at'])).total_seconds()
        row={'label':check['label'],'instance_id':output['instance_id'],'scope':report['scope'],'model_server_initialization_seconds':load,'condition_wall_seconds':read_json(condition/'run.json')['elapsed_seconds'],'agent_completion_requests':len(agent),'agent_prompt_tokens':sum(e['response']['usage']['prompt_tokens'] for e in agent),'agent_completion_tokens':sum(e['response']['usage']['completion_tokens'] for e in agent),'agent_generation_seconds':sum(e['seconds'] for e in agent),'peak_allocated_bytes':max([e['peak_allocated_bytes'] for e in completions],default=0),'peak_reserved_bytes':max([e['peak_reserved_bytes'] for e in completions],default=0),'agent_actions_by_type':actions,'last_logged_global_step':max(steps,default=None),'patch_bytes':len(output.get('test_result',{}).get('git_patch','').encode()),'agent_error':output.get('error'),'resolved':bool(report['grade']['resolved_instances']),'all_http_error_statuses_including_preflight':[e['status_code'] for e in events if e['event']=='http_response' and e['status_code']>=400],'evidence':str(evidence),'server_evidence':str(server)}
        row['agent_generation_tokens_per_second']=row['agent_completion_tokens']/row['agent_generation_seconds'] if row['agent_generation_seconds'] else None
        rows.append(row)
    write_json(run.path/'metrics.json',{'pipeline_completed':True,'scope':'single_first_frozen_task_smoke','max_steps':5,'task_timeout_seconds':300,'replaces_five_task_validation':False,'preflight_one_token_excluded_from_agent_totals':True,'conditions':rows})
    table='| Condition | Agent calls / output tokens | Peak allocated / reserved GiB | Patch bytes | Episode error | Resolved |\n| --- | ---: | ---: | ---: | --- | --- |\n'
    for row in rows:
        error=(row['agent_error'] or 'none').replace('|','/').replace('\n',' ')[:140]
        table+='| '+row['label']+' | '+str(row['agent_completion_requests'])+' / '+str(row['agent_completion_tokens'])+' | '+format(row['peak_allocated_bytes']/2**30,'.2f')+' / '+format(row['peak_reserved_bytes']/2**30,'.2f')+' | '+str(row['patch_bytes'])+' | '+error+' | '+str(row['resolved'])+' |\n'
    doc='# Bounded agent integration diagnostic\n\nUpdated 2026-10-19. These runs exercise the real local model, OpenHands, patch export, and official grader on the first frozen development task only. Limits are five steps and 300 seconds per episode. They do not replace the five-task comparison or establish research performance.\n\n'+table+'\nEach condition uses a fresh server/ledger. The one-token API preflight is excluded from the agent token totals but remains charged to the task budget. Memory peaks cover HTTP completions, not model-loading transients. Agent errors and empty patches remain failures of the attempted episode even when orchestration and report generation complete.\n\n[Detailed metrics](../'+str((run.path/'metrics.json').relative_to(root))+'), [sequence](../'+str(p.relative_to(root))+'/smoke_sequence.json), [remaining environment gates](development_environment_constraints.md).\n'
    (root/'docs/agent_smoke_report.md').write_text(doc)
    print(json.dumps(rows,indent=2))
