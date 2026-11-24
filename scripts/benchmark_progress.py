#!/usr/bin/env python3
"""Read-only live benchmark progress; works with runs already in progress."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import re
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def project_path(value):
    path=(ROOT/value).resolve()
    if not path.is_relative_to(ROOT.resolve()):raise ValueError('Path must stay inside the project')
    return path


def read_record(path,default=None):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return default


def tail(path,limit=131072):
    try:
        with path.open('rb') as stream:
            stream.seek(0,2);stream.seek(max(0,stream.tell()-limit))
            return stream.read().decode(errors='replace')
    except OSError:return ''


def run_index(stage):
    result=[]
    for path in (ROOT/'results'/stage).glob('*/run.json'):
        record=read_record(path)
        if isinstance(record,dict):result.append((path.parent,record))
    return result


def belongs(record,state,suite):
    cfg=record.get('config',{})
    try:return project_path(cfg['state'])==state and cfg.get('suite')==suite
    except (KeyError,TypeError,ValueError):return False


def nested_run(record,parent):
    # Audited nested Run contexts share a PID; subprocess stages use parent_pid.
    pid=parent.get('pid')
    return (pid is not None and (record.get('pid')==pid or record.get('parent_pid')==pid)
            and record.get('started_at','')>=parent.get('started_at','')
            and (not parent.get('ended_at') or record.get('started_at','')<=parent['ended_at']))


def logged_child(parent_path,child_path):
    log=parent_path/'stdout.log'
    # The parent Tee records nested Run IDs, avoiding PID collisions across nodes.
    return not log.exists() or 'Run log: '+str(child_path) in tail(log).splitlines()


def agent_details(item,key,condition,suite,agents):
    candidates=[]
    if item.get('agent_run'):
        path=project_path(item['agent_run']);meta=read_record(path/'run.json',{})
        candidates=[(path,meta)]
    elif item.get('run'):
        parent_path=project_path(item['run']);parent=read_record(parent_path/'run.json',{})
        for path,meta in agents:
            cfg=meta.get('config',{})
            if (nested_run(meta,parent) and logged_child(parent_path,path) and cfg.get('instance_id')==key
                    and cfg.get('condition')==condition and cfg.get('suite')==suite):
                candidates.append((path,meta))
    if not candidates:return None
    path,meta=max(candidates,key=lambda pair:pair[1].get('started_at',''))
    logs=[path/'command-001.stderr.log',path/'command-001.stdout.log']
    # Pinned evaluator logs INFO/STEP to this instance file, not to its console.
    logs+=list((path/'openhands').glob('*/*/*/infer_logs/instance_'+key+'.log'))
    steps=[];latest=None
    for log in logs:
        text=tail(log)
        steps.extend(int(n) for n in re.findall(r'GLOBAL STEP\s+(\d+)',text))
        try:
            if log.stat().st_size and (latest is None or log.stat().st_mtime>latest):latest=log.stat().st_mtime
        except OSError:pass
    return {'run':str(path.relative_to(ROOT)),'recorded_status':meta.get('status','unknown'),
            'started_at':meta.get('started_at'),'ended_at':meta.get('ended_at'),
            'reported_step':max(steps) if steps else None,'last_log_unix':latest}


def snapshot(state,suite='development',max_tasks=None):
    state=project_path(state);protocol=read_record(state/'protocol.json')
    if not isinstance(protocol,dict):return {'state':str(state.relative_to(ROOT)),'error':'Study is not planned or protocol is unreadable'}
    frozen=protocol.get('frozen_development_ids' if suite=='development' else 'frozen_final_ids',[])
    ids=frozen[:max_tasks] if max_tasks else frozen
    parents=[]
    for stage in ('cloud_benchmark_prepare','cloud_benchmark_predict','cloud_benchmark_grade','benchmark_preparation_reuse'):
        for path,meta in run_index(stage):
            if meta.get('status')=='running' and belongs(meta,state,suite):parents.append((stage,path,meta))
    agents=run_index('benchmark_agent');graders=run_index('benchmark_grade')
    rows=[];counts=dict(prepared=0,predictions=0,graded=0,interrupted=0,agent_errors=0)
    for key in ids:
        folder=state/suite/'tasks'/key
        prepared=read_record(folder/'prepared.json',{}).get('passed') is True
        counts['prepared']+=prepared
        for condition in ('base','adapter'):
            item=read_record(folder/(condition+'.json'),{})
            status=item.get('status','prepared' if prepared else 'not prepared')
            counts['predictions']+=status in ('predicted','graded')
            counts['graded']+=status=='graded'
            counts['interrupted']+=status=='interrupted'
            counts['agent_errors']+=bool(item.get('agent_error'))
            grading=[]
            if status=='predicted':
                for path,meta in graders:
                    cfg=meta.get('config',{})
                    if cfg.get('instance_id')!=key or cfg.get('condition')!=condition:continue
                    if any(stage=='cloud_benchmark_grade' and nested_run(meta,parent) and logged_child(parent_path,path)
                           for stage,parent_path,parent in parents):grading.append((path,meta))
            grader=max(grading,key=lambda pair:pair[1].get('started_at','')) if grading else None
            rows.append({'instance_id':key,'condition':condition,'status':status,
                         'resolved':item.get('resolved') if status=='graded' else None,
                         'agent_error':bool(item.get('agent_error')),
                         'grader':{'run':str(grader[0].relative_to(ROOT)),'recorded_status':grader[1].get('status'),
                                   'started_at':grader[1].get('started_at')} if grader else None,
                         'agent':agent_details(item,key,condition,suite,agents)})
    active=[{'stage':stage,'run':str(path.relative_to(ROOT)),'started_at':meta.get('started_at')} for stage,path,meta in parents]
    return {'state':str(state.relative_to(ROOT)),'suite':suite,'frozen_tasks':len(frozen),'shown_tasks':len(ids),
            'episode_count':2*len(ids),'step_limit':protocol.get('agent_config',{}).get('max_steps',30),
            'counts':counts,'episodes':rows,'recorded_active_stages':active}


def bar(done,total,width=24):
    filled=int(width*min(max(done,0),total)/total) if total else 0
    return '['+'#'*filled+'-'*(width-filled)+f'] {done}/{total}'


def elapsed(start,end=None,now=None):
    try:
        finish=datetime.fromisoformat(end).timestamp() if end else (now if now is not None else time.time())
        seconds=max(0,int(finish-datetime.fromisoformat(start).timestamp()))
        return f'{seconds//60}m{seconds%60:02d}s'
    except (ValueError,TypeError):return '?'


def render(report,now=None):
    now=time.time() if now is None else now
    if report.get('error'):return report['state']+': '+report['error']
    count=report['counts'];tasks=report['shown_tasks'];episodes=report['episode_count']
    lines=[f"{report['state']} | {report['suite']} | showing {tasks}/{report['frozen_tasks']} frozen tasks",
           'Prepared tasks    '+bar(count['prepared'],tasks),
           'Saved predictions '+bar(count['predictions'],episodes),
           'Graded episodes   '+bar(count['graded'],episodes),
           f"Interrupted: {count['interrupted']} | Saved agent errors: {count['agent_errors']}"]
    for row in report['episodes']:
        agent=row['agent'];status=row['status']
        if row['resolved'] is not None:status+=' | '+('resolved' if row['resolved'] else 'unresolved')
        if row['agent_error']:status+=' | agent error'
        if row['grader']:status+=' | grading '+str(row['grader']['recorded_status'])+' (recorded)'
        lines.append(f"  {row['instance_id']} / {row['condition']}: {status}")
        if agent:
            suffix=''
            if agent.get('last_log_unix') is not None:suffix=f" | last log {max(0,int(now-agent['last_log_unix']))}s ago"
            lines.append('    '+elapsed(agent.get('started_at'),agent.get('ended_at'),now)+
                         ' | recorded '+agent['recorded_status']+suffix)
            if agent['reported_step'] is not None:
                lines.append('    Reported step '+bar(agent['reported_step'],report['step_limit'])+' (step budget, not completion %)')
    for stage in report['recorded_active_stages']:
        lines.append('  Recorded stage: '+stage['stage']+' | '+elapsed(stage.get('started_at'),now=now))
        lines.append('    '+stage['run'])
    lines.append('Counts track saved artifacts. Recorded running/started status does not prove process liveness; no ETA is inferred.')
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state',action='append',required=True,help='Repeat for multiple model studies')
    parser.add_argument('--suite',choices=['development','final'],default='development')
    parser.add_argument('--max-tasks',type=int,help='Show only the first N frozen tasks (for the current batch)')
    parser.add_argument('--watch',action='store_true');parser.add_argument('--interval',type=float,default=5)
    parser.add_argument('--json',action='store_true')
    args=parser.parse_args()
    if args.interval<1:parser.error('--interval must be at least one second')
    if args.max_tasks is not None and args.max_tasks<1:parser.error('--max-tasks must be positive')
    try:
        while True:
            reports=[snapshot(state,args.suite,args.max_tasks) for state in args.state]
            if args.watch and sys.stdout.isatty() and not args.json:print('\033[2J\033[H',end='')
            if args.json:print(json.dumps(reports),flush=True)
            else:
                print(datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'),flush=True)
                print('\n\n'.join(render(report) for report in reports),flush=True)
            if not args.watch:break
            time.sleep(args.interval)
    except KeyboardInterrupt:pass


if __name__=='__main__':main()
