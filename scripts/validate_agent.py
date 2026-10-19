#!/usr/bin/env python3
"""Audit pinned CodeAct imports and replay released XML actions without executing them."""
from pathlib import Path
import sys,atexit,json
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'));sys.path.insert(0,str(root/'vendor/SWE-Gym-OpenHands'))
from swe_study.common import setup_environment,read_jsonl,write_json
setup_environment()
from swe_study.audit import Run
with Run('agent_compatibility',{'mode':'imports and action conversion only; no runtime commands'}) as run:
    import openhands.runtime.impl.eventstream.eventstream_runtime as runtime_module
    atexit.unregister(runtime_module.remove_all_runtime_containers)
    import openhands.agenthub
    import openhands.core.main
    from litellm import ModelResponse
    from openhands.llm.fn_call_converter import convert_non_fncall_messages_to_fncall_messages
    from openhands.agenthub.codeact_agent.function_calling import response_to_actions
    from openhands.agenthub.codeact_agent.function_calling import get_tools
    configured={t['function']['name'] for t in get_tools()}
    checked=[];seen=set()
    for _,row in read_jsonl('data/processed/swegym/train.jsonl'):
        if {t['function']['name'] for t in row['tools']}!=configured: raise ValueError('Released and configured tool names differ')
        for entry in row['messages']:
            content=entry['content']
            if entry['role']!='assistant' or '<function=' not in content: continue
            message=convert_non_fncall_messages_to_fncall_messages([{'role':'assistant','content':content}],row['tools'])[0]
            calls=message.get('tool_calls')
            if not calls: raise ValueError('Released action was not parsed')
            names={x['function']['name'] for x in calls}
            if names<=seen: continue
            actions=response_to_actions(ModelResponse(choices=[{'index':0,'message':message,'finish_reason':'tool_calls'}]))
            if not actions: raise ValueError('No CodeAct action')
            checked.append({'instance_id':row['instance_id'],'tool_names':sorted(names),'action_types':[type(x).__name__ for x in actions]})
            seen.update(names)
            if seen==configured: break
        if seen==configured: break
    if seen!=configured: raise ValueError('Not all configured tools replayed')
    write_json(run.path/'compatibility.json',{'passed':True,'checked':checked,'commands_executed':False,'container_isolation_validated':False})
    print(json.dumps(checked,indent=2))
