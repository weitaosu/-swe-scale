#!/usr/bin/env python3
"""Replay controlled view/edit/test/patch actions in one isolated OpenHands container."""
from pathlib import Path
import asyncio,sys,json,atexit
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'));sys.path.insert(0,str(root/'vendor/SWE-Gym-OpenHands'))
from swe_study.common import setup_environment,read_json,write_json
setup_environment()
from swe_study.audit import Run
from swe_study.storage import StorageGuard
from swe_study.docker_storage import require_storage
from swe_study.runtime import configure_runtime
if '--child' not in sys.argv:
    with Run('runtime_supervision',{'scope':'single-container deterministic replay; storage guarded'}) as supervisor:
        supervisor.command([sys.executable,str(Path(__file__).resolve()),'--child'],guard=StorageGuard(),timeout=2400)
    raise SystemExit(0)
with Run('runtime_replay',{'mode':'scripted tool mechanics; no model or benchmark score','task':'pvlib__pvlib-python-807'}) as run:
    configure_runtime(run.path);require_storage();guard=StorageGuard();guard.check(15*10**9)
    from openhands.core.config import AppConfig,SandboxConfig
    from openhands.events import EventStream
    from openhands.storage.memory import InMemoryFileStore
    from openhands.runtime.impl.eventstream.eventstream_runtime import EventStreamRuntime
    from openhands.agenthub.codeact_agent.codeact_agent import CodeActAgent
    from openhands.events.action import CmdRunAction,IPythonRunCellAction
    image=read_json('manifests/image_overrides.json')['overrides']['pvlib__pvlib-python-807']['image']
    cfg=AppConfig(run_as_openhands=False,workspace_base=None,workspace_mount_path=None,file_store='memory',file_store_path=str(run.path/'events'),cache_dir=str(root/'.cache/openhands'),sandbox=SandboxConfig(base_container_image=image,use_host_network=False,keep_runtime_alive=False,timeout=60,platform='linux/amd64'))
    runtime=EventStreamRuntime(cfg,EventStream(run.id,InMemoryFileStore()),sid='cs6158-'+run.id,plugins=CodeActAgent.sandbox_plugins)
    try:
        asyncio.run(runtime.connect())
        runtime.container.reload();details=runtime.container.attrs
        isolation={'id':details['Id'],'mounts':details.get('Mounts',[]),'privileged':details['HostConfig'].get('Privileged'),'network':details['HostConfig'].get('NetworkMode'),'port_bindings':details['HostConfig'].get('PortBindings',{})}
        write_json(run.path/'container_isolation.json',isolation)
        if isolation['mounts'] or isolation['privileged']: raise ValueError('Unexpected container mounts/privilege')
        if any(binding.get('HostIp')!='127.0.0.1' for values in isolation['port_bindings'].values() for binding in values): raise ValueError('Runtime port is not loopback-only')
        marker='/testbed/.cs6158_smoke_'+run.id+'.txt'
        actions=[CmdRunAction(command='cd /testbed && git rev-parse HEAD'),IPythonRunCellAction(code='print(file_editor(command="create", path='+repr(marker)+', file_text="before\\n"))'),IPythonRunCellAction(code='print(file_editor(command="str_replace", path='+repr(marker)+', old_str="before", new_str="after"))'),CmdRunAction(command='cd /testbed && python -c '+"\"from pathlib import Path; assert 'after' in Path('"+marker+"').read_text()\""),CmdRunAction(command='cd /testbed && git add -- '+marker+' && git diff --cached -- '+marker)]
        observations=[]
        for action in actions:
            guard.check();action.timeout=60;obs=runtime.run_action(action)
            record={'action':type(action).__name__,'observation':type(obs).__name__,'content':getattr(obs,'content',str(obs)),'exit_code':getattr(obs,'exit_code',None)}
            observations.append(record);run.event('replay_action',**record)
            if record['exit_code'] not in {None,0} or 'Error' in record['observation']: raise ValueError('Replay action failed')
        if '+after' not in observations[-1]['content']: raise ValueError('Patch export did not contain the edit')
        write_json(run.path/'replay.json',{'passed':True,'benchmark_score':False,'observations':observations})
    finally:
        if runtime.container is not None:
            try:
                (run.path/'container.stdout.log').write_bytes(runtime.container.logs(stdout=True,stderr=False))
                (run.path/'container.stderr.log').write_bytes(runtime.container.logs(stdout=False,stderr=True))
                runtime.container.reload();write_json(run.path/'container_final_state.json',runtime.container.attrs['State'])
            except Exception as exc: run.event('container_log_error',error=str(exc))
        runtime.close()
