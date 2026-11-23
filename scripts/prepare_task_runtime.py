#!/usr/bin/env python3
"""CPU-only runtime construction and tool-mechanics check for one frozen task."""
import argparse
import asyncio
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'vendor/SWE-Gym-OpenHands'))
from swe_study.audit import Run
from swe_study.common import read_json,setup_environment,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-file',required=True)
    args=parser.parse_args();setup_environment()
    rows=json.loads(Path(args.task_file).read_text())
    if len(rows)!=1 or any(rows[0].get(key) for key in ('patch','test_patch','hints_text')):
        raise ValueError('Runtime preparation requires one public task with no reference patch')
    with Run('benchmark_runtime_prepare',vars(args)) as run:
        from swe_study.runtime import configure_runtime,configure_task_images
        from swe_study.docker_storage import require_storage
        from swe_study.storage import StorageGuard
        configure_runtime(run.path);configure_task_images(rows,run.path,run.id)
        require_storage();StorageGuard().check(8*10**9)
        import yaml
        from openhands.core.config import AppConfig,SandboxConfig
        from openhands.events import EventStream
        from openhands.storage.memory import InMemoryFileStore
        from openhands.runtime.impl.eventstream.eventstream_runtime import EventStreamRuntime
        from openhands.agenthub.codeact_agent.codeact_agent import CodeActAgent
        from openhands.events.action import CmdRunAction,IPythonRunCellAction
        task=yaml.safe_load((ROOT/'vendor/swe-bench-tasks/tasks'/rows[0]['instance_id']/'task.yaml').read_text())
        repairs=read_json('manifests/image_overrides.json')['overrides']
        image=repairs.get(rows[0]['instance_id'],{}).get('image',task['image'])
        cfg=AppConfig(run_as_openhands=False,workspace_base=None,workspace_mount_path=None,
                      file_store='memory',cache_dir=str(ROOT/'.cache/openhands'),
                      sandbox=SandboxConfig(base_container_image=image,use_host_network=False,
                                            keep_runtime_alive=False,timeout=60,platform='linux/amd64'))
        runtime=EventStreamRuntime(cfg,EventStream(run.id,InMemoryFileStore()),sid='cs6158-'+run.id,
                                    plugins=CodeActAgent.sandbox_plugins)
        try:
            asyncio.run(runtime.connect());runtime.container.reload()
            details=runtime.container.attrs
            if details.get('Mounts') or details['HostConfig'].get('Privileged'):
                raise ValueError('Runtime unexpectedly exposes a host mount or privilege')
            for bindings in details['HostConfig'].get('PortBindings',{}).values():
                if any(b.get('HostIp')!='127.0.0.1' for b in bindings): raise ValueError('Runtime port is not loopback-only')
            # This executes the agent's command tool, independently of model quality.
            action=CmdRunAction(command='git -C /testbed rev-parse HEAD');action.timeout=60
            observation=runtime.run_action(action)
            if getattr(observation,'exit_code',None)!=0 or rows[0]['base_commit'] not in observation.content:
                raise ValueError('Runtime task checkout/command execution failed')
            marker='/testbed/.swe_study_probe_'+run.id.replace('-','_')
            actions=[IPythonRunCellAction(code='print(file_editor(command="create", path='+repr(marker)+', file_text="before\\n"))'),
                     IPythonRunCellAction(code='print(file_editor(command="view", path='+repr(marker)+'))'),
                     IPythonRunCellAction(code='print(file_editor(command="str_replace", path='+repr(marker)+', old_str="before", new_str="after"))'),
                     CmdRunAction(command='python -c '+json.dumps('from pathlib import Path; assert Path('+repr(marker)+').read_text()=="after\\n"')),
                     CmdRunAction(command='git -C /testbed add -- '+marker),
                     CmdRunAction(command='git -C /testbed diff --cached -- '+marker)]
            for action in actions:
                action.timeout=60;observation=runtime.run_action(action)
                run.event('runtime_control_action',action=type(action).__name__,content=getattr(observation,'content',''),exit_code=getattr(observation,'exit_code',None))
                if getattr(observation,'exit_code',None) not in (None,0) or 'Error' in type(observation).__name__:
                    raise ValueError('Runtime view/edit/test control failed')
            if '+after' not in observation.content: raise ValueError('Runtime patch export failed')
            write_json(run.path/'runtime.json',{'passed':True,'instance_id':rows[0]['instance_id'],
                         'base_image':image,'runtime_image':details['Config']['Image'],
                         'runtime_image_id':details['Image'],'tool_execution':'view/edit/test/patch passed'})
        finally:
            if runtime.container is not None: runtime.close()


if __name__=='__main__':main()
