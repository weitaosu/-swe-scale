#!/usr/bin/env python3
"""CPU-only container API compatibility check, with project-scoped cleanup."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from swe_study.audit import Run
from swe_study.common import setup_environment,write_json
from swe_study.storage import StorageGuard
from swe_study.docker_storage import record_storage,require_storage


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image',default='docker.io/library/python:3.12-slim')
    args=parser.parse_args();setup_environment()
    with Run('container_api_check',vars(args)) as run:
        record_storage(run);require_storage();guard=StorageGuard();guard.check(2*10**9)
        import docker
        client=docker.from_env(timeout=120)
        if not client.ping(): raise RuntimeError('Container Docker-compatible API did not respond')
        run.command(['docker','pull',args.image],guard=guard,timeout=900)
        image=client.images.get(args.image)
        labels={'cs6158.project':'local-swe-study','cs6158.condition_id':run.id}
        container=None
        try:
            container=client.containers.run(args.image,['python','-m','http.server','8765','--bind','0.0.0.0'],
                        name='openhands-runtime-cs6158-'+run.id,detach=True,labels=labels,
                        ports={'8765/tcp':('127.0.0.1',None)})
            result=container.exec_run(['python','-c','from pathlib import Path; Path("/tmp/probe").write_text("ok"); print(Path("/tmp/probe").read_text())'])
            if result.exit_code or result.output.strip()!=b'ok': raise RuntimeError('Container exec/write failed')
            stream=container.get_archive('/tmp/probe')[0]
            if not sum(len(chunk) for chunk in stream): raise RuntimeError('Container archive export failed')
            container.reload();details=container.attrs
            if details.get('Mounts') or details['HostConfig'].get('Privileged'):
                raise RuntimeError('Unexpected test-container mount or privilege')
            bindings=details['NetworkSettings']['Ports']['8765/tcp']
            if not bindings or any(b.get('HostIp')!='127.0.0.1' for b in bindings):
                raise RuntimeError('Container port is not bound to loopback')
            import urllib.request,time
            deadline=time.monotonic()+30
            while True:
                try:
                    with urllib.request.urlopen('http://127.0.0.1:'+bindings[0]['HostPort'],timeout=2) as response:
                        if response.status!=200: raise RuntimeError('Published container port failed')
                    break
                except OSError:
                    if time.monotonic()>=deadline: raise
                    time.sleep(.2)
            report={'passed':True,'api_version':client.version(),'image_id':image.id,
                    'container_exec_archive_and_loopback_port':'passed',
                    'limitation':'API compatibility check only; task gold controls and real agent rollout still required'}
            write_json(run.path/'compatibility.json',report);print(json.dumps(report,indent=2))
        finally:
            if container is not None: container.remove(force=True)
            client.close()


if __name__=='__main__':main()
