#!/usr/bin/env python3
"""Inspect/stop only this user's project Podman API service on the current host."""
import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from swe_study.audit import Run
from swe_study.common import setup_environment,write_json


def matching_service(path,uid,endpoint):
    try:
        status=(path/'status').read_text()
        owner=next(line for line in status.splitlines() if line.startswith('Uid:'))
        if int(owner.split()[1])!=uid:return False
        argv=[part.decode(errors='replace') for part in (path/'cmdline').read_bytes().split(b'\0') if part]
        executable=Path(argv[0]).name if argv else ''
        is_podman=(executable in {'podman','podman-hpc'} or
                   (executable.startswith('python') and len(argv)>1 and Path(argv[1]).name=='podman-hpc'))
        operations={'system','run','create','exec','container','build','stop','rm','ps','inspect','pull','unshare'}
        command=next((i for i,arg in enumerate(argv[1:],1) if arg in operations),len(argv))
        return is_podman and endpoint in argv and argv[command:command+2]==['system','service']
    except (OSError,StopIteration,ValueError):return False


def service_pids(uid,endpoint,proc=Path('/proc')):
    return sorted(int(p.name) for p in proc.iterdir() if p.name.isdigit() and matching_service(p,uid,endpoint))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stop',action='store_true',help='Send SIGTERM to verified project API-service processes')
    args=parser.parse_args();setup_environment()
    uid=os.getuid();endpoint=f'unix:///tmp/{uid}_swe_study_podman/docker.sock'
    if os.environ.get('DOCKER_HOST',endpoint)!=endpoint:
        raise SystemExit('Unexpected socket preserved; source cloud/nersc/container_env.sh first')
    with Run('container_service_stop',vars(args)) as run:
        pids=service_pids(uid,endpoint)
        report={'hostname':__import__('socket').gethostname(),'endpoint':endpoint,
                'matched_pids':pids,'signaled_pids':[],'stop_requested':args.stop}
        if args.stop:
            for pid in pids:
                # Recheck immediately before signaling, including owner and exact endpoint.
                if matching_service(Path('/proc')/str(pid),uid,endpoint):
                    try:os.kill(pid,signal.SIGTERM)
                    except ProcessLookupError:continue
                    report['signaled_pids'].append(pid)
            deadline=time.monotonic()+10
            while service_pids(uid,endpoint) and time.monotonic()<deadline:time.sleep(.2)
            report['remaining_pids']=service_pids(uid,endpoint)
        write_json(run.path/'service.json',report);print(json.dumps(report,indent=2))
        if report.get('remaining_pids'):raise RuntimeError('Project service still running; inspect before restarting')


if __name__=='__main__':main()
