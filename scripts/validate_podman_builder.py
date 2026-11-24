#!/usr/bin/env python3
"""Reproduce the pinned builder's Podman version rejection and verify its overlay."""
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'vendor/SWE-Gym-OpenHands'))
from swe_study.audit import Run
from swe_study.common import setup_environment,write_json


def main():
    setup_environment()
    with Run('podman_builder_validation') as run:
        from openhands.runtime.builder.docker import DockerRuntimeBuilder
        version={'Version':'5.8.2','ApiVersion':'1.44',
                 'Components':[{'Name':'Podman Engine','Version':'5.8.2'}]}
        class Client:
            def version(self):return version
        client=Client()
        try: DockerRuntimeBuilder(client)
        except RuntimeError as exc:
            if 'Docker server version' not in str(exc): raise
            original_error=str(exc)
        else: raise AssertionError('Pinned original constructor did not reproduce the reported failure')
        os.environ['SWE_STUDY_CONTAINER_ENGINE']='podman-hpc'
        from swe_study.runtime import configure_runtime
        configure_runtime(run.path)
        builder=DockerRuntimeBuilder(client)
        assert builder.docker_client is client and builder.rolling_logger is not None
        assert client.version()==version and version['Version']=='5.8.2'
        report={'passed':True,'original_error_reproduced':original_error,
                'pinned_builder_initialized':'passed','reported_engine_version':version,
                'limitation':'Real pinned OpenHands imports and overlay with a simulated Podman response; no container build or agent rollout'}
        write_json(run.path/'validation.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
