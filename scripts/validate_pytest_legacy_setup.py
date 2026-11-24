#!/usr/bin/env python3
"""Reproduce removed legacy setup support and check the proposed pytest pin."""
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from swe_study.audit import Run
from swe_study.common import setup_environment,write_json


def main():
    import pytest
    if int(pytest.__version__.split('.')[0])<8:
        raise ValueError('Run with the existing modern training-environment pytest to reproduce the failure')
    setup_environment()
    with Run('pytest_legacy_setup_validation') as run:
        folder=ROOT/'.cache/tmp'/('pytest-legacy-'+run.id);folder.mkdir()
        target=folder/'packages'
        fixture=folder/'test_legacy.py'
        fixture.write_text('''from io import BytesIO

class TestLegacyWriter:
    def setup(self):
        self.file_out = BytesIO()

    def teardown(self):
        self.file_out.close()

    def test_write(self):
        self.file_out.write(b"preserved bytes")
        assert self.file_out.getvalue() == b"preserved bytes"
''')
        command=[sys.executable,'-m','pytest',str(fixture),'-q','--capture=sys',
                 '-p','no:cacheprovider','--basetemp',str(folder/'modern-temp')]
        # Disable third-party auto-loaded plugins in each isolated check.
        os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD']='1'
        os.environ['PYTEST_ADDOPTS']=''
        try:run.command(command,timeout=60)
        except RuntimeError as exc:
            logs=list(run.path.glob('command-*.stdout.log'))
            if len(logs)!=1 or "has no attribute 'file_out'" not in logs[0].read_text():raise
        else:raise AssertionError('Modern pytest unexpectedly ran the removed legacy setup method')
        run.command([sys.executable,'-m','pip','install','--disable-pip-version-check',
                     '--target',str(target),'pytest==7.4.4'],timeout=180)
        # Use the isolated package directory; do not downgrade any project environment.
        os.environ['PYTHONPATH']=str(target)
        command[-1]=str(folder/'legacy-temp')
        run.command(command,timeout=60)
        report={'passed':True,'modern_pytest':pytest.__version__,'pinned_pytest':'7.4.4',
                'missing_file_out_reproduced':True,'legacy_setup_teardown_pin_check':'passed',
                'scope':'Minimal compatibility fixture only; real NERSC pydicom gold control remains required'}
        write_json(run.path/'validation.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
