from pathlib import Path
import sys,time,json
root=Path(__file__).resolve().parents[3];sys.path.insert(0,str(root/'src'))
from swe_study.common import setup_environment,read_json
setup_environment()
from swe_study.audit import Run
from swe_study.storage import StorageGuard
with Run('post_pilot_validation',{'scope':'real resume validation, then sequential larger-model probes'}) as run:
    guard=StorageGuard();started=time.monotonic();parent=root/'results/continue_setup/20261019T212428-a487fadc/run.json'
    run.event('waiting_for_pilot',parent=str(parent))
    while True:
        guard.check()
        try: status=json.loads(parent.read_text())['status']
        except json.JSONDecodeError: status='running'
        if status!='running': break
        if time.monotonic()-started>1200: raise TimeoutError('Pilot did not finish within wait limit')
        time.sleep(5)
    if status!='passed': raise ValueError('Pilot or fresh reload failed; later GPU stages were not started')
    run.command([root/'.envs/train/bin/python',root/'scripts/validate_resume.py'],guard=guard,timeout=2400)
    run.command([root/'.envs/train/bin/python',root/'scripts/continue_setup.py','--stage','larger-probes'],guard=guard,timeout=3600)
