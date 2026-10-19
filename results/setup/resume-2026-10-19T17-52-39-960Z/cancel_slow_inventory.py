from pathlib import Path
import os,sys,signal
root=Path(__file__).resolve().parents[3];sys.path.insert(0,str(root/'src'))
from swe_study.common import setup_environment,read_json
setup_environment()
from swe_study.audit import Run
with Run('inventory_coordination',{'action':'cancel_linux_scan','replacement':'native Windows inventory; same project scope and exclusions'}) as run:
    m=read_json('results/inventory/20261019T223527-bca3259a/run.json');pid=m['pid'];args=Path('/proc',str(pid),'cmdline').read_bytes().replace(bytes([0]),b' ').decode()
    if m['status']!='running' or 'scripts/study.py inventory' not in args:raise ValueError('Unexpected inventory process')
    os.kill(pid,signal.SIGINT);run.event('inventory_cancel_requested',pid=pid)
