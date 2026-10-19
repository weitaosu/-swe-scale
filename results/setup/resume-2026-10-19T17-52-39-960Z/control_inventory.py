from pathlib import Path
import sys,os,signal
root=Path(__file__).resolve().parents[3];sys.path.insert(0,str(root/'src'))
from swe_study.common import setup_environment,read_json,write_json
setup_environment()
from swe_study.audit import Run
mode=sys.argv[1]
if mode not in ['pause','resume']:raise ValueError('Expected pause or resume')
with Run('inventory_coordination',{'action':mode,'reason':'Avoid competing filesystem inventory and model-loading I/O'}) as run:
    meta=read_json('results/inventory/20261019T223527-bca3259a/run.json');pid=meta['pid']
    args=Path('/proc',str(pid),'cmdline').read_bytes().replace(bytes([0]),b' ').decode()
    if meta['status']!='running' or 'scripts/study.py inventory' not in args:raise ValueError('Inventory process identity does not match')
    os.kill(pid,signal.SIGSTOP if mode=='pause' else signal.SIGCONT)
    write_json(run.path/'coordination.json',{'pid':pid,'action':mode,'verified_command':args,'inventory_run':'results/inventory/20261019T223527-bca3259a'})
    print({'pid':pid,'action':mode})
