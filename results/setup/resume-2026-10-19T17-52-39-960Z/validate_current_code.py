from pathlib import Path
import sys
root=Path(__file__).resolve().parents[3];sys.path.insert(0,str(root/'src'))
from swe_study.common import setup_environment
setup_environment()
from swe_study.audit import Run
from swe_study.storage import StorageGuard
with Run('linux_validation',{'scope':'all current regression tests after scoped smoke and image repair changes'}) as run:
    run.command([root/'.envs/train/bin/python','-B','-m','pytest','-q','--capture=sys','-p','no:cacheprovider','--basetemp',str(root/'.cache/tmp/pytest_linux_current')],guard=StorageGuard(),timeout=600)
