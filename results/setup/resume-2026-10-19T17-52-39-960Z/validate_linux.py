from pathlib import Path
import sys,ast
root=Path(__file__).resolve().parents[3];sys.path.insert(0,str(root/'src'))
from swe_study.common import setup_environment
setup_environment()
from swe_study.audit import Run
from swe_study.storage import StorageGuard
with Run('linux_validation',{'scope':'project regression suite and source syntax'}) as run:
    paths=[*root.glob('scripts/*.py'),*root.glob('src/swe_study/*.py'),*root.glob('tests/*.py')]
    for p in paths: ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p))
    run.event('syntax_passed',files=len(paths))
    run.command([root/'.envs/train/bin/python','-B','-m','pytest','-q','--capture=sys','-p','no:cacheprovider','--basetemp',root/'.cache/tmp/pytest_linux_resume'],guard=StorageGuard(),timeout=600)
