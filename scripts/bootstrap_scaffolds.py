#!/usr/bin/env python3
"""Create isolated, audited Linux environments for the pinned agent or grader."""
from pathlib import Path
import argparse,sys,tomllib,collections
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'src'))
from swe_study.common import setup_environment,local_path,write_json
setup_environment()
from swe_study.audit import Run
from swe_study.storage import StorageGuard
p=argparse.ArgumentParser();p.add_argument('--name',choices=['agent','eval'],required=True);a=p.parse_args()
with Run('bootstrap_'+a.name,vars(a)) as run:
    guard=StorageGuard();guard.check(4*10**9)
    env=local_path('.envs/'+a.name);python=env/'bin/python'
    if not python.exists():
        if env.exists() and any(env.iterdir()): raise ValueError('Nonempty environment preserved')
        run.command([sys.executable,'-m','venv','--copies','--without-pip',env],guard=guard,timeout=120)
    run.command([sys.executable,'-m','pip','--python',python,'install','--no-cache-dir','pip==24.0'],guard=guard,timeout=300)
    if a.name=='agent':
        vendor=root/'vendor/SWE-Gym-OpenHands'
        project=tomllib.loads((vendor/'pyproject.toml').read_text())
        lock=tomllib.loads((vendor/'poetry.lock').read_text())
        normalize=lambda n:n.lower().replace('_','-').replace('.','-')
        packages=collections.defaultdict(list)
        for package in lock['package']: packages[normalize(package['name'])].append(package)
        # The historical lock's 1.52.x LiteLLM releases are unavailable on PyPI.
        # 1.53.1 satisfies the fork's ^1.52.3 declaration and has the same dependency bounds.
        packages['litellm'][0]['version']='1.53.1'
        write_json(run.path/'dependency_overrides.json',{'litellm':{'locked':'1.52.5','selected':'1.53.1','reason':'Pinned PyPI version endpoint returns 404; earliest available stable release satisfying declared ^1.52.3; compatibility checks required'}})
        constraints=[]
        for name,entries in sorted(packages.items()):
            versions={e['version'] for e in entries}
            if len(versions)==1 and not any(e.get('source') for e in entries): constraints.append(name+'=='+next(iter(versions)))
        requirements=[]
        for name,spec in project['tool']['poetry']['dependencies'].items():
            if name=='python': continue
            entries=packages[normalize(name)]
            versions={e['version'] for e in entries}
            if len(versions)!=1: raise ValueError('Ambiguous root pin '+name)
            extras='['+','.join(spec.get('extras',[]))+']' if isinstance(spec,dict) and spec.get('extras') else ''
            requirements.append(name+extras+'=='+next(iter(versions)))
        req=root/'configs/requirements_agent.txt';con=root/'configs/constraints_agent.txt'
        req.write_text('# Main dependencies from the pinned OpenHands poetry.lock. Optional GPU/index groups excluded.\n'+'\n'.join(requirements)+'\n')
        con.write_text('# Unique package versions from the pinned OpenHands lock.\n'+'\n'.join(constraints)+'\n')
        run.command([python,'-m','pip','install','--no-cache-dir','--constraint',con,'--requirement',req],guard=guard,timeout=2400)
        run.command([python,'-m','pip','install','--no-cache-dir','--no-deps','-e',vendor],guard=guard,timeout=300)
    else:
        run.command([python,'-m','pip','install','--no-cache-dir','-e',root/'vendor/SWE-bench'],guard=guard,timeout=2400)
    frozen=run.command([python,'-m','pip','freeze','--all'],guard=guard)
    target=root/'manifests'/(a.name+'_packages.freeze.txt')
    target.write_text(Path(frozen['stdout_path']).read_text())
    run.command([python,'-m','pip','check'],guard=guard)
    write_json(run.path/'environment.json',{'path':str(env),'freeze_file':str(target),'source':'pinned vendor checkout'})
