from pathlib import Path
import sys,json,subprocess,os
root=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(root/'src'))
from swe_study.common import setup_environment,write_json
setup_environment()
from swe_study.audit import Run
from swe_study.docker_storage import record_storage
with Run('wsl_storage_discovery') as run:
    command=['powershell.exe','-NoProfile','-NonInteractive','-Command',r"Get-ChildItem -LiteralPath 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Lxss' | Get-ItemProperty | Select-Object DistributionName,BasePath,VhdFileName | ConvertTo-Json -Compress"]
    result=run.command(command,timeout=30)
    entries=json.loads(Path(result['stdout_path']).read_text(encoding='utf-8-sig'))
    if isinstance(entries,dict): entries=[entries]
    name=os.environ.get('WSL_DISTRO_NAME')
    matches=[x for x in entries if x['DistributionName']==name]
    if len(matches)!=1: raise ValueError('Cannot uniquely identify active distribution '+str(name))
    base=matches[0]['BasePath'].removeprefix(chr(92)*2+'?'+chr(92))
    result=run.command(['wslpath','-u',base],timeout=10)
    folder=Path(Path(result['stdout_path']).read_text().strip())
    disks=list(folder.glob('*.vhdx'))
    if len(disks)!=1: raise ValueError('Cannot uniquely identify distribution VHDX: '+str(disks))
    run.event('wsl_mapping',distribution=name,base_path=str(folder),backing_file=str(disks[0]))
    record_storage(run,str(disks[0]))
