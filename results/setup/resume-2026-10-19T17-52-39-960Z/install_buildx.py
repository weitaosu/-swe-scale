from pathlib import Path
import sys,json,urllib.request,hashlib,os
root=Path(__file__).resolve().parents[3];sys.path.insert(0,str(root/'src'))
from swe_study.common import setup_environment,write_json,sha256_file
setup_environment()
from swe_study.audit import Run
from swe_study.storage import StorageGuard
with Run('buildx_setup',{'version':'v0.17.1','location':'project-local; no global Docker changes'}) as run:
    guard=StorageGuard();guard.check(256*1024**2)
    with urllib.request.urlopen('https://api.github.com/repos/docker/buildx/releases/tags/v0.17.1',timeout=30) as response: release=json.load(response)
    write_json(run.path/'release.json',release)
    assets={a['name']:a for a in release['assets']}
    binary='buildx-v0.17.1.linux-amd64'
    check=next(name for name in assets if name.endswith('checksums.txt'))
    with urllib.request.urlopen(assets[check]['browser_download_url'],timeout=30) as response: checksums=response.read().decode()
    (run.path/'checksums.txt').write_text(checksums)
    expected=next(line.split()[0] for line in checksums.splitlines() if line.split()[-1].lstrip('*')==binary)
    target=root/'.tools/docker/cli-plugins/docker-buildx';target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():
        with urllib.request.urlopen(assets[binary]['browser_download_url'],timeout=60) as response,target.open('wb') as output:
            while data:=response.read(4*1024**2): guard.check(len(data));output.write(data)
    if sha256_file(target)!=expected: raise ValueError('Buildx checksum mismatch')
    target.chmod(0o755)
    config=root/'.cache/docker_agent';config.mkdir(parents=True,exist_ok=True)
    write_json(config/'config.json',{'cliPluginsExtraDirs':[str(target.parent)]})
    os.environ['DOCKER_CONFIG']=str(config)
    result=run.command(['docker','buildx','version'],guard=guard,timeout=20)
    write_json('manifests/buildx.json',{'version':'v0.17.1','path':str(target),'sha256':expected,'docker_config':str(config),'source':release['html_url'],'run':str(run.path)})
