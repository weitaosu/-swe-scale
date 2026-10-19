"""Project-local compatibility overlay for the pinned OpenHands runtime builder."""
from pathlib import Path
import atexit,hashlib,json,os,shutil,tomllib
from .common import ROOT,read_json,write_json,sha256_file

def loopback_ports(ports):
    result={}
    for key,value in (ports or {}).items():
        if isinstance(value,list):
            result[key]=[{**item,'HostIp':'127.0.0.1'} if isinstance(item,dict) else {'HostIp':'127.0.0.1','HostPort':str(item)} for item in value]
        elif isinstance(value,dict): result[key]=[{**value,'HostIp':'127.0.0.1'}]
        else: result[key]=('127.0.0.1',value[-1] if isinstance(value,tuple) else value)
    return result


def configure_runtime(log_dir):
    log_dir=Path(log_dir);log_dir.mkdir(parents=True,exist_ok=True)
    buildx=read_json('manifests/buildx.json')
    if sha256_file(Path(buildx['path']))!=buildx['sha256']: raise ValueError('Project Buildx changed')
    os.environ['DOCKER_CONFIG']=buildx['docker_config']
    os.environ['OH_RUNTIME_RUNTIME_IMAGE_REPO']='cs6158/runtime'
    import openhands.runtime.impl.eventstream.eventstream_runtime as runtime_module
    atexit.unregister(runtime_module.remove_all_runtime_containers)
    import openhands.runtime.utils.runtime_build as rb
    from openhands.runtime.builder.docker import DockerRuntimeBuilder
    from .audit import Run
    from .storage import StorageGuard
    def audited_build(self,path,tags,platform=None,use_local_cache=False,extra_build_args=None):
        if not tags: raise ValueError('No runtime image tags')
        command=['docker','buildx','build','--progress=plain','--load']
        for tag in tags: command.extend(['--tag',tag])
        if platform: command.extend(['--platform',platform])
        command.extend(extra_build_args or []);command.append(path)
        with Run('runtime_build',{'tags':tags,'platform':platform,'build_context':path}) as build_run:
            build_run.command(command,guard=StorageGuard(),timeout=2400)
            image=self.docker_client.images.get(tags[0])
            build_run.event('runtime_image_built',image_id=image.id,tags=tags)
        return tags[0]
    DockerRuntimeBuilder.build=audited_build
    import docker,yaml
    from docker.models.containers import ContainerCollection
    original_container_run=ContainerCollection.run
    def loopback_container(self,*args,**kwargs):
        if kwargs.get('name','').startswith('openhands-runtime-cs6158-'):
            if kwargs.get('privileged') or kwargs.get('volumes') or kwargs.get('network_mode')=='host':
                raise ValueError('Project runtime requires an unprivileged container with no host mounts')
            kwargs['ports']=loopback_ports(kwargs.get('ports'))
            labels=dict(kwargs.get('labels') or {})
            labels.update({'cs6158.project':'local-swe-study','cs6158.condition_id':os.environ.get('CS6158_CONDITION_ID',log_dir.name)})
            kwargs['labels']=labels
        try: return original_container_run(self,*args,**kwargs)
        except Exception as exc:
            write_json(log_dir/'container_create_error.json',{'error_type':type(exc).__name__,'error':str(exc),'name':kwargs.get('name'),'ports':kwargs.get('ports')})
            raise
    ContainerCollection.run=loopback_container
    allowed=set()
    for task_id in read_json('manifests/dev_tasks.json')['instance_ids']:
        allowed.add(yaml.safe_load((ROOT/'vendor/swe-bench-tasks/tasks'/task_id/'task.yaml').read_text())['image'])
    def audited_image_exists(self,image_name,pull_from_repo=True):
        try:
            self.docker_client.images.get(image_name);return True
        except docker.errors.ImageNotFound: pass
        # Project runtime tags are local build outputs, never remote inputs.
        if not pull_from_repo or image_name.startswith('cs6158/') or image_name not in allowed: return False
        try:
            with Run('runtime_image_pull',{'image':image_name}) as pull_run:
                guard=StorageGuard();guard.check(15*10**9)
                pull_run.command(['docker','pull',image_name],guard=guard,timeout=1200)
                image=self.docker_client.images.get(image_name)
                write_json(pull_run.path/'image.json',{'id':image.id,'image':image_name,'repo_digests':image.attrs.get('RepoDigests'),'expanded_bytes':image.attrs.get('Size')})
            return True
        except RuntimeError: return False
    DockerRuntimeBuilder.image_exists=audited_image_exists
    vendor=ROOT/'vendor/SWE-Gym-OpenHands'
    project=tomllib.loads((vendor/'pyproject.toml').read_text())
    lock=tomllib.loads((vendor/'poetry.lock').read_text())
    pins={p['name'].lower().replace('_','-'):p['version'] for p in lock['package']}
    requirements=(ROOT/'configs/requirements_agent.txt').read_text()
    for name in project['tool']['poetry']['group']['runtime']['dependencies']:
        requirements+=name+'=='+pins[name.lower().replace('_','-')]+'\n'
    constraints=(ROOT/'configs/constraints_agent.txt').read_text()
    fingerprint=hashlib.sha256((requirements+constraints+'local-overlay-v2-required-browser').encode()).hexdigest()
    original=rb.prep_build_folder
    def prep(build_folder,*args,**kwargs):
        original(build_folder,*args,**kwargs)
        folder=Path(build_folder)
        (folder/'code/requirements_runtime.txt').write_text(requirements)
        (folder/'code/constraints_agent.txt').write_text(constraints)
        path=folder/'Dockerfile';text=path.read_text()
        old='poetry install --only main,runtime --no-interaction --no-root'
        if old not in text and 'poetry install' in text: raise ValueError('Unexpected runtime install template')
        text=text.replace(old,'poetry run pip install --no-cache-dir --constraint /openhands/code/constraints_agent.txt --requirement /openhands/code/requirements_runtime.txt')
        copy='COPY ./code/pyproject.toml ./code/poetry.lock /openhands/code/'
        text=text.replace(copy,copy+'\nCOPY ./code/constraints_agent.txt ./code/requirements_runtime.txt /openhands/code/')
        path.write_text(text)
        archive=log_dir/'runtime_builds'/hashlib.sha256(text.encode()).hexdigest()[:16];archive.mkdir(parents=True,exist_ok=True)
        for source in [path,folder/'code/requirements_runtime.txt',folder/'code/constraints_agent.txt']:
            shutil.copy2(source,archive/source.name)
    rb.prep_build_folder=prep
    previous=rb.get_hash_for_lock_files
    rb.get_hash_for_lock_files=lambda base:rb.truncate_hash(hashlib.md5((previous(base)+fingerprint).encode()).hexdigest())
    details={'fingerprint':fingerprint,'changes':['main/runtime dependencies pinned from source lock with recorded LiteLLM 1.53.1 override','Chromium installed because the fork initializes its browser unconditionally; browsing tools disabled','project-local Buildx and Docker client config','broad atexit runtime cleanup disabled'],'vendor_sources_modified':False}
    write_json(log_dir/'runtime_overlay.json',details)
    return details


def configure_task_images(rows,log_dir,run_id):
    """Map public task images and assign unique project-owned container names."""
    import uuid,subprocess,yaml
    from openhands.core.config import AppConfig,SandboxConfig
    from openhands.runtime.impl.eventstream.eventstream_runtime import EventStreamRuntime
    repairs=read_json('manifests/image_overrides.json')['overrides'] if (ROOT/'manifests/image_overrides.json').exists() else {}
    mapping={};runtime_images={}
    for row in rows:
        task=yaml.safe_load((ROOT/'vendor/swe-bench-tasks/tasks'/row['instance_id']/'task.yaml').read_text())
        if any(task.get(k)!=row.get(k) for k in ['instance_id','repo','base_commit']): raise ValueError('Public runtime task identity mismatch')
        image=task['image'];repair=repairs.get(row['instance_id'])
        if repair:
            if repair['base_image']!=image or repair['status']!='gold_and_blank_controls_passed': raise ValueError('Runtime image repair is unvalidated')
            actual=subprocess.check_output(['docker','image','inspect','--format','{{.Id}}',repair['image']],text=True).strip()
            if actual!=repair['image_id']: raise ValueError('Runtime repaired image changed')
            image=repair['image']
        legacy=('docker.io/xingyaoww/sweb.eval.x86_64.'+row['instance_id'].replace('__','_s_')).lower()
        mapping[legacy]=image
        if repair and repair.get("runtime_image"):
            runtime_image=repair["runtime_image"]
            runtime_id=subprocess.check_output(['docker','image','inspect','--format','{{.Id}}',runtime_image],text=True).strip()
            if runtime_id!=repair.get("runtime_image_id",repair["image_id"]): raise ValueError('Cached runtime identity changed')
            runtime_images[legacy]=runtime_image
    original_sandbox=SandboxConfig.__init__
    def sandbox(self,*args,**kwargs):
        image=kwargs.get('base_container_image')
        if image and image.startswith('docker.io/xingyaoww/'):
            if image not in mapping: raise ValueError('Runtime image outside frozen task mapping')
            kwargs['base_container_image']=mapping[image]
            if image in runtime_images: kwargs['runtime_container_image']=runtime_images[image]
        original_sandbox(self,*args,**kwargs)
    SandboxConfig.__init__=sandbox
    original_app=AppConfig.__init__
    def app(self,*args,**kwargs):
        kwargs.setdefault('file_store','local');kwargs.setdefault('file_store_path',str(Path(log_dir)/'events'))
        kwargs.setdefault('cache_dir',str(ROOT/'.cache/openhands'))
        original_app(self,*args,**kwargs)
    AppConfig.__init__=app
    original_runtime=EventStreamRuntime.__init__
    def runtime(self,*args,**kwargs):
        original_runtime(self,*args,**kwargs)
        self.container_name='openhands-runtime-cs6158-'+run_id+'-'+uuid.uuid4().hex[:12]
    EventStreamRuntime.__init__=runtime
    write_json(Path(log_dir)/'task_image_mapping.json',{'mapping':mapping,'runtime_images':runtime_images,'container_prefix':'openhands-runtime-cs6158-'+run_id,'gold_material_exposed':False})
    return mapping
