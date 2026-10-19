from __future__ import annotations
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path
from .common import ROOT, local_path, read_json, write_json
from .audit import utc


def _json_command(args):
    p=subprocess.run(args,capture_output=True,text=True,encoding="utf-8",errors="replace",timeout=20)
    if p.returncode: raise RuntimeError(f"Docker probe failed: {p.stderr[-2000:]}")
    return json.loads(p.stdout)


def docker_identity():
    info=_json_command(["docker","info","--format","json"])
    if not info.get("ServerVersion") or info.get("ServerErrors"):
        raise RuntimeError("Docker client is installed but the daemon is unavailable: "+str(info.get("ServerErrors")))
    context=_json_command(["docker","context","inspect"])[0]
    return {"server_id":info.get("ID"),"server_version":info["ServerVersion"],"docker_root":info.get("DockerRootDir"),"operating_system":info.get("OperatingSystem"),"context":context["Name"],"endpoint":context["Endpoints"]["docker"]["Host"]}


def record_storage(run,backing_file=None):
    identity=docker_identity();endpoint=identity["endpoint"]
    if not endpoint.startswith(("unix://","npipe://")): raise RuntimeError("Remote Docker storage is outside this local-machine budget")
    virtual=(os.name=="nt" or "microsoft" in platform.release().lower() or "desktop" in str(identity["operating_system"]).lower())
    if virtual and not backing_file:
        raise RuntimeError("Docker/WSL VM storage needs --backing-file pointing to its actual local VHDX/backing file. No image operation was started.")
    physical=Path(backing_file or identity["docker_root"]).expanduser().resolve(strict=True)
    if backing_file and not physical.is_file(): raise ValueError("The Docker backing file must exist")
    folder=physical if physical.is_dir() else physical.parent
    data={"identity":identity,"physical_storage_path":str(physical),"mapping_basis":"operator supplied VM backing file" if backing_file else "native local daemon DockerRootDir", "initial_physical_free_bytes":shutil.disk_usage(folder).free,"same_volume_as_project":folder.stat().st_dev==ROOT.stat().st_dev,"recorded_at":utc()}
    target=local_path("manifests/docker_storage.json")
    if target.exists():
        existing=read_json(target)
        if existing["physical_storage_path"]!=data["physical_storage_path"] or existing["identity"]["context"]!=identity["context"]:
            raise ValueError("Existing Docker budget baseline differs and was preserved")
        data=existing
    else: write_json(target,data)
    from .storage import StorageGuard
    StorageGuard().check(15*10**9)
    run.event("docker_storage_verified",**data);print(json.dumps(data,indent=2));return data


def require_storage():
    target=local_path("manifests/docker_storage.json")
    if not target.exists(): raise RuntimeError("Run docker-storage before image operations; the physical Docker storage baseline is missing")
    stored=read_json(target);current=docker_identity()
    for key in ["server_id","context","endpoint","docker_root"]:
        if stored["identity"].get(key)!=current.get(key): raise RuntimeError("Docker daemon/context differs from the recorded storage baseline")
    return stored
