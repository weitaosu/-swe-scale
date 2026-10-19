from __future__ import annotations
import fnmatch
import json
import subprocess
import sys
import time
from pathlib import Path
from .common import ROOT, local_path, read_json, write_json, sha256_file
from .storage import StorageGuard


def selected_files(siblings, kind, split=None):
    result=[]
    for s in siblings:
        name=s.rfilename
        if kind=="model":
            keep=name.endswith((".safetensors",".json",".txt",".jinja")) or name in {"LICENSE","README.md"}
            keep=keep and "/" not in name
        else:
            keep=name in {"README.md","LICENSE","dataset_infos.json"}
            if name.endswith(".parquet"):
                basename=Path(name).name
                keep=basename.startswith(split+"-") or basename==split+".parquet" or ("/"+split+"/") in ("/"+name)
        if keep: result.append(s)
    suffix=".safetensors" if kind=="model" else ".parquet"
    if not any(s.rfilename.endswith(suffix) for s in result):
        raise ValueError(f"No {suffix} payload matched the intended asset split {split!r}; inspect the source schema")
    return result


def download_assets(run, kind="all", only=None):
    from huggingface_hub import HfApi,hf_hub_download
    api=HfApi()
    inventory=read_json("manifests/planned_downloads_4090.json")
    guard=StorageGuard();matched=0
    for group,asset_kind in [("models","model"),("datasets","dataset")]:
        if kind not in {"all",asset_kind}: continue
        manifest_path=f"manifests/{asset_kind}_files.json"
        manifest=read_json(manifest_path) if local_path(manifest_path).exists() else {"assets":[]}
        for asset in inventory[group]:
            if only and only not in asset["repo"]: continue
            matched+=1
            entry={"repo":asset["repo"],"kind":asset_kind,"target":asset["target"],"split":asset.get("split"),"status":"resolving"}
            manifest["assets"]=[x for x in manifest["assets"] if x["repo"]!=asset["repo"]]+[entry]
            write_json(manifest_path,manifest)
            run.event("asset_resolving",**entry)
            try:
                info=api.repo_info(asset["repo"],repo_type=asset_kind,revision=asset.get("revision") or "main",files_metadata=True,timeout=10)
                entry["revision"]=info.sha
                asset["revision"]=info.sha
                write_json("manifests/planned_downloads_4090.json",inventory)
                files=selected_files(info.siblings,asset_kind,asset.get("split"))
                if any(s.size is None for s in files): raise ValueError("Remote file size is unknown; storage cannot be reserved safely")
                remaining=sum(s.size or 0 for s in files if not local_path(Path(asset["target"])/s.rfilename).exists())
                guard.check(int(remaining*1.15)+256*1024**2)
                entry.update(status="downloading",files=[])
                for item in files:
                    target=local_path(Path(asset["target"])/item.rfilename)
                    expected_sha=getattr(item.lfs,"sha256",None) if item.lfs else None
                    if target.exists():
                        actual=sha256_file(target)
                        if target.stat().st_size==item.size and (not expected_sha or actual==expected_sha):
                            entry["files"].append({"path":target.relative_to(ROOT).as_posix(),"bytes":target.stat().st_size,"sha256":actual,"reused":True})
                            continue
                        raise ValueError(f"Existing file differs from the pinned remote asset: {target}. It was preserved.")
                    guard.check(int((item.size or 0)*1.15)+128*1024**2)
                    started=time.perf_counter()
                    downloaded=hf_hub_download(asset["repo"],filename=item.rfilename,repo_type=asset_kind,revision=info.sha,local_dir=local_path(asset["target"]))
                    target=Path(downloaded)
                    actual=sha256_file(target)
                    if item.size is not None and target.stat().st_size!=item.size: raise ValueError(f"Size mismatch: {target}")
                    if expected_sha and actual!=expected_sha: raise ValueError(f"LFS SHA256 mismatch: {target}")
                    file_record={"path":target.relative_to(ROOT).as_posix(),"bytes":target.stat().st_size,"sha256":actual,"expected_lfs_sha256":expected_sha,"seconds":time.perf_counter()-started}
                    entry["files"].append(file_record)
                    run.event("asset_file_verified",**file_record)
                    write_json(manifest_path,manifest)
                    guard.snapshot()
                entry["status"]="verified"
                write_json(manifest_path,manifest)
            except Exception as e:
                entry.update(status="blocked",error_type=type(e).__name__,error=str(e))
                write_json(manifest_path,manifest)
                run.event("asset_blocked",**entry)
                raise
    if not matched: raise ValueError("No approved asset matches this selection")
    return True


REPOS={"SWE-bench":"https://github.com/SWE-bench/SWE-bench.git","SWE-Gym-OpenHands":"https://github.com/SWE-Gym/OpenHands.git","swe-bench-tasks":"https://github.com/SWE-bench/swe-bench-tasks.git"}


def checkout_repos(run):
    import re
    guard=StorageGuard()
    manifest=read_json("manifests/source_repositories.json") if local_path("manifests/source_repositories.json").exists() else {"repositories":{}}
    for name,url in REPOS.items():
        dest=local_path(Path("vendor")/name)
        entry=manifest["repositories"].setdefault(name,{"url":url,"path":dest.relative_to(ROOT).as_posix(),"status":"pending"})
        entry["path"]=dest.relative_to(ROOT).as_posix()
        try:
            if not entry.get("commit"):
                result=run.command(["git","ls-remote",url,"HEAD"],timeout=30)
                commit=Path(result["stdout_path"]).read_text(encoding="utf-8").split()[0]
                if not re.fullmatch(r"[0-9a-f]{40}",commit): raise ValueError("Unexpected source revision")
                entry["commit"]=commit
                write_json("manifests/source_repositories.json",manifest)
            if not (dest/".git").exists():
                if dest.exists() and any(dest.iterdir()): raise ValueError(f"Non-checkout directory preserved: {dest}")
                guard.check(1024**3)
                run.command(["git","clone","--filter=blob:none","--no-checkout",url,str(dest)],timeout=600,guard=guard)
            run.command(["git","fetch","--depth","1","origin",entry["commit"]],cwd=dest,timeout=600,guard=guard)
            run.command(["git","checkout","--detach",entry["commit"]],cwd=dest,timeout=600,guard=guard)
            entry["status"]="checked_out"
            write_json("manifests/source_repositories.json",manifest)
        except Exception as e:
            entry.update(status="blocked",error=str(e));write_json("manifests/source_repositories.json",manifest);raise
