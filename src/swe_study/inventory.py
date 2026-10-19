from __future__ import annotations
import os
from pathlib import Path
from collections import defaultdict
from .common import ROOT,sha256_file,write_json
from .audit import utc
from .storage import StorageGuard


def inventory(run):
    entries=[];totals=defaultdict(lambda:{"files":0,"bytes":0});excluded=["manifests/artifact_inventory.json",str(run.path.relative_to(ROOT)).replace("\\","/")]
    for folder,dirs,files in os.walk(ROOT,followlinks=False):
        dirs[:]=sorted(d for d in dirs if d!=".git" and not (Path(folder)/d).is_symlink())
        for name in sorted(files):
            item=Path(folder)/name;relative=item.relative_to(ROOT).as_posix()
            if any(relative==part or relative.startswith(part+"/") for part in excluded): continue
            try:
                if item.is_symlink():
                    entries.append({"path":relative,"kind":"symlink","target":os.readlink(item)});continue
                size=item.stat().st_size;group=relative.split("/")[0]
                record={"path":relative,"bytes":size};totals[group]["files"]+=1;totals[group]["bytes"]+=size
                fixture=relative.startswith((".cache/tmp/fixture_",".cache/tmp/tiny-qwen-fixture/"))
                if size<=8*1024**2 and (group not in {".envs",".cache","models","vendor"} or fixture): record["sha256"]=sha256_file(item)
                else: record["hash_policy"]="Payload hashes/revisions are in asset manifests; runtime/cache files are inventoried by size."
                entries.append(record)
            except OSError as exc: entries.append({"path":relative,"error":str(exc)})
    report={"captured_at":utc(),"project_root":str(ROOT),"file_count":len(entries),"logical_file_bytes":sum(v["bytes"] for v in totals.values()),"by_top_level":dict(totals),"excluded":excluded+[".git/","directory symlinks"],"note":"Point-in-time logical file sizes; allocated disk space, OS paging, other apps, and later logs differ. Active inventory run and index itself excluded to avoid self-referential hashes.","volume_storage":StorageGuard().snapshot(),"files":entries}
    write_json("manifests/artifact_inventory.json",report);run.event("inventory_written",file_count=len(entries),logical_file_bytes=report["logical_file_bytes"])
    print({"file_count":len(entries),"logical_file_bytes":report["logical_file_bytes"],"index":"manifests/artifact_inventory.json"})
    return report
