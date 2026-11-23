from __future__ import annotations
import shutil
import os
import time
from pathlib import Path
from .common import ROOT, local_path, read_json, write_json


class StorageLimit(RuntimeError): pass


_project_usage_cache = {}


def project_bytes():
    """Count project files, never follow symlinks outside the project."""
    key = str(ROOT)
    cached = _project_usage_cache.get(key)
    if cached and time.monotonic() - cached[0] < 60:
        return cached[1]
    total = 0
    pending = [ROOT]
    while pending:
        try:
            with os.scandir(pending.pop()) as entries:
                for entry in entries:
                    if entry.is_symlink(): continue
                    if entry.is_dir(follow_symlinks=False): pending.append(entry.path)
                    elif entry.is_file(follow_symlinks=False):
                        try: total += entry.stat(follow_symlinks=False).st_size
                        except FileNotFoundError: pass  # Concurrent temporary-file cleanup.
        except FileNotFoundError:
            pass  # An installer can remove a temporary directory during the walk.
    _project_usage_cache[key] = (time.monotonic(), total)
    return total


class StorageGuard:
    def __init__(self):
        self.baseline = read_json("manifests/storage_baseline.json")
    def check(self, reserve_bytes=0):
        free = shutil.disk_usage(ROOT).free
        project_accounting = self.baseline.get("accounting_mode") == "project"
        growth = (max(0, project_bytes() - self.baseline["project_initial_bytes"]) if project_accounting
                  else max(0, self.baseline["initial_free_bytes"] - free))
        docker_path=local_path("manifests/docker_storage.json")
        docker_free=None
        if docker_path.exists():
            docker=read_json(docker_path)
            physical=Path(docker["physical_storage_path"])
            if not physical.exists(): raise StorageLimit("Recorded Docker storage path is unavailable")
            docker_free=shutil.disk_usage(physical if physical.is_dir() else physical.parent).free
            if docker_free-reserve_bytes<self.baseline["min_free_bytes"]: raise StorageLimit("Docker physical volume would violate the 25 GB free-space floor")
            if not docker["same_volume_as_project"]:
                growth+=max(0,docker["initial_physical_free_bytes"]-docker_free)
        if free-reserve_bytes < self.baseline["min_free_bytes"]:
            raise StorageLimit("The operation would violate the 25 GB free-space floor")
        if growth+reserve_bytes > self.baseline["max_incremental_bytes"]:
            raise StorageLimit("The operation would exceed the approved 70 GB incremental storage cap")
        return {"free_bytes":free,"conservative_volume_growth_bytes":growth,"reserved_bytes":reserve_bytes,"docker_physical_free_bytes":docker_free,
                "accounting_mode":self.baseline.get("accounting_mode","filesystem"),"measurement_cache_seconds":60 if project_accounting else 0}
    def snapshot(self):
        d = self.check()
        d["accounting_note"] = ("Project file growth is cached for up to 60 seconds; filesystem free floor is checked each time. User quota must also be checked with the site quota tool."
                                if self.baseline.get("accounting_mode") == "project" else
                                "Volume growth includes other concurrent use on this volume; Docker on another volume requires an explicit baseline before image operations.")
        write_json("manifests/storage_usage.json", d)
        return d
