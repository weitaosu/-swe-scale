from __future__ import annotations
import shutil
from pathlib import Path
from .common import ROOT, local_path, read_json, write_json


class StorageLimit(RuntimeError): pass


class StorageGuard:
    def __init__(self):
        self.baseline = read_json("manifests/storage_baseline.json")
    def check(self, reserve_bytes=0):
        free = shutil.disk_usage(ROOT).free
        growth = max(0, self.baseline["initial_free_bytes"] - free)
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
        return {"free_bytes":free,"conservative_volume_growth_bytes":growth,"reserved_bytes":reserve_bytes,"docker_physical_free_bytes":docker_free}
    def snapshot(self):
        d = self.check()
        d["accounting_note"] = "Volume growth includes other concurrent use on this volume; Docker on another volume requires an explicit baseline before image operations."
        write_json("manifests/storage_usage.json", d)
        return d
