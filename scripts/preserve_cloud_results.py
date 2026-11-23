#!/usr/bin/env python3
"""Create a checksummed, self-contained CPU archive of one completed cloud training run."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from swe_study.audit import Run
from swe_study.common import local_path, read_json, setup_environment, sha256_file, write_json
from swe_study.storage import StorageGuard


def preservation_files(adapter):
    adapter = local_path(adapter); metadata = read_json(adapter/"experiment.json"); cfg = metadata["config"]
    roots = [adapter, local_path(cfg["model"]), local_path(cfg["train_file"]),
             local_path(cfg["validation_file"]), local_path(metadata["config_path"]),
             ROOT/"configs", ROOT/"src", ROOT/"scripts", ROOT/"cloud", ROOT/"manifests",
             ROOT/"pyproject.toml", ROOT/"results/manual_inference", ROOT/"results/validation_summaries",
             ROOT/"results/cloud_validation",ROOT/"results/benchmark_3b"]
    plan = Path(str(adapter)+".plan.json")
    if plan.exists(): roots.append(plan)
    # Include the exact orchestrator run and its source snapshot.
    roots.append(local_path(metadata["config_path"]).parent)
    for stage in ('benchmark_agent','benchmark_grade','benchmark_gold_control','benchmark_blank_control',
                  'benchmark_runtime_prepare','runtime_build','runtime_image_pull','cloud_serve',
                  'cloud_benchmark_plan','cloud_benchmark_prepare','cloud_benchmark_predict',
                  'cloud_benchmark_grade','cloud_benchmark_summary','cloud_benchmark_cleanup','container_api_check'):
        roots.append(ROOT/'results'/stage)
    for folder in (ROOT/'results').glob('cloud_train_rank*'):
        for path in folder.glob('*/run.json'):
            try:
                source=read_json(path).get('config',{}).get('config')
                if source and local_path(source)==local_path(metadata['config_path']): roots.append(path.parent)
            except (ValueError,OSError): continue
    files = set()
    for source in roots:
        if not source.exists(): continue
        for path in ([source] if source.is_file() else source.rglob("*")):
            if path.is_symlink(): raise ValueError(f"Symlink preserved in place; archive requires real local files: {path}")
            if path.is_file() and not {"__pycache__", ".cache", ".git"} & set(path.relative_to(ROOT).parts):
                local_path(path); files.add(path)
    for key, name in (("train_source_sha256","train_file"),("validation_source_sha256","validation_file")):
        if sha256_file(local_path(cfg[name])) != metadata[key]:
            raise ValueError(f"Recorded {name} changed; do not archive it as the original run")
    if not (adapter/"adapter_model.safetensors").is_file(): raise ValueError("Missing adapter weights")
    return sorted(files)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--archive", action="store_true", help="Also create an uncompressed tar; preserves all source files")
    args=parser.parse_args(); setup_environment()
    with Run("cloud_preservation",vars(args)) as run:
        guard=StorageGuard(); files=preservation_files(args.adapter)
        size=sum(p.stat().st_size for p in files)
        guard.check(size+128*1024**2 if args.archive else 128*1024**2)
        records=[]
        manifest=run.path/"preservation_manifest.json"
        def save_manifest():
            write_json(manifest,{"adapter":args.adapter,"files":records,"logical_bytes":size,
                                 "scope":"Model, complete adapter/checkpoints, data, config, sources and available result artifacts"})
        if args.archive:
            target=run.path/"preservation.tar"
            with tarfile.open(target,"x") as archive:
                for path in files:
                    guard.check()
                    info=archive.gettarinfo(path,arcname="swe-study-a100/"+path.relative_to(ROOT).as_posix())
                    digest=hashlib.sha256()
                    with path.open('rb') as stream:
                        class HashReader:
                            def read(self,size):
                                block=stream.read(size);digest.update(block);return block
                        archive.addfile(info,HashReader())
                    records.append({"path":path.relative_to(ROOT).as_posix(),"bytes":info.size,"sha256":digest.hexdigest()})
                save_manifest()
                archive.add(manifest,arcname="swe-study-a100/"+manifest.relative_to(ROOT).as_posix(),recursive=False)
            digest=sha256_file(target)
            target.with_suffix(".tar.sha256").write_text(f"{digest}  {target.name}\n")
            print("Archive:",target)
        else:
            for path in files:
                records.append({"path":path.relative_to(ROOT).as_posix(),"bytes":path.stat().st_size,"sha256":sha256_file(path)})
            save_manifest()
        print("Manifest:",manifest)
        print("Logical bytes:",size)
        print("Scratch archives are staging copies; copy the archive and checksum to HPSS or project storage for durable preservation.")
        guard.snapshot()


if __name__=="__main__":main()
