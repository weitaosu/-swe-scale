#!/usr/bin/env python3
"""Create a small cloud source bundle, excluding credentials and host state."""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
FOLDERS = {"src", "scripts", "configs", "tests", "cloud"}
SUFFIXES = {".py", ".sh", ".slurm", ".yaml", ".yml", ".txt", ".md", ".toml"}
MANIFESTS = ["planned_downloads_4090.json", "dev_tasks.json", "eval_100.json",
             "model_files.json", "dataset_files.json", "source_repositories.json"]


def bundle_files():
    files = {}
    for name in sorted(FOLDERS):
        for path in sorted((ROOT / name).rglob("*")):
            if path.is_file() and not path.is_symlink() and (path.suffix in SUFFIXES or path.relative_to(ROOT).as_posix()=="cloud/nersc/podman-bin/docker") and "__pycache__" not in path.parts:
                files[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    for name in MANIFESTS:
        path = ROOT / "manifests" / name
        files["manifests/" + name] = path.read_bytes()
    files["pyproject.toml"] = (ROOT / "pyproject.toml").read_bytes()
    files["README.md"] = (ROOT / "cloud/README.md").read_bytes()
    files["AGENTS.md"] = (ROOT / "cloud/AGENTS.md").read_bytes()
    files[".gitignore"] = (ROOT / ".gitignore").read_bytes()
    return files


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="dist/swe-study-a100-" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S") + ".tar.gz")
    args = p.parse_args()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT) or output.suffixes[-2:] != [".tar", ".gz"]:
        p.error("--output must be a project-local .tar.gz file")
    if output.exists(): p.error("Existing bundle preserved; choose a fresh output path")
    files = bundle_files()
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    manifest = {"created_at": dt.datetime.now(dt.timezone.utc).isoformat(), "files_sha256": hashes,
                "contents": "Source, configs, frozen asset/task manifests. No weights, data, credentials, environments or host accounting."}
    files["manifests/cloud_bundle.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle, tarfile.open(fileobj=handle, mode="w:gz") as archive:
        for name, data in sorted(files.items()):
            info = tarfile.TarInfo("swe-study-a100/" + name)
            info.size = len(data)
            info.mode = 0o755 if name.endswith((".sh", ".slurm", "/podman-bin/docker")) else 0o644
            # TarInfo defaults to epoch 0 (1970), which breaks later ZIP snapshots.
            # A fixed 2000-01-01 UTC also avoids timezone-dependent 1980 boundaries.
            info.mtime = 946684800
            archive.addfile(info, io.BytesIO(data))
    sha = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_name(output.name + ".sha256").write_text(f"{sha}  {output.name}\n", encoding="utf-8")
    evidence = {"archive": output.relative_to(ROOT).as_posix(), "sha256": sha,
                "bytes": output.stat().st_size, "file_count": len(files), **manifest}
    record = ROOT / "results/cloud_package" / (output.name + ".json")
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: evidence[key] for key in ["archive", "sha256", "bytes", "file_count"]}, indent=2))


if __name__ == "__main__": main()
