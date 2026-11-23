#!/usr/bin/env python3
"""Validate an extracted cloud bundle without installing packages or downloading assets."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from swe_study.common import local_path, setup_environment, write_json
from swe_study.audit import Run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", required=True)
    p.add_argument("--gpu", action="store_true", help="Run a tiny random single-GPU NF4 train/reload/inference diagnostic")
    p.add_argument("--tests", action="store_true", help="Run the regression suite inside the fresh bundle")
    args = p.parse_args()
    setup_environment()
    with Run("cloud_bundle_validation", vars(args)) as run:
        destination = ROOT / ".cache/tmp" / ("cloud-bundle-" + run.id)
        destination.mkdir()
        with tarfile.open(local_path(args.bundle)) as archive:
            # This package contains regular files only. Explicit extraction also
            # avoids a broken tarfile data_filter backport on the local Python.
            for member in archive:
                target = (destination / member.name).resolve()
                if not member.isfile() or not target.is_relative_to(destination.resolve()):
                    raise ValueError(f"Unexpected archive entry: {member.name}")
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source:
                    target.write_bytes(source.read())
                target.chmod(member.mode & 0o777)
        project = destination / "swe-study-a100"
        for forbidden in [".env", ".envs", "manifests/storage_baseline.json", "manifests/docker_storage.json"]:
            if (project / forbidden).exists(): raise ValueError(f"Bundle imported forbidden host state: {forbidden}")
        manifest = json.loads((project / "manifests/cloud_bundle.json").read_text())
        from swe_study.common import sha256_file
        for name, digest in manifest["files_sha256"].items():
            if sha256_file(project / name) != digest: raise ValueError(f"Bundle checksum differs: {name}")
        run.command(["bash", "-n", "cloud/run.sh", "cloud/all.sh", "cloud/train.slurm"], cwd=project)
        run.command([sys.executable, "scripts/cloud.py", "--help"], cwd=project)
        code = "import runpy; ns=runpy.run_path('scripts/cloud.py'); ns['initialize']()"
        run.command([sys.executable, "-c", code], cwd=project)
        run.command([sys.executable, "scripts/cloud.py", "train", "--gpus", "4", "--dry-run"], cwd=project)
        report = {"passed": True, "bundle": args.bundle, "file_hashes_checked": len(manifest["files_sha256"]),
                  "extracted_project": str(project.relative_to(ROOT)), "synthetic_gpu": None,
                  "limitation": "No A100, multi-GPU NCCL or full-model feasibility/performance validation"}
        if args.tests:
            tests = run.command([sys.executable, "-B", "-m", "pytest", "-q", "--capture=sys", "-p", "no:cacheprovider",
                                 "--basetemp", ".cache/tmp/pytest-cloud"], cwd=project, timeout=600)
            report["regression_stdout"] = Path(tests["stdout_path"]).read_text()
        if args.gpu:
            result = run.command([sys.executable, "scripts/cloud.py", "check-distributed", "--gpus", "1"],
                                 cwd=project, timeout=900)
            checks = list((project / "results/cloud_check-distributed").glob("*/diagnostic.json"))
            if len(checks) != 1: raise ValueError("Expected one tiny GPU diagnostic result")
            report["synthetic_gpu"] = json.loads(checks[0].read_text())
            fixture = checks[0].parent
            model = project / "models/Qwen2.5-Coder-1.5B-Instruct"
            shutil.copytree(fixture / "random-model", model)
            inference = fixture / "inference.json"
            run.command([sys.executable, "scripts/cloud.py", "infer", "--size", "1.5B",
                         "--adapter", (fixture / "adapter").relative_to(project), "--prompt", "issue",
                         "--max-context", "128", "--max-new-tokens", "4", "--output", inference.relative_to(project)],
                        cwd=project, timeout=300)
            generation = json.loads(inference.read_text())
            if not 0 < generation["completion_tokens"] <= 4: raise ValueError("Inference did not produce the bounded completion")
            report["synthetic_inference"] = {"passed": True, "completion_tokens": generation["completion_tokens"]}
        write_json(run.path / "validation.json", report)
        print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
