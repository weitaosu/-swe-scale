#!/usr/bin/env python3
"""Check the remote installer against the previously delivered source archive."""
import argparse
import json
from pathlib import Path
import runpy
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from swe_study.audit import Run
from swe_study.common import local_path, setup_environment, sha256_file, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    args = parser.parse_args()
    setup_environment()
    installer = ROOT / "deliverables/nersc_7b_checkpoint_patch.py"
    with Run("checkpoint_patch_validation", vars(args)) as run:
        destination = ROOT / ".cache/tmp" / ("checkpoint-patch-" + run.id)
        destination.mkdir()
        with tarfile.open(local_path(args.bundle)) as archive:
            for member in archive:
                target = (destination / member.name).resolve()
                if not member.isfile() or not target.is_relative_to(destination):
                    raise ValueError(f"Unsafe archive entry: {member.name}")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.extractfile(member).read())
        project = destination / "swe-study-a100"
        command = [sys.executable, installer, "--root", project]
        run.command(command)
        files = runpy.run_path(installer)["FILES"]
        for name, item in files.items():
            if (project / name).read_text() != item["text"]:
                raise AssertionError(f"Installed source differs: {name}")
        run.command(command)  # Re-running is harmless.
        target = project / "scripts/cloud.py"
        target.write_text(target.read_text() + "\n# unfamiliar user customization\n")
        preserved = target.read_bytes()
        try:
            run.command(command)
        except RuntimeError as exc:
            if "Unfamiliar existing file preserved" not in str(exc):
                raise
        else:
            raise AssertionError("Installer overwrote an unfamiliar customization")
        if target.read_bytes() != preserved:
            raise AssertionError("Unfamiliar source was changed")
        report = {"passed": True, "bundle": args.bundle, "installed_files_checked": len(files),
                  "idempotent": True, "unfamiliar_source_preserved": True,
                  "installer_sha256": sha256_file(installer)}
        write_json(run.path / "validation.json", report)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
