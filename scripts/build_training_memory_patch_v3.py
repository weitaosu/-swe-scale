#!/usr/bin/env python3
"""Build a source-checked cloud installer for chunked projection plus CE."""
import hashlib
import json
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from swe_study.audit import Run
from swe_study.common import write_json


def sha(text): return hashlib.sha256(text.encode()).hexdigest()


def main():
    with Run("training_memory_v3_build") as run:
        previous = runpy.run_path(str(ROOT/"deliverables/nersc_training_memory_patch_v2.py"))["FILES"]
        names = list(previous) + ["src/swe_study/memory_head.py", "tests/test_memory_head.py",
                                  "scripts/validate_head_memory.py"]
        payload = {name:{"text":(ROOT/name).read_text(),
                         "allowed_before_sha256":[sha(previous[name]["text"])] if name in previous else []}
                   for name in names}
        template = (ROOT/"deliverables/nersc_memory_patch.py").read_text()
        lines = template.splitlines(keepends=True)
        indices = [i for i,line in enumerate(lines) if line.startswith("FILES = json.loads(")]
        if len(indices) != 1: raise ValueError("Unexpected installer template")
        lines[indices[0]] = "FILES = json.loads("+repr(json.dumps(payload))+")\n"
        installer = "".join(lines).replace(
            "Install opt-in chunked loss after the checkpoint patch; preserve unknown sources.",
            "Install chunked output projection plus loss over memory v2; preserve unknown sources.")
        installer = installer.replace("memory-patch-backups", "training-memory-v3-backups").replace(
            "memory-patch-", "training-memory-v3-")
        installer = installer.replace("SWE_STUDY_LOSS_CHUNK_TOKENS=128",
                                     "SWE_STUDY_HEAD_CHUNK_TOKENS=128 SWE_STUDY_MLP_CHUNK_TOKENS=1024")
        installer = installer.replace("Memory-safe loss installed.", "Training memory v3 installed.")
        compile(installer, "nersc_training_memory_patch_v3.py", "exec")
        output = ROOT/"deliverables/nersc_training_memory_patch_v3.py"; output.write_text(installer)
        report = {"output":str(output.relative_to(ROOT)), "bytes":output.stat().st_size,
                  "sha256":sha(installer), "files":names, "upgrade_from":"memory v2"}
        write_json(run.path/"build.json",report); print(json.dumps(report,indent=2))


if __name__ == "__main__": main()
