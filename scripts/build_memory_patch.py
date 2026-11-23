#!/usr/bin/env python3
"""Build a checked, reversible installer for opt-in memory-safe Qwen loss."""
import hashlib
import json
import runpy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from swe_study.audit import Run
from swe_study.common import write_json

BLOCK = '''    chunk_tokens=int(os.environ.get("SWE_STUDY_LOSS_CHUNK_TOKENS", "0"))
    if chunk_tokens:
        from .memory_loss import enable_chunked_loss
        enable_chunked_loss(model,chunk_tokens)
        run.event("chunked_loss_enabled",chunk_tokens=chunk_tokens,objective="unchanged assistant causal cross-entropy",labeled_logits_returned=False)
'''


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def main():
    with Run("memory_patch_build") as run:
        training = (ROOT / "src/swe_study/training.py").read_text()
        if training.count(BLOCK) != 1 or training.count(',"loss_chunk_tokens":chunk_tokens') != 1:
            raise ValueError("Unexpected training source; preserve the previous installer")
        previous = training.replace(BLOCK, "", 1).replace(',"loss_chunk_tokens":chunk_tokens', "", 1)
        checkpoint_files = runpy.run_path(str(ROOT / "deliverables/nersc_7b_checkpoint_patch.py"))["FILES"]
        if previous != checkpoint_files["src/swe_study/training.py"]["text"]:
            raise ValueError("Previous training source differs from the shipped checkpoint patch")
        names = ["src/swe_study/training.py", "src/swe_study/memory_loss.py",
                 "tests/test_memory_loss.py", "scripts/validate_memory_loss.py"]
        payload = {name: {"text": (ROOT / name).read_text(),
                          "allowed_before_sha256": [sha(previous)] if name == names[0] else []}
                   for name in names}
        installer = '''#!/usr/bin/env python3
"""Install opt-in chunked loss after the checkpoint patch; preserve unknown sources."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import tempfile

FILES = json.loads(PAYLOAD_LITERAL)

def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if not (root/"cloud/nersc/env.sh").is_file() or not (root/"src/swe_study/checkpoints.py").is_file():
        raise SystemExit("Choose the project with the checkpoint support patch already installed.")
    changes = []
    for name, item in FILES.items():
        target = (root/name).resolve()
        if not target.is_relative_to(root):
            raise SystemExit("Target escaped project root")
        compile(item["text"], name, "exec")
        old = target.read_text() if target.exists() else None
        if old == item["text"]:
            continue
        if old is not None and sha(old) not in item["allowed_before_sha256"]:
            raise SystemExit(f"Unfamiliar source preserved: {name}. Stop and share this message.")
        if old is None and item["allowed_before_sha256"]:
            raise SystemExit(f"Required source missing: {name}")
        changes.append((name, target, old, item["text"]))
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S") + f"-{os.getpid()}"
    backup = root/".cache/memory-patch-backups"/stamp
    for name, target, old, text in changes:
        if old is not None:
            saved = backup/name
            saved.parent.mkdir(parents=True, exist_ok=True)
            saved.write_text(old)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=target.name+".", suffix=".tmp", dir=target.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(text)
            Path(temporary).replace(target)
        finally:
            Path(temporary).unlink(missing_ok=True)
        print(f"Installed {name}")
    log = root/"logs"/f"memory-patch-{stamp}.json"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(json.dumps({"changed": [name for name, *_ in changes], "backup": str(backup),
                              "installed_sha256": {name: sha(item["text"]) for name,item in FILES.items()},
                              "activation": "SWE_STUDY_LOSS_CHUNK_TOKENS=128"}, indent=2))
    print("Memory-safe loss installed. Enable with: export SWE_STUDY_LOSS_CHUNK_TOKENS=128")
    print(f"Installation log: {log}")

if __name__ == "__main__":
    main()
'''.replace("PAYLOAD_LITERAL", repr(json.dumps(payload)))
        output = ROOT / "deliverables/nersc_memory_patch.py"
        output.write_text(installer)
        report = {"output": str(output.relative_to(ROOT)), "sha256": sha(installer),
                  "bytes": output.stat().st_size, "files": names,
                  "previous_training_sha256": sha(previous)}
        write_json(run.path / "build.json", report)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
