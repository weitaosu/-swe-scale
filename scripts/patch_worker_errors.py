#!/usr/bin/env python3
"""Install prompt worker exception reporting in an extracted cloud project."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

OLD = '''            try: fit(run, args.config, args.resume)
            finally:
                if torch.distributed.is_initialized(): torch.distributed.destroy_process_group()
'''
NEW = '''            try: fit(run, args.config, args.resume)
            except BaseException as error:
                # A failed rank must report its original exception before any
                # collective teardown, which can otherwise hide it for minutes.
                import traceback
                traceback.print_exc()
                sys.stderr.flush()
                run.event("training_worker_failed",rank=rank,error_type=type(error).__name__,error=str(error))
                raise
            else:
                if torch.distributed.is_initialized(): torch.distributed.destroy_process_group()
'''


def patch_text(text):
    if text.count(NEW) == 1 and OLD not in text:
        return text
    if text.count(OLD) != 1 or NEW in text:
        raise ValueError("Unfamiliar worker source preserved. Stop and share this message.")
    result = text.replace(OLD, NEW, 1)
    compile(result, "scripts/cloud.py", "exec")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    target = (root / "scripts/cloud.py").resolve()
    if not target.is_relative_to(root) or not (root / "cloud/nersc/env.sh").is_file():
        raise SystemExit("Choose the extracted cloud project root")
    before = target.read_text()
    after = patch_text(before)
    if before == after:
        print("Worker error reporting already installed.")
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + f"-{os.getpid()}"
    backup = root / ".cache/worker-error-backups" / stamp / "cloud.py"
    backup.parent.mkdir(parents=True, exist_ok=True)
    backup.write_text(before)
    fd, temporary = tempfile.mkstemp(prefix="cloud.py.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(after)
        Path(temporary).replace(target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    log = root / "logs" / f"worker-error-patch-{stamp}.json"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(json.dumps({"backup": str(backup), "target": str(target),
                              "before_sha256": hashlib.sha256(before.encode()).hexdigest(),
                              "after_sha256": hashlib.sha256(after.encode()).hexdigest()}, indent=2))
    print("Worker error reporting installed. Resume with the original training options.")
    print(f"Installation log: {log}")


if __name__ == "__main__":
    main()
