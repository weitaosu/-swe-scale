#!/usr/bin/env python3
"""Build a stdlib-only, source-checked installer for the remote NERSC checkout."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def main():
    training = (ROOT / "src/swe_study/training.py").read_text()
    begin = training.index("        def on_step_end(", training.index("    class Metrics(TrainerCallback):"))
    end = training.index("        def on_log(", begin)
    previous_training = training[:begin] + training[end:]
    previous_training = previous_training.replace('''        def _run_epoch(self,*args,**kwargs):
            if self.state.global_step>=self.state.max_steps:
                self.control.should_training_stop=True
                return
            return super()._run_epoch(*args,**kwargs)
''', "").replace('''        def _load_best_model(self):
            # Best-model selection can restore the resume checkpoint itself.
            # Check actual optimizer updates before that selection changes weights.
            self.adapter_digest_before_best_reload=adapter_digest(self.model)
            return super()._load_best_model()
''', "").replace("    resume_step=None\n", "").replace(
        '        resume_step=read_json(checkpoint/"trainer_state.json")["global_step"]\n', "").replace('''    optimized_digest=getattr(trainer,"adapter_digest_before_best_reload",after)
    finalized_completed_checkpoint=(resume_step is not None and result.global_step==resume_step
                                    and resume_step>=trainer.state.max_steps)
    if before==optimized_digest and not finalized_completed_checkpoint:
        raise ValueError("Adapter parameters did not change during optimization")
''', '    if before==after: raise ValueError("Adapter parameters did not change")\n').replace(
        '"adapter_updated":before!=optimized_digest,\n      "adapter_hash_after_optimization":optimized_digest,"finalized_completed_checkpoint":finalized_completed_checkpoint,',
        '"adapter_updated":True,').replace(
        '"adapter_updated":metadata["adapter_updated"],"gpu_hours":elapsed/3600',
        '"adapter_updated":True,"gpu_hours":elapsed/3600')
    cloud = (ROOT / "scripts/cloud.py").read_text()
    previous_cloud = cloud.replace('''    interval = cfg.get("checkpoint_every_steps")
    if interval is not None and (type(interval) is not int or interval <= 0):
        raise ValueError("checkpoint_every_steps must be a positive integer")
''', "").replace('''    if args.resume == "latest":
        from swe_study.checkpoints import latest_complete
        args.resume = latest_complete(output, cfg, count).relative_to(ROOT).as_posix()
        run.event("resume_checkpoint_selected", checkpoint=args.resume)
        print(f"Resuming complete checkpoint: {args.resume}")
''', "").replace('s.add_argument("--resume", help="Complete checkpoint path, or latest for a verified completion record")',
                  's.add_argument("--resume")')
    common = (ROOT / "src/swe_study/common.py").read_text()
    before = common.index("    fd, name = tempfile.mkstemp(", common.index("def write_json("))
    after = common.index("\n\ndef sha256_file", before)
    previous_common = (common[:before] + '''    temp = p.with_name(p.name + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\\n", encoding="utf-8")
    temp.replace(p)
''' + common[after:]).replace("import tempfile\n", "", 1)
    previous = {"src/swe_study/training.py": previous_training,
                "scripts/cloud.py": previous_cloud, "src/swe_study/common.py": previous_common}
    names = list(previous) + ["src/swe_study/checkpoints.py", "configs/cloud/qlora_7b_step_resume.yaml",
                              "scripts/validate_checkpoint_resume.py"]
    payload = {name: {"text": (ROOT / name).read_text(),
                      "allowed_before_sha256": [digest(previous[name])] if name in previous else []}
               for name in names}
    installer = '''#!/usr/bin/env python3
"""Install per-optimizer-step recovery support; preserve unfamiliar source files."""
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
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    if not (root / "pyproject.toml").is_file() or not (root / "cloud/nersc/env.sh").is_file():
        raise SystemExit("Choose the extracted swe-study-a100 project root.")
    changes = []
    for name, item in FILES.items():
        target = (root / name).resolve()
        if not target.is_relative_to(root):
            raise SystemExit("Target escaped project root.")
        text = item["text"]
        if name.endswith(".py"):
            compile(text, name, "exec")
        old = target.read_text() if target.exists() else None
        if old == text:
            continue
        if old is not None and sha(old) not in item["allowed_before_sha256"]:
            raise SystemExit(f"Unfamiliar existing file preserved: {name}. Share this output before continuing.")
        if old is None and item["allowed_before_sha256"]:
            raise SystemExit(f"Required source file missing: {name}")
        changes.append((name, target, old, text))
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S") + f"-{os.getpid()}"
    backup = root / ".cache/checkpoint-patch-backups" / stamp
    for name, target, old, text in changes:
        if old is not None:
            saved = backup / name
            saved.parent.mkdir(parents=True, exist_ok=True)
            saved.write_text(old)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(text)
            Path(temporary).replace(target)
        finally:
            Path(temporary).unlink(missing_ok=True)
        print(f"Installed {name}")
    log = root / "logs" / f"checkpoint-patch-{stamp}.json"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(json.dumps({"changed": [name for name, *_ in changes], "backup": str(backup),
                               "installed_sha256": {name: sha(item["text"]) for name, item in FILES.items()}}, indent=2))
    print("Checkpoint support ready. Use configs/cloud/qlora_7b_step_resume.yaml.")
    print(f"Installation log: {log}")

if __name__ == "__main__":
    main()
'''.replace("PAYLOAD_LITERAL", repr(json.dumps(payload)))
    output = ROOT / "deliverables/nersc_7b_checkpoint_patch.py"
    output.parent.mkdir(exist_ok=True)
    output.write_text(installer)
    print(json.dumps({"output": str(output), "bytes": output.stat().st_size,
                      "sha256": digest(installer)}, indent=2))


if __name__ == "__main__":
    main()
