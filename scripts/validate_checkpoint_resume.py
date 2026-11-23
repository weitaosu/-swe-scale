#!/usr/bin/env python3
"""Audited random-model checkpoint interruption/resume diagnostic, never a quality test."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from swe_study.audit import Run
from swe_study.common import setup_environment, sha256_file, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=int, choices=[1, 2], default=1)
    args = parser.parse_args()
    setup_environment()
    with Run("checkpoint_resume_validation", vars(args)) as run:
        import yaml
        import importlib.util
        spec = importlib.util.spec_from_file_location("cloud_package", ROOT / "scripts/package_cloud.py")
        package = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(package)
        project = ROOT / ".cache/tmp" / ("checkpoint-resume-" + run.id)
        for name, data in package.bundle_files().items():
            target = project / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        run.command([sys.executable, "scripts/cloud.py", "init", "--storage-accounting", "project"], cwd=project)
        run.command([sys.executable, "scripts/cloud.py", "check-distributed", "--gpus", str(args.gpus)],
                    cwd=project, timeout=900)
        fixture = next((project / "results/cloud_check-distributed").glob("*/synthetic.yaml"))
        cfg = yaml.safe_load(fixture.read_text())
        cfg.update(output="checkpoints/step-resume", max_steps=4, checkpoint_every_steps=1,
                   save_strategy="epoch", eval_strategy="epoch", save_total_limit=3,
                   world_size=args.gpus, effective_batch_size=args.gpus)
        config = project / "configs/cloud/synthetic_step_resume.yaml"
        config.write_text(yaml.safe_dump(cfg, sort_keys=False))
        # The fixture starts with the exact plan the cloud launcher enforces on resume.
        plan = {"config": cfg, "train_sha256": sha256_file(project / cfg["train_file"]),
                "validation_sha256": sha256_file(project / cfg["validation_file"])}
        plan_path = project / (cfg["output"] + ".plan.json")
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(json.dumps(plan))
        worker = project / "scripts/synthetic_interrupt_worker.py"
        worker.write_text('''import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from transformers import Trainer,TrainerCallback
from swe_study.audit import Run
from swe_study.training import train
from swe_study.common import setup_environment
setup_environment()
class Interrupt(TrainerCallback):
    def on_save(self,args,state,control,**kwargs):
        if state.global_step==2:
            raise RuntimeError("EXPECTED_CHECKPOINT_INTERRUPTION")
original=Trainer.train
def interrupted(self,*args,**kwargs):
    self.add_callback(Interrupt())
    return original(self,*args,**kwargs)
Trainer.train=interrupted
with Run("synthetic_checkpoint_interruption") as run:
    train(run,"configs/cloud/synthetic_step_resume.yaml")
''')
        try:
            run.command([sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
                         f"--nproc-per-node={args.gpus}", worker.relative_to(project)], cwd=project, timeout=900)
        except RuntimeError:
            logs = list((project / "results/synthetic_checkpoint_interruption").glob("*/run.json"))
            if len(logs) != args.gpus or any(
                    "EXPECTED_CHECKPOINT_INTERRUPTION" not in p.read_text() for p in logs):
                raise
        else:
            raise AssertionError("Synthetic worker failed to interrupt at checkpoint 2")
        checkpoint = project / cfg["output"] / "checkpoint-2"
        completion = json.loads((checkpoint / "checkpoint_complete.json").read_text())
        if completion["global_step"] != 2 or completion["world_size"] != args.gpus:
            raise AssertionError("Checkpoint completion record differs")
        run.command([sys.executable, "scripts/cloud.py", "train", "--config", config.relative_to(project),
                     "--gpus", str(args.gpus), "--resume", "latest"], cwd=project, timeout=900)
        metadata = json.loads((project / cfg["output"] / "experiment.json").read_text())
        if (metadata["global_step"] != 4 or not metadata["resume_from_checkpoint"].endswith("checkpoint-2")
                or not metadata["optimizer_state_entries"]):
            raise AssertionError("Training did not continue optimizer state from step 2 to step 4")
        report = {"passed": True, "synthetic_fixture": True, "gpu_count": args.gpus,
                  "interrupted_after_step": 2, "resumed_to_step": 4,
                  "fresh_reload": "passed", "validation_strategy": "epoch", "checkpoint_every_steps": 1,
                  "extracted_project": str(project.relative_to(ROOT)),
                  "limitation": "Random tiny model; no real 7B memory, quality or performance claim"}
        write_json(run.path / "validation.json", report)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
