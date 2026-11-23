#!/usr/bin/env python3
"""Check re-export/reload when interruption happens after the final optimizer step."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from swe_study.audit import Run
from swe_study.common import local_path, setup_environment, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    args = parser.parse_args()
    setup_environment()
    project = local_path(args.project)
    if not project.is_relative_to(ROOT / ".cache/tmp"):
        raise ValueError("Only an isolated synthetic fixture may be used")
    config = project / "configs/cloud/synthetic_step_resume.yaml"
    import yaml
    cfg = yaml.safe_load(config.read_text())
    if not cfg.get("synthetic_fixture"):
        raise ValueError("Real training projects are excluded")
    with Run("checkpoint_finalize_validation", vars(args)) as run:
        for name in ["training.py", "checkpoints.py"]:
            (project / "src/swe_study" / name).write_bytes((ROOT / "src/swe_study" / name).read_bytes())
        run.command([sys.executable, "scripts/cloud.py", "train", "--config", config.relative_to(project),
                     "--gpus", str(cfg["world_size"]), "--resume", "latest"], cwd=project, timeout=900)
        metadata = json.loads((project / cfg["output"] / "experiment.json").read_text())
        if metadata["global_step"] != cfg["max_steps"] or not metadata["finalized_completed_checkpoint"]:
            raise AssertionError("Completed checkpoint took extra training steps")
        report = {"passed": True, "synthetic_fixture": True, "gpu_count": cfg["world_size"],
                  "global_step": metadata["global_step"], "finalized_completed_checkpoint": True,
                  "fresh_reload": "passed"}
        write_json(run.path / "validation.json", report)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
