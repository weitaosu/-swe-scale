#!/usr/bin/env python3
"""Check cloud v3 installer backups, idempotence and fail-before-write safety."""
import json
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from swe_study.audit import Run
from swe_study.common import setup_environment, write_json


def main():
    setup_environment()
    with Run("training_memory_v3_installer_validation") as run:
        old = runpy.run_path(str(ROOT/"deliverables/nersc_training_memory_patch_v2.py"))["FILES"]
        new = runpy.run_path(str(ROOT/"deliverables/nersc_training_memory_patch_v3.py"))["FILES"]
        project = ROOT/".cache/tmp"/("installer-v3-"+run.id)
        for name,text in {**{k:v["text"] for k,v in old.items()},
                          "cloud/nersc/env.sh":"# fixture\n", "src/swe_study/checkpoints.py":"# fixture\n"}.items():
            path = project/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_text(text)
        command = [sys.executable,"deliverables/nersc_training_memory_patch_v3.py","--root",str(project)]
        run.command(command,timeout=60)
        for name,item in new.items():
            assert (project/name).read_text()==item["text"]
        backups = list((project/".cache/training-memory-v3-backups").glob("*/src/swe_study/training.py"))
        assert len(backups)==1 and backups[0].read_text()==old["src/swe_study/training.py"]["text"]
        run.command(command,timeout=60)
        logs=[json.loads(p.read_text()) for p in (project/"logs").glob("training-memory-v3-*.json")]
        assert len(logs)==2 and any(not log["changed"] for log in logs)
        # An unfamiliar existing source must prevent EVERY write, even to
        # an earlier file that would otherwise have an accepted old hash.
        training=project/"src/swe_study/training.py"
        training.write_text(old["src/swe_study/training.py"]["text"])
        unknown=project/"scripts/validate_head_memory.py"; unknown.write_text("# unfamiliar user source\n")
        before={name:(project/name).read_bytes() for name in new}
        try: run.command(command,timeout=60)
        except RuntimeError: pass
        else: raise AssertionError("Unfamiliar source was not rejected")
        assert before=={name:(project/name).read_bytes() for name in new}
        report={"passed":True,"backup":"verified","idempotence":"verified",
                "unknown_sources":"rejected before any writes","project":str(project.relative_to(ROOT))}
        write_json(run.path/"validation.json",report); print(json.dumps(report,indent=2))


if __name__ == "__main__": main()
