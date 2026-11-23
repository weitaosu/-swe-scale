import runpy
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_memory_installer_backup_idempotence_and_unknown_source_preservation(tmp_path):
    installer = ROOT / "deliverables/nersc_memory_patch.py"
    files = runpy.run_path(str(installer))["FILES"]
    builder = runpy.run_path(str(ROOT / "scripts/build_memory_patch.py"))
    before = files["src/swe_study/training.py"]["text"].replace(builder["BLOCK"], "", 1).replace(
        ',"loss_chunk_tokens":chunk_tokens', "", 1)
    env = tmp_path / "cloud/nersc/env.sh"
    env.parent.mkdir(parents=True); env.write_text("fixture")
    target = tmp_path / "src/swe_study/training.py"
    target.parent.mkdir(parents=True); target.write_text(before)
    (target.parent / "checkpoints.py").write_text("fixture")
    command = [sys.executable, str(installer), "--root", str(tmp_path)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    assert target.read_text() == files["src/swe_study/training.py"]["text"]
    backup = list((tmp_path / ".cache/memory-patch-backups").glob("*/src/swe_study/training.py"))
    assert len(backup) == 1 and backup[0].read_text() == before
    second = subprocess.run(command, check=True, capture_output=True, text=True)
    assert "Installed " not in second.stdout
    target.write_text("# unfamiliar source\n")
    protected = subprocess.run(command, capture_output=True, text=True)
    assert protected.returncode != 0 and "Unfamiliar source preserved" in protected.stderr
    assert target.read_text() == "# unfamiliar source\n"
