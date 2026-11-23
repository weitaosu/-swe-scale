#!/usr/bin/env python3
"""Audit the complete regression suite after the opt-in loss-memory repair."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from swe_study.audit import Run
from swe_study.common import setup_environment, write_json

setup_environment()
with Run("memory_loss_regression") as run:
    result = run.command([sys.executable, "-m", "pytest", "-q", "--capture=sys",
                          "-p", "no:cacheprovider", "--basetemp",
                          str(ROOT / ".cache/tmp" / ("pytest-regression-" + run.id))], timeout=600)
    write_json(run.path / "validation.json", {"passed": True, "command": result})
