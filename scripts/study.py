#!/usr/bin/env python
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from swe_study.common import setup_environment
setup_environment()
from swe_study.cli import main
if __name__ == "__main__": main()
