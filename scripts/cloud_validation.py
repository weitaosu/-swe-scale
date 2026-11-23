#!/usr/bin/env python3
"""Evaluate frozen NF4 base and its saved LoRA on exactly the training validation split."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from swe_study.audit import Run
from swe_study.common import setup_environment
from swe_study.validation import run_validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--allow-synthetic", action="store_true")
    args = parser.parse_args(); setup_environment()
    with Run("cloud_validation", vars(args)) as run:
        run_validation(run, args.adapter, args.allow_synthetic)


if __name__ == "__main__": main()
