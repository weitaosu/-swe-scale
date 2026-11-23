#!/usr/bin/env python3
"""Audit equivalent loss/gradients and optional synthetic GPU memory/recovery checks."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from swe_study.audit import Run, versions
from swe_study.common import setup_environment, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--checkpoint-resume", action="store_true")
    args = parser.parse_args()
    setup_environment()
    with Run("memory_loss_validation", vars(args)) as run:
        run.command([sys.executable, "-m", "pytest", "tests/test_memory_loss.py", "-q",
                     "--capture=sys", "-p", "no:cacheprovider", "--basetemp",
                     str(ROOT / ".cache/tmp" / ("pytest-memory-loss-" + run.id))], timeout=300)
        report = {"passed": True, "versions": versions(), "loss_gradient_equivalence": "passed",
                  "synthetic_fixture": True,
                  "limitation": "No real 3B/7B 32K feasibility or quality claim"}
        if args.gpu:
            import torch
            from transformers.loss.loss_utils import ForCausalLMLoss
            from swe_study.memory_loss import chunked_causal_loss
            labels = torch.arange(4096, device="cuda").remainder(32768).unsqueeze(0)
            measurements = []
            reference_sample = None
            for optimized in (False, True):
                gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                torch.manual_seed(6158)
                logits = torch.randn(1, 4096, 32768, device="cuda", dtype=torch.bfloat16,
                                     requires_grad=True)
                loss = (chunked_causal_loss(logits, labels, 32768, chunk_tokens=128)
                        if optimized else ForCausalLMLoss(logits, labels, 32768))
                loss.backward(); torch.cuda.synchronize()
                sample = logits.grad[0, :8, :32].detach().cpu().clone()
                value = float(loss.detach())
                measurements.append({"optimized": optimized, "loss": value,
                                     "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                                     "peak_reserved_bytes": torch.cuda.max_memory_reserved()})
                if reference_sample is None:
                    reference_sample, reference_loss = sample, value
                else:
                    torch.testing.assert_close(reference_sample, sample, atol=0, rtol=0)
                    if abs(reference_loss - value) > 1e-5:
                        raise AssertionError("GPU loss differs")
                del logits, loss
            if measurements[1]["peak_allocated_bytes"] >= measurements[0]["peak_allocated_bytes"] * 0.75:
                raise AssertionError("Synthetic loss workspace did not shrink materially")
            report.update(gpu=torch.cuda.get_device_name(), synthetic_gpu_memory=measurements)
            gc.collect(); torch.cuda.empty_cache()
        if args.checkpoint_resume:
            os.environ["SWE_STUDY_LOSS_CHUNK_TOKENS"] = "128"
            run.command([sys.executable, "scripts/validate_checkpoint_resume.py", "--gpus", "1"],
                        timeout=1500)
            report["one_gpu_nf4_lora_interruption_resume_and_reload"] = "passed"
        write_json(run.path / "validation.json", report)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
