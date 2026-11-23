# Cloud execution instructions

This is the portable single-node A100 source bundle. Run native Linux commands
from this directory; WSL and workstation absolute paths do not apply here.

- Use Python 3.11 or 3.12 to bootstrap, then `.envs/train/bin/python`.
- Read `README.md`; use `bash cloud/run.sh` for setup/data/train/infer/serve.
- Preserve the 70 GB incremental storage cap and 25 GB free-space floor.
  Initialize accounting only in a fresh bundle with `cloud/run.sh setup`.
- Keep model/data/cache/checkpoints/logs under this project. Preserve run history.
- Distributed training is single-node DDP, one process and one NF4 replica per
  visible GPU. Do not launch the orchestrator with torchrun or one Slurm task per GPU.
- Keep microbatch 1, assistant-only targets, BF16/NF4 and target-only logits.
- Do not silently truncate trajectories. Record context exclusions and coverage.
- Use exact checkpoint resumes with the same configuration, data and GPU count.
- Keep real GPU validation distinct from CPU/synthetic checks. Do not claim A100
  or multi-GPU performance until measured on that hardware.
- Model inference and chat serving do not execute tools or establish SWE-bench scores.
- Docker/agent grading is a separate workflow and needs a fresh physical Docker
  storage baseline before any project image operation. No cloud helper pulls images.
- Do not launch paid work unless the user asks; generating scripts is not a cloud
  job submission. Test with pytest `--capture=sys -p no:cacheprovider` and a local basetemp.
