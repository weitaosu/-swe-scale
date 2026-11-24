#!/usr/bin/env python3
"""Portable, audited single-node A100 training and inference entry point."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from swe_study.common import (local_path, read_json, read_jsonl, setup_environment,
                              sha256_file, write_json)
from swe_study.audit import Run


def initialize(accounting=None):
    """Only initialize a fresh cloud bundle; never rebase the workstation budget."""
    baseline = ROOT / "manifests/storage_baseline.json"
    marker = ROOT / "manifests/cloud_environment.json"
    if (ROOT / "manifests/docker_storage.json").exists():
        raise RuntimeError("Machine-specific Docker accounting found. Extract the cloud bundle into a fresh directory.")
    if baseline.exists() or marker.exists():
        if not (baseline.exists() and marker.exists()):
            raise RuntimeError("Existing storage accounting is preserved. Extract the cloud bundle into a fresh directory.")
        saved = read_json(marker)
        if saved["project_root"] != str(ROOT) or saved["baseline_sha256"] != sha256_file(baseline):
            raise RuntimeError("Cloud directory moved or baseline changed. Use a fresh cloud bundle; do not reset the baseline.")
        if accounting and read_json(baseline).get("accounting_mode", "filesystem") != accounting:
            raise RuntimeError("Existing accounting mode is preserved; use a fresh directory to select another mode.")
        return
    disk = shutil.disk_usage(ROOT)
    if disk.free < 40 * 10**9:
        raise RuntimeError("At least 40 GB must be free before setup (25 GB floor plus installation reserve).")
    from swe_study.storage import project_bytes
    mode = accounting or "filesystem"
    write_json(baseline, {"schema_version": 1, "captured_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                         "project_root": str(ROOT), "project_initial_bytes": project_bytes() if mode == "project" else 0,
                         "accounting_mode": mode,
                         "initial_free_bytes": disk.free, "volume_total_bytes": disk.total,
                         "max_incremental_bytes": 70 * 10**9, "min_free_bytes": 25 * 10**9})
    write_json(marker, {"project_root": str(ROOT), "baseline_sha256": sha256_file(baseline),
                        "purpose": "Fresh cloud host accounting; no workstation baseline imported"})


def require_cloud():
    if not (ROOT / "manifests/cloud_environment.json").exists():
        raise RuntimeError("Run bash cloud/run.sh setup in an extracted cloud bundle first.")
    initialize()
    from swe_study.storage import StorageGuard
    return StorageGuard()


def study(run, *args, timeout=14400):
    return run.command([sys.executable, "scripts/study.py", *args], timeout=timeout,
                       guard=require_cloud())


def doctor(run, count=1):
    import torch
    from swe_study.preflight import run_preflight
    report = run_preflight(run)
    if not torch.cuda.is_available() or torch.cuda.device_count() < count:
        raise RuntimeError(f"Need {count} visible CUDA GPUs. Check the allocation and CUDA_VISIBLE_DEVICES.")
    devices = []
    for device in range(count):
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError(f"GPU {device} does not support BF16")
            props = torch.cuda.get_device_properties(device)
            # Exercise the actual installed NF4 CUDA kernel, not just imports.
            import bitsandbytes as bnb
            x = torch.randn(128, 128, device=f"cuda:{device}", dtype=torch.bfloat16)
            packed, state = bnb.functional.quantize_4bit(x, quant_type="nf4")
            restored = bnb.functional.dequantize_4bit(packed, state)
            if not torch.isfinite(restored).all(): raise RuntimeError("NF4 kernel produced non-finite values")
            devices.append({"index": device, "name": props.name,
                            "total_memory_bytes": props.total_memory,
                            "free_memory_bytes": torch.cuda.mem_get_info()[0], "nf4_kernel": "passed"})
            del x, packed, state, restored
            torch.cuda.empty_cache()
    run.event("cloud_gpu_checks", devices=devices)
    write_json(run.path / "cloud_gpu_checks.json", devices)
    return report


def setup(args):
    if sys.platform != "linux" or sys.version_info[:2] not in {(3, 11), (3, 12)}:
        raise RuntimeError("Use Linux x86_64 with Python 3.11 or 3.12; set PYTHON=/path/to/python3.12 for setup.")
    import platform
    if platform.machine() != "x86_64": raise RuntimeError("This pinned CUDA wheel setup targets Linux x86_64.")
    initialize(args.storage_accounting)
    from swe_study.storage import StorageGuard
    guard = StorageGuard()
    with Run("cloud_setup", vars(args)) as run:
        guard.check(15 * 10**9)
        env = ROOT / ".envs/train"
        python = env / "bin/python"
        if not python.exists():
            if env.exists() and any(env.iterdir()): raise RuntimeError("Incomplete environment preserved; use a fresh bundle.")
            run.command([sys.executable, "-m", "venv", "--copies", str(env)], timeout=180, guard=guard)
        run.command([python, "-m", "pip", "install", "--no-cache-dir", "torch==2.9.1",
                     "--index-url", "https://download.pytorch.org/whl/" + args.cuda], timeout=7200, guard=guard)
        run.command([python, "-m", "pip", "install", "--no-cache-dir", "-c", "configs/constraints_train.txt",
                     "-e", str(ROOT) + "[ml,test]"], timeout=7200, guard=guard)
        run.command([python, "-m", "pip", "check"], timeout=120, guard=guard)
        frozen = run.command([python, "-m", "pip", "freeze", "--all"], timeout=120, guard=guard)
        (ROOT / "manifests/cloud_train_packages.freeze.txt").write_text(
            Path(frozen["stdout_path"]).read_text(), encoding="utf-8")
        if not args.skip_gpu_check:
            run.command([python, "scripts/cloud.py", "doctor"], timeout=300, guard=guard)
        else:
            run.event("gpu_check_deferred", reason="Install on login node; run doctor in a GPU allocation")


def model_path(size):
    return f"models/Qwen2.5-Coder-{size}-Instruct"


@contextmanager
def training_lock(output):
    """An advisory lock prevents two launchers writing the same adapter."""
    import fcntl
    import hashlib
    folder = ROOT / ".cache/cloud_locks"
    folder.mkdir(parents=True, exist_ok=True)
    lock = folder / (hashlib.sha256(str(output).encode()).hexdigest() + ".lock")
    with lock.open("a") as handle:
        try: fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError(f"Another training launcher owns {output}") from None
        try: yield
        finally: fcntl.flock(handle, fcntl.LOCK_UN)


def resolved_config(args, world_size):
    import yaml
    cfg = yaml.safe_load(local_path(args.config or f"configs/cloud/qlora_{args.size.lower()}.yaml").read_text())
    overrides = {"output": args.output, "max_length": args.max_length, "learning_rate": args.learning_rate,
                 "num_train_epochs": args.epochs, "max_steps": args.max_steps,
                 "gradient_accumulation_steps": args.grad_accum}
    cfg.update({key: value for key, value in overrides.items() if value is not None})
    if args.epochs is not None and args.max_steps is None: cfg["max_steps"] = -1
    if args.grad_accum is None and "global_batch_size" in cfg:
        batch = cfg["global_batch_size"]
        if batch < world_size or batch % world_size:
            raise ValueError(f"global_batch_size={batch} must be divisible by GPU count={world_size}; set --grad-accum explicitly")
        cfg["gradient_accumulation_steps"] = batch // world_size
    if cfg["max_length"] <= 0 or cfg["gradient_accumulation_steps"] <= 0:
        raise ValueError("Context and gradient accumulation must be positive")
    interval = cfg.get("checkpoint_every_steps")
    if interval is not None and (type(interval) is not int or interval <= 0):
        raise ValueError("checkpoint_every_steps must be a positive integer")
    if cfg.get("num_train_epochs", 3) <= 0 or cfg.get("max_steps", -1) == 0:
        raise ValueError("Epochs must be positive and max_steps must be positive or -1")
    cfg["world_size"] = world_size
    cfg["effective_batch_size"] = world_size * cfg["gradient_accumulation_steps"]
    for name in ["model", "train_file", "validation_file", "output"]: local_path(cfg[name])
    return cfg


def train(args, run):
    import yaml
    count = args.gpus
    if count < 1: raise RuntimeError("No GPUs selected")
    cfg = resolved_config(args, count)
    if args.dry_run:
        print(yaml.safe_dump(cfg, sort_keys=False))
        return
    doctor(run, count)
    output = local_path(cfg["output"])
    if args.resume == "latest":
        from swe_study.checkpoints import latest_complete
        args.resume = latest_complete(output, cfg, count).relative_to(ROOT).as_posix()
        run.event("resume_checkpoint_selected", checkpoint=args.resume)
        print(f"Resuming complete checkpoint: {args.resume}")
    if args.resume and not local_path(args.resume).is_relative_to(output):
        raise ValueError("Resume checkpoint must be inside the selected output directory")
    with training_lock(output):
        if output.exists() and any(output.iterdir()) and not args.resume:
            raise ValueError("Output is nonempty; use --resume or a fresh --output")
        for name in ["train_file", "validation_file"]:
            if not local_path(cfg[name]).is_file(): raise FileNotFoundError("Run cloud prepare before training")
        if not local_path(Path(cfg["model"]) / "config.json").exists():
            raise FileNotFoundError("Model missing; run cloud prepare with the matching --size")
        plan_path = local_path(str(output.relative_to(ROOT)) + ".plan.json")
        plan = {"config": cfg, "train_sha256": sha256_file(local_path(cfg["train_file"])),
                "validation_sha256": sha256_file(local_path(cfg["validation_file"]))}
        if args.resume:
            if not plan_path.exists() or read_json(plan_path) != plan:
                raise ValueError("Exact resume requires the same saved config, GPU count, data, and output. Reuse the original options.")
        elif plan_path.exists():
            raise ValueError("An earlier training plan exists; use a fresh --output or resume its checkpoint")
        else:
            write_json(plan_path, plan)
        config = run.path / "resolved_config.yaml"
        config.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        run.event("cloud_training_plan", **plan)
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
                   f"--nproc-per-node={count}", "scripts/cloud.py", "train-worker", "--config",
                   config.relative_to(ROOT).as_posix()]
        if args.resume: command += ["--resume", args.resume]
        run.command(command, timeout=args.timeout_hours * 3600, guard=require_cloud())
        # A separate Python process verifies that the final saved adapter reloads.
        study(run, "validate-checkpoint", "--adapter", output.relative_to(ROOT).as_posix())
        run.event("cloud_training_and_reload_passed", output=str(output), gpu_count=count)
        print(f"Adapter ready: {output}")


def inference(args, run):
    import torch
    from transformers import set_seed
    from swe_study.training import load_model
    if args.prompt is not None:
        messages = ([{"role": "system", "content": args.system}] if args.system else [])
        messages += [{"role": "user", "content": args.prompt}]
    else:
        messages = json.loads(local_path(args.messages).read_text())
        if not isinstance(messages, list) or not messages: raise ValueError("Messages JSON must be a nonempty list")
    if args.temperature < 0 or args.max_new_tokens < 1: raise ValueError("Invalid generation settings")
    set_seed(args.seed)
    model, tokenizer = load_model(model_path(args.size), "nf4", args.adapter)
    model.eval();model.config.use_cache = True
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, add_special_tokens=False, return_tensors="pt").to("cuda")
    length = inputs["input_ids"].shape[1]
    if length + args.max_new_tokens > args.max_context:
        raise ValueError("Prompt plus requested output exceeds --max-context; no silent truncation")
    options = {"max_new_tokens": args.max_new_tokens, "do_sample": args.temperature > 0,
               "pad_token_id": tokenizer.pad_token_id}
    if args.temperature > 0: options.update(temperature=args.temperature, top_p=args.top_p)
    import time
    torch.cuda.synchronize();start = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        generated = model.generate(**inputs, **options)[0, length:]
    torch.cuda.synchronize()
    result = {"model": model_path(args.size), "adapter": args.adapter, "messages": messages,
              "text": tokenizer.decode(generated, skip_special_tokens=True), "prompt_tokens": length,
              "completion_tokens": len(generated), "generation_seconds": time.perf_counter() - start,
              "seed": args.seed, "temperature": args.temperature}
    write_json(run.path / "generation.json", result)
    if args.output:
        if local_path(args.output).exists(): raise FileExistsError("Inference output already exists; choose a fresh path")
        write_json(args.output, result)
    print(result["text"])


def check_distributed(args, run):
    """Random tiny Qwen + real NF4/LoRA/DDP, checkpoint and fresh reload."""
    import torch
    import yaml
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
    from swe_study.common import write_jsonl
    directory = run.path / "random-model"
    vocab = {word: i for i, word in enumerate(["[UNK]", "<|im_start|>", "<|im_end|>", "<pad>",
                                                "system", "user", "assistant", "fix", "issue", "done"])}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="<pad>",
                                       eos_token="<|im_end|>", additional_special_tokens=["<|im_start|>"])
    tokenizer.chat_template = "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
    tokenizer.save_pretrained(directory)
    torch.manual_seed(6158)
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=64, hidden_size=64, intermediate_size=128,
                            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                            max_position_embeddings=256, tie_word_embeddings=False,
                            bos_token_id=1, eos_token_id=2, pad_token_id=3))
    model.save_pretrained(directory, safe_serialization=True)
    rows = [{"instance_id": f"synthetic-{i}", "trajectory_id": f"synthetic-{i}",
             "messages": [{"role": "user", "content": "issue"}, {"role": "assistant", "content": "fix done"}],
             "synthetic_fixture": True} for i in range(16)]
    train_file, val_file = run.path / "train.jsonl", run.path / "validation.jsonl"
    write_jsonl(train_file, rows[:12]);write_jsonl(val_file, rows[12:])
    relative = lambda path: path.relative_to(ROOT).as_posix()
    cfg = {"model": relative(directory), "train_file": relative(train_file), "validation_file": relative(val_file),
           "output": relative(run.path / "adapter"), "max_length": 128, "max_train_examples": None,
           "max_eval_examples": None, "max_steps": 2, "gradient_accumulation_steps": 1,
           "learning_rate": 0.001, "lora_rank": 4, "lora_alpha": 8, "lora_dropout": 0,
           "save_steps": 1, "eval_steps": 1, "save_total_limit": 2,
           "seed": 6158, "quantization": "nf4", "target_only_logits": True, "synthetic_fixture": True}
    config = run.path / "synthetic.yaml"
    config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    train_args = parser().parse_args(["train", "--config", relative(config), "--gpus", str(args.gpus)])
    train(train_args, run)
    report = {"passed": True, "synthetic_fixture": True, "gpu_count": args.gpus,
              "scope": "Random tiny Qwen NF4/LoRA training, DDP when GPUs>1, save and fresh reload; no real-model feasibility claim"}
    write_json(run.path / "diagnostic.json", report)
    print(json.dumps(report, indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("setup", help="Create the pinned Linux CUDA environment")
    s.add_argument("--cuda", choices=["cu126", "cu128", "cu130"], default="cu126")
    s.add_argument("--skip-gpu-check", action="store_true", help="Install on a login node; doctor must run later in a GPU allocation")
    s.add_argument("--storage-accounting", choices=["filesystem", "project"])
    s = sub.add_parser("init", help="Initialize a fresh storage baseline before model downloads")
    s.add_argument("--storage-accounting", choices=["filesystem", "project"], default="filesystem")
    sub.add_parser("doctor", help="Check CUDA/BF16/NF4 and record hardware")
    s = sub.add_parser("check-distributed", help="Tiny random NF4/LoRA/DDP checkpoint diagnostic")
    s.add_argument("--gpus", type=int, default=1)
    s = sub.add_parser("prepare", help="Download pinned model/data; audit, split, tokenize")
    s.add_argument("--size", choices=["1.5B", "3B", "7B", "all"], default="7B")
    s.add_argument("--max-length", type=int, default=8192)
    s = sub.add_parser("train", help="Launch one training worker per selected GPU and validate reload")
    s.add_argument("--size", choices=["1.5B", "3B", "7B"], default="7B")
    s.add_argument("--config")
    s.add_argument("--gpus", type=int, default=1, help="Default: one GPU; select 2 or 4 to use more")
    s.add_argument("--output");s.add_argument("--resume", help="Complete checkpoint path, or latest for a verified completion record")
    s.add_argument("--max-length", type=int);s.add_argument("--epochs", type=float)
    s.add_argument("--max-steps", type=int);s.add_argument("--grad-accum", type=int)
    s.add_argument("--learning-rate", type=float)
    s.add_argument("--timeout-hours", type=int, default=72)
    s.add_argument("--dry-run", action="store_true")
    s = sub.add_parser("train-worker", help=argparse.SUPPRESS)
    s.add_argument("--config", required=True);s.add_argument("--resume")
    s = sub.add_parser("infer", help="Generate with the base model or a trained adapter")
    s.add_argument("--size", choices=["1.5B", "3B", "7B"], default="7B")
    s.add_argument("--adapter")
    prompt = s.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt");prompt.add_argument("--messages", help="Project-local JSON array of chat messages")
    s.add_argument("--system", default="You are a helpful coding assistant.")
    s.add_argument("--max-context", type=int, default=8192)
    s.add_argument("--max-new-tokens", type=int, default=512)
    s.add_argument("--temperature", type=float, default=0)
    s.add_argument("--top-p", type=float, default=0.95)
    s.add_argument("--seed", type=int, default=6158);s.add_argument("--output")
    s = sub.add_parser("serve", help="Loopback chat API for ordinary prompts or frozen-task evaluation")
    s.add_argument("--size", choices=["1.5B", "3B", "7B"], default="7B")
    s.add_argument("--adapter");s.add_argument("--port", type=int, default=8615)
    s.add_argument("--max-context", type=int, default=8192)
    s.add_argument("--max-new-tokens", type=int, default=2048)
    s.add_argument("--development-budget", action="store_true", help="Require frozen development task identity and budget")
    s.add_argument("--task-budget-file", help="Project-local frozen task identity file")
    s.add_argument("--paired", action="store_true", help="Expose base and LoRA model aliases from one adapter load")
    s.add_argument("--budget-ledger-dir", help="Persistent project-local per-condition token ledgers")
    s.add_argument('--context-policy',choices=['reject','history-window-v1'],default='reject',help='Explicit recorded history handling for frozen-task serving')
    return p


def main():
    args = parser().parse_args()
    setup_environment()
    if args.command == "setup": return setup(args)
    if args.command == "init":
        initialize(args.storage_accounting)
        print("Fresh cloud storage accounting is ready")
        return
    require_cloud().check()
    rank = os.environ.get("RANK", "0")
    stage = f"cloud_train_rank{rank}" if args.command == "train-worker" else "cloud_" + args.command
    with Run(stage, vars(args)) as run:
        if args.command == "doctor": doctor(run)
        elif args.command == "prepare":
            sizes = ["1.5B", "3B", "7B"] if args.size == "all" else [args.size]
            for size in sizes: study(run, "download", "--kind", "model", "--only", size)
            study(run, "download", "--kind", "dataset")
            study(run, "select-tasks");study(run, "prepare")
            for size in sizes:
                study(run, "tokenize", "--model", model_path(size), "--max-length", str(args.max_length))
        elif args.command == "train": train(args, run)
        elif args.command == "check-distributed": check_distributed(args, run)
        elif args.command == "train-worker":
            import torch
            from swe_study.training import train as fit
            try: fit(run, args.config, args.resume)
            except BaseException as error:
                # A failed rank must report its original exception before any
                # collective teardown, which can otherwise hide it for minutes.
                import traceback
                traceback.print_exc()
                sys.stderr.flush()
                run.event("training_worker_failed",rank=rank,error_type=type(error).__name__,error=str(error))
                raise
            else:
                if torch.distributed.is_initialized(): torch.distributed.destroy_process_group()
        elif args.command == "infer": inference(args, run)
        elif args.command == "serve":
            from swe_study.common import assert_local_port_available
            from swe_study.serving import serve
            assert_local_port_available(args.port)
            if args.development_budget and args.task_budget_file: raise ValueError("Choose one task-budget file option")
            serve(run, model_path(args.size), args.adapter, args.port, args.max_context, args.max_new_tokens,
                  task_file=args.task_budget_file or ("data/metadata/dev_task_budgets.jsonl" if args.development_budget else None),
                  paired=args.paired,budget_ledger_dir=args.budget_ledger_dir,context_policy=args.context_policy)


if __name__ == "__main__": main()
