"""Paired baseline/adapter validation on the completed run's exact held-out split."""
from contextlib import nullcontext
import math
from .common import ROOT, canonical_hash, local_path, read_json, read_jsonl, sha256_file, write_json


def validation_metadata(adapter):
    adapter = local_path(adapter)
    metadata = read_json(adapter / "experiment.json")
    cfg = metadata["config"]
    source = local_path(cfg["validation_file"])
    if sha256_file(source) != metadata["validation_source_sha256"]:
        raise ValueError("Validation file differs from the completed training run")
    if not cfg.get("target_only_logits") or cfg.get("quantization") != "nf4":
        raise ValueError("Expected the recorded NF4/assistant-target training configuration")
    train_ids = {row.get("instance_id") for _, row in read_jsonl(cfg["train_file"])}
    val_ids = {row.get("instance_id") for _, row in read_jsonl(source)}
    if train_ids & val_ids:
        raise ValueError("Training and validation task identities overlap")
    return metadata


def evaluate_conditions(run, model, tokenizer, rows, *, use_cpu=False):
    """Use the same Trainer loss/reduction and batch size as recorded training."""
    import torch
    from transformers import Trainer, TrainingArguments
    from .training import assistant_target_inputs
    from .tokenization import SupervisedCollator

    class ValidationTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            return super().compute_loss(model, assistant_target_inputs(inputs),
                                        return_outputs=return_outputs,
                                        num_items_in_batch=num_items_in_batch)

    args = TrainingArguments(output_dir=str(run.path / "trainer"), use_cpu=use_cpu,
                             bf16=not use_cpu, per_device_eval_batch_size=1,
                             dataloader_num_workers=0, remove_unused_columns=False,
                             label_names=["labels"], prediction_loss_only=True,
                             report_to="none", disable_tqdm=True, seed=6158)
    trainer = ValidationTrainer(model=model, args=args, eval_dataset=rows,
                                data_collator=SupervisedCollator(tokenizer.pad_token_id),
                                processing_class=tokenizer)
    results = {}
    # disable_adapter restores the original frozen base weights, including the
    # same unquantized-layer precision, without loading a second model copy.
    for condition in ("base", "adapter"):
        run.event("validation_condition_started", condition=condition, examples=len(rows))
        print(f"Evaluating {condition}: {len(rows)} complete validation histories", flush=True)
        with model.disable_adapter() if condition == "base" else nullcontext():
            metrics = trainer.evaluate(metric_key_prefix=condition)
        value = metrics[f"{condition}_loss"]
        if not math.isfinite(value):
            raise ValueError("Non-finite validation loss")
        results[condition] = metrics
        write_json(run.path / f"{condition}_metrics.json", metrics)
        run.event("validation_condition_finished", condition=condition, metrics=metrics)
    if trainer.optimizer is not None or trainer.state.global_step != 0:
        raise AssertionError("Validation unexpectedly created optimizer updates")
    return results


def run_validation(run, adapter, allow_synthetic=False):
    metadata = validation_metadata(adapter)
    if metadata.get("synthetic_fixture") and not allow_synthetic:
        raise ValueError("Synthetic fixture requires --allow-synthetic; it is not a quality evaluation")
    import torch
    from .audit import versions
    from .memory_head import enable_chunked_head_loss
    from .storage import StorageGuard
    from .training import load_model, training_examples
    if int(__import__("os").environ.get("WORLD_SIZE", "1")) != 1 or torch.cuda.device_count() != 1:
        raise ValueError("Run this paired comparison in a one-GPU allocation, without torchrun")
    guard = StorageGuard(); guard.check(128 * 1024**2)
    cfg = metadata["config"]
    model, tokenizer = load_model(cfg["model"], "nf4", adapter)
    model.config.use_cache = False
    enable_chunked_head_loss(model, 128)
    rows, exclusions = training_examples(tokenizer, cfg["validation_file"], cfg["max_length"],
                                         cfg.get("max_eval_examples", 16), cfg.get("seed", 6158))
    if len(rows) != metadata["validation_examples"]:
        raise ValueError("Retained validation count differs from the completed training run")
    selection = [{key: row[key] for key in ("instance_id", "trajectory_id", "rendered_sha256", "target_tokens")}
                 | {"processed_tokens": len(row["input_ids"])} for row in rows]
    write_json(run.path / "validation_selection.json", {"examples": selection, "excluded": exclusions})
    identity = {"validation_source_sha256": metadata["validation_source_sha256"],
                "selection_sha256": canonical_hash(selection), "examples": len(rows),
                "max_length": cfg["max_length"], "seed": cfg.get("seed",6158),
                "aggregation": "Trainer mean over histories; per-history assistant causal cross-entropy",
                "model": cfg["model"], "adapter": str(local_path(adapter).relative_to(ROOT)),
                "adapter_weights_sha256": sha256_file(local_path(adapter) / "adapter_model.safetensors"),
                "synthetic_fixture": metadata.get("synthetic_fixture", False), "versions": versions()}
    write_json(run.path / "evaluation_identity.json", identity)
    results = evaluate_conditions(run, model, tokenizer, rows)
    base, adapted = results["base"]["base_loss"], results["adapter"]["adapter_loss"]
    recorded = metadata.get("best_eval_loss")
    matches = recorded is not None and math.isclose(adapted, recorded, rel_tol=.001, abs_tol=.0002)
    report = {**identity, "base_validation_loss": base, "adapter_validation_loss": adapted,
              "adapter_minus_base": adapted-base,
              "relative_loss_reduction_percent": 100*(base-adapted)/base,
              "recorded_training_best_eval_loss": recorded, "adapter_matches_recorded_loss": matches,
              "comparison_scope": "Held-out assistant response prediction; not a SWE-bench score",
              "condition_metrics": results}
    write_json(run.path / "comparison.json", report)
    print(__import__("json").dumps(report, indent=2), flush=True)
    guard.snapshot()
    if not matches:
        raise ValueError("Adapter validation differs from its recorded best loss; inspect before comparing quality")
    return report
