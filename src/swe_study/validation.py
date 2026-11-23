"""Paired baseline/adapter validation on the completed run's exact held-out split."""
from contextlib import nullcontext
import math
from .common import ROOT, canonical_hash, local_path, read_json, read_jsonl, sha256_file, write_json


def recorded_trainer_loss(histories, world_size):
    """Reconstruct Trainer 5.5.3's microbatch-1, cross-rank token-normalized metric.

    Accelerate pads each final rank batch using the first histories, then drops
    the duplicate *loss entries*. Their token counts still entered the shared
    denominator before gathering. Preserve that historical aggregation exactly.
    """
    if type(world_size) is not int or world_size<1 or not histories:
        raise ValueError('Need histories and a positive training GPU count')
    if any(row['target_tokens']<=0 or not math.isfinite(row['loss']) for row in histories):
        raise ValueError('Invalid per-history loss or target count')
    total=0.0;size=len(histories)
    for start in range(0,size,world_size):
        group=[histories[(start+rank)%size] for rank in range(world_size)]
        denominator=sum(row['target_tokens'] for row in group)
        retained=min(world_size,size-start)
        total+=sum(world_size*row['loss']*row['target_tokens']/denominator for row in group[:retained])
    return total/size


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
            count=int((inputs['labels']!=-100).sum().item())
            result=super().compute_loss(model, assistant_target_inputs(inputs),
                                       return_outputs=return_outputs,
                                       num_items_in_batch=num_items_in_batch)
            loss=result[0] if return_outputs else result
            histories.append({'history_index':len(histories),'target_tokens':count,
                              'loss':float(loss.detach().float().mean().cpu())})
            return result

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
        histories=[]
        run.event("validation_condition_started", condition=condition, examples=len(rows))
        print(f"Evaluating {condition}: {len(rows)} complete validation histories", flush=True)
        with model.disable_adapter() if condition == "base" else nullcontext():
            metrics = trainer.evaluate(metric_key_prefix=condition)
        value = metrics[f"{condition}_loss"]
        if not math.isfinite(value):
            raise ValueError("Non-finite validation loss")
        results[condition] = dict(metrics)
        if len(histories)!=len(rows):raise ValueError('Per-history validation coverage differs')
        write_json(run.path / f"{condition}_history_losses.json",histories)
        results[condition]['history_losses']=histories
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
    world_size=int(metadata.get('gpu_count',cfg.get('world_size',1)))
    reconstructed=recorded_trainer_loss(results['adapter']['history_losses'],world_size)
    reproduced=recorded is not None and math.isclose(reconstructed,recorded,rel_tol=.001,abs_tol=.0002)
    reference={'passed':reproduced,'training_gpu_count':world_size,
               'recorded_loss':recorded,'reconstructed_training_aggregation_loss':reconstructed,
               'canonical_single_gpu_history_mean':adapted,
               'explanation':'Trainer 5.5.3 averages tokens across ranks within each evaluation step; single-GPU results use a mean over histories. Final padded ranks affect the historical denominator.'}
    write_json(run.path/'training_metric_reproduction.json',reference)
    report = {**identity, "base_validation_loss": base, "adapter_validation_loss": adapted,
              "adapter_minus_base": adapted-base,
              "relative_loss_reduction_percent": 100*(base-adapted)/base,
              "recorded_training_best_eval_loss": recorded, "adapter_matches_recorded_loss": matches,
              "training_metric_reproduction":reference,
              "comparison_scope": "Held-out assistant response prediction; not a SWE-bench score",
              "condition_metrics": {key:{k:v for k,v in metrics.items() if k!='history_losses'}
                                    for key,metrics in results.items()}}
    write_json(run.path / "comparison.json", report)
    print(__import__("json").dumps(report, indent=2), flush=True)
    guard.snapshot()
    if not reproduced:
        raise ValueError("Adapter validation does not reproduce its recorded training metric after matching GPU aggregation; inspect before comparing quality")
    return report
