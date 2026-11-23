"""Completion records distinguish resumable checkpoints from interrupted writes."""
from .common import canonical_hash, local_path, read_json, sha256_file, write_json


def checkpoint_files(path, world_size):
    path = local_path(path)
    names = ["optimizer.pt", "scheduler.pt", "trainer_state.json", "adapter_config.json"]
    names += ([f"rng_state_{rank}.pth" for rank in range(world_size)]
              if world_size > 1 else ["rng_state.pth"])
    names.append("adapter_model.safetensors" if (path / "adapter_model.safetensors").exists()
                 else "adapter_model.bin")
    missing = [name for name in names if not (path / name).is_file() or not (path / name).stat().st_size]
    if missing:
        raise ValueError(f"Incomplete checkpoint {path}: {missing}")
    return names


def mark_complete(path, config, world_size, step):
    path = local_path(path)
    names = checkpoint_files(path, world_size)
    if read_json(path / "trainer_state.json")["global_step"] != step:
        raise ValueError("Checkpoint Trainer step differs from completion record")
    record = {"schema_version": 1, "global_step": step, "world_size": world_size,
              "config_sha256": canonical_hash(config),
              "files_sha256": {name: sha256_file(path / name) for name in names}}
    write_json(path / "checkpoint_complete.json", record)
    return record


def latest_complete(output, config, world_size):
    """Check hashes, config and all ranks' RNG state; skip interrupted saves."""
    output = local_path(output)
    candidates = [path for path in output.glob("checkpoint-*")
                  if path.is_dir() and path.name.removeprefix("checkpoint-").isdigit()]
    for path in sorted(candidates, key=lambda p: int(p.name.split("-")[-1]), reverse=True):
        try:
            record = read_json(path / "checkpoint_complete.json")
            names = checkpoint_files(path, world_size)
            step = int(path.name.split("-")[-1])
            if config.get("max_steps", -1)>0 and step>config["max_steps"]:
                continue
            if (record["schema_version"] != 1 or record["world_size"] != world_size
                    or record["global_step"] != step
                    or record["config_sha256"] != canonical_hash(config)
                    or set(record["files_sha256"]) != set(names)
                    or read_json(path / "trainer_state.json")["global_step"] != step):
                continue
            if all(sha256_file(path / name) == record["files_sha256"][name] for name in names):
                return path
        except (OSError, ValueError, KeyError, TypeError):
            continue
    raise ValueError("No matching complete checkpoint found; preserve this output and inspect its logs.")
