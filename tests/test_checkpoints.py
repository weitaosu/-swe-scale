import json

import pytest

from swe_study import common
from swe_study.checkpoints import latest_complete, mark_complete


def fixture_checkpoint(root, step, world=2):
    path = root / "adapter" / f"checkpoint-{step}"
    path.mkdir(parents=True)
    names = ["optimizer.pt", "scheduler.pt", "adapter_config.json", "adapter_model.safetensors"]
    names += [f"rng_state_{rank}.pth" for rank in range(world)] if world > 1 else ["rng_state.pth"]
    for name in names:
        (path / name).write_bytes(b"fixture payload")
    (path / "trainer_state.json").write_text(json.dumps({"global_step": step}))
    return path


def test_latest_checkpoint_skips_unfinished_and_corrupt_saves(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "ROOT", tmp_path)
    config = {"checkpoint_every_steps": 1}
    previous = fixture_checkpoint(tmp_path, 1)
    mark_complete(previous, config, 2, 1)
    current = fixture_checkpoint(tmp_path, 2)
    assert latest_complete("adapter", config, 2) == previous
    mark_complete(current, config, 2, 2)
    assert latest_complete("adapter", config, 2) == current
    (current / "optimizer.pt").write_bytes(b"interrupted write")
    assert latest_complete("adapter", config, 2) == previous


def test_completion_requires_every_rank_rng_and_same_configuration(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "ROOT", tmp_path)
    checkpoint = fixture_checkpoint(tmp_path, 1)
    (checkpoint / "rng_state_1.pth").unlink()
    with pytest.raises(ValueError, match="Incomplete checkpoint"):
        mark_complete(checkpoint, {}, 2, 1)
    (checkpoint / "rng_state_1.pth").write_bytes(b"restored")
    mark_complete(checkpoint, {}, 2, 1)
    with pytest.raises(ValueError, match="No matching complete"):
        latest_complete("adapter", {"different": True}, 2)
    with pytest.raises(ValueError, match="No matching complete"):
        latest_complete("adapter", {}, 1)


def test_single_gpu_completion_record(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "ROOT", tmp_path)
    checkpoint = fixture_checkpoint(tmp_path, 4, world=1)
    mark_complete(checkpoint, {}, 1, 4)
    assert latest_complete("adapter", {}, 1) == checkpoint


def test_recovery_profile_preserves_epoch_selection_and_global_batch():
    from test_cloud import module
    cloud = module("cloud")
    args = cloud.parser().parse_args(["train", "--config", "configs/cloud/qlora_7b_step_resume.yaml", "--gpus", "2"])
    config = cloud.resolved_config(args, 2)
    assert config["checkpoint_every_steps"] == 1
    assert config["eval_strategy"] == config["save_strategy"] == "epoch"
    assert config["gradient_accumulation_steps"] == 4
    assert config["effective_batch_size"] == 8
    assert config["save_total_limit"] == 3


def test_checkpoint_beyond_planned_steps_is_ignored(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "ROOT", tmp_path)
    config = {"max_steps": 1}
    first = fixture_checkpoint(tmp_path, 1)
    mark_complete(first, config, 2, 1)
    unexpected = fixture_checkpoint(tmp_path, 2)
    mark_complete(unexpected, config, 2, 2)
    assert latest_complete("adapter", config, 2) == first
