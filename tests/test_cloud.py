"""Portability, data coverage and distributed launch contract checks (no large GPU jobs)."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from swe_study.common import ROOT
from swe_study.training import training_examples


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


@pytest.mark.parametrize("failure", [RuntimeError("synthetic worker failure"), MemoryError("synthetic memory failure"), None])
def test_worker_reports_original_failure_without_collective_teardown(monkeypatch, capsys, failure):
    import torch
    import swe_study.training
    cloud = module("cloud")
    calls = []
    events = []
    args = SimpleNamespace(command="train-worker", config="fixture.yaml", resume=None)
    monkeypatch.setattr(cloud, "parser", lambda: SimpleNamespace(parse_args=lambda: args))
    monkeypatch.setattr(cloud, "setup_environment", lambda: None)
    monkeypatch.setattr(cloud, "require_cloud", lambda: SimpleNamespace(check=lambda: None))
    class FixtureRun:
        def __init__(self, *args): pass
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def event(self, event, **fields): events.append((event, fields))
    monkeypatch.setattr(cloud, "Run", FixtureRun)
    def fit(*args):
        calls.append("fit")
        if failure is not None:
            raise failure
    monkeypatch.setattr(swe_study.training, "train", fit)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "destroy_process_group", lambda: calls.append("destroy"))
    if failure is None:
        cloud.main()
        assert calls == ["fit", "destroy"] and not events
    else:
        with pytest.raises(type(failure), match=str(failure)):
            cloud.main()
        assert calls == ["fit"]
        assert str(failure) in capsys.readouterr().err
        assert events[0][0] == "training_worker_failed"
        assert events[0][1]["error"] == str(failure)


def test_worker_error_patch_preserves_unknown_sources_and_is_idempotent():
    installer = module("patch_worker_errors")
    text = (ROOT / "scripts/cloud.py").read_text()
    assert installer.patch_text(text) == text
    previous = text.replace(installer.NEW, installer.OLD, 1)
    assert previous != text and installer.patch_text(previous) == text
    with pytest.raises(ValueError, match="Unfamiliar worker source preserved"):
        installer.patch_text("# user-modified worker\n")


def test_cloud_bundle_excludes_workstation_state_and_secrets():
    files = module("package_cloud").bundle_files()
    assert "scripts/cloud.py" in files and "cloud/train.slurm" in files
    assert "configs/cloud/qlora_7b.yaml" in files
    assert "manifests/dev_tasks.json" in files
    assert "manifests/planned_downloads_4090.json" in files
    assert "manifests/storage_baseline.json" not in files
    assert "manifests/docker_storage.json" not in files
    assert not any(name.startswith((".envs/", ".cache/", "models/", "data/", "results/", ".git/")) for name in files)
    assert ".env" not in files
    assert b"native Linux" in files["AGENTS.md"]
    plan = json.loads(files["manifests/planned_downloads_4090.json"])
    assert all(len(a["revision"]) == 40 for group in ["models", "datasets"] for a in plan[group])


@pytest.mark.parametrize("world,accum", [(1, 8), (2, 4), (4, 2), (8, 1)])
def test_distributed_launch_keeps_global_batch_fixed(world, accum):
    cloud = module("cloud")
    args = cloud.parser().parse_args(["train", "--size", "7B"])
    cfg = cloud.resolved_config(args, world)
    assert cfg["gradient_accumulation_steps"] == accum
    assert cfg["effective_batch_size"] == 8
    assert cfg["max_steps"] == -1 and cfg["num_train_epochs"] == 3
    assert cfg["max_train_examples"] is None and cfg["max_eval_examples"] is None


def test_nondivisible_gpu_counts_require_explicit_batch_choice():
    cloud = module("cloud")
    with pytest.raises(ValueError, match="divisible"):
        cloud.resolved_config(cloud.parser().parse_args(["train"]), 3)
    cfg = cloud.resolved_config(cloud.parser().parse_args(["train", "--grad-accum", "2"]), 3)
    assert cfg["effective_batch_size"] == 6


def test_cloud_context_and_schedule_overrides_are_applied():
    cloud = module("cloud")
    args = cloud.parser().parse_args(["train", "--size", "3B", "--max-length", "32768", "--epochs", "5",
                                      "--output", "checkpoints/cloud/another-run"])
    cfg = cloud.resolved_config(args, 4)
    assert cfg["max_length"] == 32768 and cfg["num_train_epochs"] == 5
    assert cfg["output"] == "checkpoints/cloud/another-run"
    assert "3B" in cfg["model"]
    assert args.gpus == 1  # User's A100 allocation preference: use only what is needed.


def test_cloud_initialization_refuses_workstation_baseline(tmp_path, monkeypatch):
    cloud = module("cloud")
    monkeypatch.setattr(cloud, "ROOT", tmp_path)
    folder = tmp_path / "manifests"
    folder.mkdir()
    baseline = folder / "storage_baseline.json"
    baseline.write_text('{"workstation":true}')
    original = baseline.read_bytes()
    with pytest.raises(RuntimeError, match="preserved"): cloud.initialize()
    assert baseline.read_bytes() == original


def test_cloud_initialization_is_idempotent_and_does_not_reset_budget(tmp_path, monkeypatch):
    cloud = module("cloud")
    monkeypatch.setattr(cloud, "ROOT", tmp_path)
    monkeypatch.setattr(cloud.shutil, "disk_usage", lambda _: SimpleNamespace(free=100 * 10**9, total=200 * 10**9))
    cloud.initialize()
    path = tmp_path / "manifests/storage_baseline.json"
    original = path.read_bytes()
    monkeypatch.setattr(cloud.shutil, "disk_usage", lambda _: SimpleNamespace(free=60 * 10**9, total=200 * 10**9))
    cloud.initialize()
    assert path.read_bytes() == original
    assert json.loads(original)["max_incremental_bytes"] == 70 * 10**9
    assert json.loads(original)["min_free_bytes"] == 25 * 10**9
    data = json.loads(original);data["initial_free_bytes"] = 150 * 10**9
    path.write_text(json.dumps(data))
    with pytest.raises(RuntimeError, match="baseline changed"): cloud.initialize()


def test_all_history_training_reports_every_overlength_exclusion(tmp_path):
    from test_pipeline import CharacterTokenizer
    rows = [{"instance_id": str(i), "messages": [{"role": "user", "content": "q"},
             {"role": "assistant", "content": "a" if i < 3 else "x" * 200}]} for i in range(5)]
    path = tmp_path / "source.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    retained, excluded = training_examples(CharacterTokenizer(), path, 100, None, 6158)
    assert len(retained) == 3 and len(excluded) == 2
    assert all(row["reason"].startswith("over_context:") for row in excluded)
    assert training_examples(CharacterTokenizer(), path, 100, None, 6158) == (retained, excluded)
    assert len(training_examples(CharacterTokenizer(), path, 100, 2, 6158)[0]) == 2


def test_cloud_training_lock_blocks_competing_writers(tmp_path, monkeypatch):
    cloud = module("cloud")
    monkeypatch.setattr(cloud, "ROOT", tmp_path)
    output = tmp_path / "adapter"
    with cloud.training_lock(output):
        with pytest.raises(RuntimeError, match="Another training launcher"):
            with cloud.training_lock(output): pass
    with cloud.training_lock(output): pass


def test_nf4_loader_binds_to_the_worker_local_rank(tmp_path, monkeypatch):
    import torch
    import transformers
    from swe_study.training import load_model
    (tmp_path / "config.json").write_text("{}")
    tokenizer = SimpleNamespace(pad_token_id=3)
    seen = {}
    monkeypatch.setenv("LOCAL_RANK", "2")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: seen.update(device=device))
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: tokenizer)
    def fake_model(*args, **kwargs):
        seen.update(kwargs)
        return "model"
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", fake_model)
    model, actual = load_model(tmp_path, quantization="none")
    assert model == "model" and actual is tokenizer
    assert seen["device"] == 2 and seen["device_map"] == {"": 2}
    assert seen["dtype"] == torch.bfloat16 and seen["local_files_only"]


def test_shared_filesystem_accounting_ignores_other_users_volume_growth(tmp_path, monkeypatch):
    import swe_study.storage as storage
    guard = storage.StorageGuard()
    guard.baseline = {"initial_free_bytes": 900e9, "min_free_bytes": 25e9,
                      "max_incremental_bytes": 70e9, "accounting_mode": "project", "project_initial_bytes": 1e9}
    monkeypatch.setattr(storage, "local_path", lambda _: tmp_path / "absent-docker.json")
    monkeypatch.setattr(storage.shutil, "disk_usage", lambda _: SimpleNamespace(free=100e9))
    monkeypatch.setattr(storage, "project_bytes", lambda: 11e9)
    assert guard.check()["conservative_volume_growth_bytes"] == 10e9
    monkeypatch.setattr(storage, "project_bytes", lambda: 72e9)
    with pytest.raises(storage.StorageLimit, match="70 GB"): guard.check()


def test_login_node_setup_can_defer_gpu_check():
    cloud = module("cloud")
    args = cloud.parser().parse_args(["setup", "--skip-gpu-check", "--storage-accounting", "project"])
    assert args.skip_gpu_check and args.storage_accounting == "project"


def test_source_snapshot_accepts_pre_1980_timestamps(tmp_path, monkeypatch):
    import os
    import zipfile
    import swe_study.audit as audit
    folder = tmp_path / "src"
    folder.mkdir()
    source = folder / "epoch_fixture.py"
    source.write_text("# Preserved source content\n")
    os.utime(source, (0, 0))
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    with audit.Run("unit_epoch_snapshot") as run:
        pass
    with zipfile.ZipFile(run.path / "source_snapshot.zip") as archive:
        assert archive.read("src/epoch_fixture.py") == source.read_bytes()
        assert archive.getinfo("src/epoch_fixture.py").date_time[0] == 1980


def test_cloud_archive_assigns_zip_safe_timestamps(tmp_path, monkeypatch):
    import sys
    import tarfile
    import uuid
    package = module("package_cloud")
    output = tmp_path / ("bundle-" + uuid.uuid4().hex + ".tar.gz")
    monkeypatch.setattr(sys, "argv", ["package_cloud.py", "--output", str(output)])
    package.main()
    with tarfile.open(output) as archive:
        assert all(member.mtime == 946684800 for member in archive.getmembers())
