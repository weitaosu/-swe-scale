"""Shared manifests remain complete when independent runs publish concurrently."""
from concurrent.futures import ThreadPoolExecutor
import json

import pytest

from swe_study import common


def test_concurrent_json_writers_publish_complete_documents(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "ROOT", tmp_path)
    target = tmp_path / "shared.json"

    def publish(worker):
        for iteration in range(40):
            common.write_json(target, {"worker": worker, "iteration": iteration, "payload": "x" * 2048})
            document = json.loads(target.read_text())
            assert document["payload"] == "x" * 2048

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(publish, range(4)))
    assert json.loads(target.read_text())["iteration"] == 39
    assert list(tmp_path.glob("*.tmp")) == []


def test_failed_json_serialization_preserves_existing_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "ROOT", tmp_path)
    target = tmp_path / "shared.json"
    common.write_json(target, {"status": "previous"})
    circular = []
    circular.append(circular)
    with pytest.raises(ValueError, match="Circular reference"):
        common.write_json(target, circular)
    assert json.loads(target.read_text()) == {"status": "previous"}
    assert list(tmp_path.glob("*.tmp")) == []
