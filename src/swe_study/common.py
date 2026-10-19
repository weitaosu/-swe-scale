from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def local_path(value: str | Path) -> Path:
    p = Path(value)
    p = (p if p.is_absolute() else ROOT / p).resolve()
    if not p.is_relative_to(ROOT.resolve()):
        raise ValueError(f"Output or project input must stay within {ROOT}: {p}")
    return p


def setup_environment() -> None:
    folders = {
        "HF_HOME": ".cache/huggingface", "HF_HUB_CACHE": ".cache/huggingface/hub",
        "HF_DATASETS_CACHE": ".cache/datasets", "HF_XET_CACHE": ".cache/huggingface/xet",
        "PIP_CACHE_DIR": ".cache/pip", "UV_CACHE_DIR": ".cache/uv",
        "TMPDIR": ".cache/tmp", "TEMP": ".cache/tmp", "TMP": ".cache/tmp",
        "TORCH_HOME": ".cache/torch", "CUDA_CACHE_PATH": ".cache/cuda",
        "TRITON_CACHE_DIR": ".cache/triton", "TORCHINDUCTOR_CACHE_DIR": ".cache/inductor",
    }
    for key, folder in folders.items():
        p = ROOT / folder
        p.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(p)
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def read_json(path: str | Path):
    return json.loads(local_path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, data) -> None:
    p = local_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_name(p.name + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    temp.replace(p)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def read_jsonl(path: str | Path):
    with local_path(path).open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.strip():
                yield i, json.loads(line)


def write_jsonl(path: str | Path, rows) -> None:
    p = local_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_name(p.name + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    temp.replace(p)
