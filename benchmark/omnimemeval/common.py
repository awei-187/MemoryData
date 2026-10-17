"""Stable identities and atomic artifacts for the OmniMemEval integration."""
import hashlib
import json
import os
from pathlib import Path

PROTOCOL = "memorydata_omnimemeval_v1"
UPSTREAM = "a5c8b4b"
REPO_ROOT = Path(__file__).resolve().parents[2]


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")).hexdigest()


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def require_identity(path, identity):
    path = Path(path)
    if path.exists():
        with path.open(encoding="utf-8") as stream:
            existing = json.load(stream)
        if existing != identity:
            raise ValueError(f"Incompatible saved run at {path}; use a new artifact directory.")
    else:
        atomic_json(path, identity)
