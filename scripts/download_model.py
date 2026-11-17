#!/usr/bin/env python3
"""Download one pinned Qwen snapshot and record every downloaded asset's hash."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

MODELS = {
    "qwen4b": {"repo_id": "Qwen/Qwen3-4B-Instruct-2507", "revision": "cdbee75f17c01a7cc42f958dc650907174af0554"},
    "qwen17b": {"repo_id": "Qwen/Qwen3-1.7B", "revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"},
}
REQUEST_FILE = ".download_request.json"
MANIFEST_FILE = "model_manifest.json"


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def inventory(out: Path) -> list[dict]:
    assets = []
    for path in sorted(out.rglob("*")):
        relative = path.relative_to(out)
        if ".cache" in relative.parts or relative.as_posix() in {REQUEST_FILE, MANIFEST_FILE}:
            continue
        if len(relative.parts) == 1 and path.name.startswith(".model_manifest.") and path.name.endswith(".tmp"):
            continue
        if path.is_symlink():
            raise ValueError("Downloaded assets must be regular files for a portable model directory")
        if not path.is_file():
            continue
        assets.append({"file": relative.as_posix(), "bytes": path.stat().st_size, "sha256": digest(path)})
    names = {entry["file"] for entry in assets}
    if not {"config.json", "tokenizer_config.json"}.issubset(names) or not any(name.endswith(".safetensors") for name in names):
        raise ValueError("Snapshot is missing expected model/tokenizer assets")
    return assets


def read_regular_json(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Required regular metadata file is missing: {path.name}")
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Invalid metadata object: {path.name}")
    return value


def download_model(alias: str, out: Path, *, resume: bool = False, max_attempts: int = 3,
                   snapshot_fn=None, retry_wait=time.sleep) -> dict:
    """Resume only the same pinned request; verify completed snapshots read-only.

    snapshot_fn and retry_wait are injectable solely to support tiny offline tests.
    Hugging Face's local download cache handles interrupted asset continuation.
    """
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or not 1 <= max_attempts <= 5:
        raise ValueError("max_attempts must be an integer from 1 through 5")
    out = Path(out)
    if out.is_symlink():
        raise ValueError("Model output must not be a symlink")
    spec = MODELS[alias]
    request = {"alias": alias, **spec}
    if out.exists():
        if not resume:
            raise FileExistsError("Output already exists; use --resume for the same pinned download")
        if not out.is_dir():
            raise ValueError("Model output is not a directory")
        if read_regular_json(out / REQUEST_FILE) != request:
            raise ValueError("Existing download request does not match this alias/repository/revision")
        if (out / MANIFEST_FILE).exists() or (out / MANIFEST_FILE).is_symlink():
            manifest = read_regular_json(out / MANIFEST_FILE)
            if manifest.get("status") != "download_complete" or any(manifest.get(key) != value for key, value in request.items()):
                raise ValueError("Completed manifest does not match this pinned download request")
            assets = inventory(out)
            if manifest.get("files") != assets or manifest.get("total_bytes") != sum(item["bytes"] for item in assets):
                raise ValueError("Completed model asset inventory/hash mismatch; refusing to modify the snapshot")
            return {**manifest, "status": "verified_existing_download"}
    else:
        if resume:
            raise ValueError("--resume requires an existing directory with a matching download request")
        out.mkdir(parents=True, exist_ok=False)
        with (out / REQUEST_FILE).open("x", encoding="utf-8") as handle:
            json.dump(request, handle, indent=2, allow_nan=False)
            handle.write("\n")
    if snapshot_fn is None:
        from huggingface_hub import snapshot_download
        snapshot_fn = snapshot_download
    for attempt in range(1, max_attempts + 1):
        try:
            snapshot_fn(repo_id=spec["repo_id"], revision=spec["revision"], local_dir=str(out))
            break
        except Exception as exc:
            if attempt == max_attempts:
                raise
            print(f"Download attempt {attempt}/{max_attempts} failed ({type(exc).__name__}); retrying the same pinned request.",
                  file=sys.stderr)
            retry_wait(min(2 ** attempt, 8))
    assets = inventory(out)
    manifest = {"status": "download_complete", **request, "files": assets,
                "total_bytes": sum(entry["bytes"] for entry in assets),
                "downloader_code_sha256": digest(Path(__file__)),
                "note": "Pinned upstream model snapshot; this is not a warmed-up or trained STRIDE checkpoint."}
    # Publish a full manifest atomically without overwriting another completion.
    # Staging metadata may be recreated after a crash; model assets are untouched.
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=out,
                                     prefix=".model_manifest.", suffix=".tmp", delete=False) as handle:
        staged = Path(handle.name)
        json.dump(manifest, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(staged, out / MANIFEST_FILE)
    finally:
        staged.unlink(missing_ok=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Model directory, e.g. models/qwen4b")
    parser.add_argument("--resume", action="store_true", help="Resume a matching partial download, or verify an already complete snapshot")
    parser.add_argument("--max-attempts", type=int, choices=range(1, 6), default=3, help="Bounded download attempts per invocation (default: 3)")
    args = parser.parse_args()
    manifest = download_model(args.model, args.out, resume=args.resume, max_attempts=args.max_attempts)
    print(json.dumps({key: manifest[key] for key in ("status", "alias", "repo_id", "revision", "total_bytes")}, indent=2))


if __name__ == "__main__":
    main()
