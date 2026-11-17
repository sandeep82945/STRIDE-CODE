"""Verify this source package and its unchanged archived training modules."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path)
    args = parser.parse_args()
    manifest = json.loads((ROOT / "RELEASE_MANIFEST.json").read_text())
    for name, expected in manifest["source_files"].items():
        path = ROOT / name
        if path.is_symlink() or not path.is_file() or sha256(path) != expected:
            raise SystemExit(f"Source hash mismatch: {name}")
    for name, original_hash in manifest["archived_training_source_sha256"].items():
        if sha256(ROOT / name) != original_hash:
            raise SystemExit(f"Archived training implementation changed: {name}")
    print(f"Verified {len(manifest['source_files'])} package files")
    print(f"Verified {len(manifest['archived_training_source_sha256'])} unchanged archived training modules")
    if args.train_data:
        expected = "d319351d80df073e947576189f41c23b549584ecca4bad660a2b752427f429cc"
        if sha256(args.train_data) != expected:
            raise SystemExit("Training data hash differs from the recorded 2,048-question selection")
        rows = json.loads(args.train_data.read_text())
        if not isinstance(rows, list) or len(rows) != 2048:
            raise SystemExit("Expected 2,048 training questions")
        print("Verified the recorded 2,048-question training data")


if __name__ == "__main__":
    main()
