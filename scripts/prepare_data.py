#!/usr/bin/env python3
"""Download pinned English DAPO and freeze fresh disjoint train/val/test JSONs.

Only aggregate counts and file hashes are printed. Run --help without ML packages.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import unicodedata

DATASET = "open-r1/DAPO-Math-17k-Processed"
REVISION = "31dd309567e3da778038cc87d868b6097a3ccf68"
SOURCE_FILENAME = "en/train-00000-of-00001.parquet"
SOURCE_SHA256 = "40f672fe8b6dbeee953ae5acc75a38e99144e682450e07c625c87c734ec76285"


def digest(path: Path | str) -> str:
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def canonical(text: str) -> str:
    """Normalize presentation only; this does not detect semantic duplicates."""
    value = unicodedata.normalize("NFKC", text).casefold()
    value = re.sub(r"\\(?:left|right|displaystyle)\b", "", value)
    value = re.sub(r"\\(?:quad|qquad)\b|\\[,!;: ]", "", value)
    value = value.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    for delimiter in (r"\(", r"\)", r"\[", r"\]", "$"):
        value = value.replace(delimiter, "")
    return "".join(value.split())


def question_hash(text: str) -> str:
    return hashlib.sha256(canonical(text).encode("utf-8")).hexdigest()


def encode(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def integer_answer(value: object, row_number: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError(f"Source row {row_number} has a noninteger answer")
    if not re.fullmatch(r"[+-]?[0-9]+", str(value).strip()):
        raise ValueError(f"Source row {row_number} has a noninteger answer")
    return int(str(value).strip())


def source_records(raw: list[dict]) -> list[dict]:
    """Keep the original question and stable ID; canonicalize integer answers."""
    records = []
    for index, row in enumerate(raw):
        question = row.get("prompt")
        if not isinstance(question, str) or not canonical(question):
            raise ValueError(f"Source row {index} has an empty or non-string prompt")
        try:
            answer = integer_answer(row["reward_model"]["ground_truth"], index)
            solution = integer_answer(row["solution"], index)
            uid = row["extra_info"]["index"]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Source row {index} is missing required metadata") from exc
        if answer != solution:
            raise ValueError(f"Source row {index} has conflicting answer fields")
        if isinstance(uid, bool) or not isinstance(uid, (str, int)) or not str(uid).strip():
            raise ValueError(f"Source row {index} has an invalid source ID")
        records.append({"id": str(uid), "question": question, "ground_truth": answer})
    if len({row["id"] for row in records}) != len(records):
        raise ValueError("Duplicate source IDs")
    return records


def exclusions_from(payload: object) -> tuple[set[str], set[str]]:
    """Read question lists/records, explicit DAPO source IDs, and chat prompts.

    Generic external benchmark `id` fields are deliberately not interpreted as
    DAPO IDs. Use `source_ids` or `source_id` when referring to this pinned source.
    """
    ids: set[str] = set()
    hashes: set[str] = set()

    def add_question(value: object) -> None:
        if isinstance(value, str) and canonical(value):
            hashes.add(question_hash(value))
        elif isinstance(value, list):
            for message in value:
                if isinstance(message, dict) and message.get("role") == "user":
                    add_question(message.get("content"))

    def visit(value: object, root_list: bool = False) -> None:
        if isinstance(value, list):
            for child in value:
                if isinstance(child, str) and root_list:
                    add_question(child)
                elif isinstance(child, (dict, list)):
                    visit(child, root_list=root_list)
        elif isinstance(value, dict):
            for key in ("question", "problem", "prompt"):
                add_question(value.get(key))
            for key in ("source_id", "source_ids"):
                item = value.get(key)
                items = item if isinstance(item, list) else [item]
                for uid in items:
                    if isinstance(uid, (str, int)) and not isinstance(uid, bool):
                        ids.add(str(uid))
            if value.get("dataset") == "dapo" and "id" in value:
                uid = value["id"]
                if isinstance(uid, (str, int)) and not isinstance(uid, bool):
                    ids.add(str(uid))
            for key, child in value.items():
                if key in ("source_id", "source_ids", "question", "problem", "prompt"):
                    continue
                if isinstance(child, (dict, list)):
                    visit(child, root_list=key in ("questions", "problems"))

    visit(payload, root_list=True)
    return ids, hashes


def load_exclusions(path: Path) -> object:
    with path.open(encoding="utf-8") as handle:
        if path.suffix.lower() == ".jsonl":
            return [json.loads(line) for line in handle if line.strip()]
        return json.load(handle)


def select_splits(
    records: list[dict], excluded_ids: set[str], excluded_hashes: set[str],
    train_count: int, val_count: int, test_count: int, seed: int,
) -> tuple[dict[str, list[dict]], dict]:
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 1
           for n in (train_count, val_count, test_count)):
        raise ValueError("Train, validation, and test counts must all be positive integers")
    ids = [row["id"] for row in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate source IDs")
    frequencies = Counter(question_hash(row["question"]) for row in records)
    rejected: Counter = Counter()
    candidates = []
    for row in records:
        fingerprint = question_hash(row["question"])
        if row["id"] in excluded_ids:
            rejected["excluded_source_id"] += 1
        elif fingerprint in excluded_hashes:
            rejected["excluded_normalized_question"] += 1
        elif frequencies[fingerprint] > 1:
            rejected["duplicate_normalized_source_question"] += 1
        else:
            candidates.append(dict(row))
    required = train_count + val_count + test_count
    if len(candidates) < required:
        raise ValueError(f"Only {len(candidates)} eligible unique questions remain; {required} requested")
    # Source order is fixed by the pinned parquet. No outcome-based selection.
    random.Random(seed).shuffle(candidates)
    splits = {
        "test": candidates[:test_count],
        "val": candidates[test_count:test_count + val_count],
        "train": candidates[test_count + val_count:required],
    }
    selected = [row for split in splits.values() for row in split]
    selected_ids = {row["id"] for row in selected}
    selected_hashes = {question_hash(row["question"]) for row in selected}
    if len(selected_ids) != required or len(selected_hashes) != required:
        raise ValueError("Internal split overlap or duplicate")
    if selected_ids & excluded_ids or selected_hashes & excluded_hashes:
        raise ValueError("Internal exclusion overlap")
    return splits, {"eligible_pool": len(candidates), "rejected_rows": dict(rejected)}


def write_outputs(out: Path, splits: dict[str, list[dict]], metadata: dict) -> dict:
    """Claim a new output directory and never change an existing dataset."""
    out.mkdir(parents=True, exist_ok=False)
    split_meta = {}
    for name in ("train", "val", "test"):
        path = out / f"{name}.json"
        with path.open("xb") as handle:
            handle.write(encode(splits[name]))
        split_meta[name] = {"file": path.name, "count": len(splits[name]), "sha256": digest(path)}
    manifest = dict(metadata, splits=split_meta, status="fresh_frozen_splits")
    with (out / "manifest.json").open("xb") as handle:
        handle.write(encode(manifest))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data"), help="New output directory; must not exist")
    parser.add_argument("--train-count", type=int, default=2048)
    parser.add_argument("--val-count", type=int, default=256)
    parser.add_argument("--test-count", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source-parquet", type=Path, help="Already-downloaded pinned English parquet; hash verified")
    parser.add_argument("--exclude-json", type=Path, action="append", default=[], help="JSON/JSONL benchmark or prior-use questions; repeatable")
    args = parser.parse_args()
    if args.out.exists() or args.out.is_symlink():
        parser.error("Output directory already exists; choose a new --out path")
    if min(args.train_count, args.val_count, args.test_count) < 1:
        parser.error("All split counts must be positive")
    source = args.source_parquet
    if source is None:
        from huggingface_hub import hf_hub_download
        source = Path(hf_hub_download(repo_id=DATASET, repo_type="dataset", filename=SOURCE_FILENAME,
                                      revision=REVISION))
    if digest(source) != SOURCE_SHA256:
        raise ValueError("Pinned English DAPO parquet SHA-256 mismatch; refusing to use this source")
    import pyarrow.parquet as pq
    records = source_records(pq.read_table(source).to_pylist())
    excluded_ids, excluded_hashes = set(), set()
    examined = []
    for path in args.exclude_json:
        ids, hashes = exclusions_from(load_exclusions(path))
        if not ids and not hashes:
            raise ValueError(f"No supported exclusion records found in {path.name}; see docs/DATA.md")
        excluded_ids.update(ids)
        excluded_hashes.update(hashes)
        examined.append({"file": path.name, "sha256": digest(path), "source_id_count": len(ids),
                         "normalized_question_count": len(hashes)})
    by_id = {row["id"]: row for row in records}
    excluded_hashes.update(question_hash(by_id[uid]["question"]) for uid in excluded_ids if uid in by_id)
    splits, selection = select_splits(records, excluded_ids, excluded_hashes,
                                      args.train_count, args.val_count, args.test_count, args.seed)
    metadata = {
        "dataset": DATASET, "revision": REVISION, "config": "en", "source_split": "train",
        "source_filename": SOURCE_FILENAME, "source_sha256": SOURCE_SHA256,
        "source_url": f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{SOURCE_FILENAME}",
        "source_rows": len(records), "data_seed": args.seed, **selection,
        "selection": "Shuffle unique eligible records in pinned source order; take test, then val, then train. No model-outcome filtering.",
        "exclusions": {"files": examined, "source_id_count": len(excluded_ids),
                       "normalized_question_count": len(excluded_hashes)},
        "verification": {"cross_split_id_overlap": 0, "cross_split_normalized_question_overlap": 0,
                         "within_split_normalized_duplicates": 0, "selected_exclusion_overlap": 0},
        "row_schema": {"id": "original source ID as string", "question": "original English prompt unchanged",
                       "ground_truth": "integer answer checked against both source answer fields"},
        "preparation_code_sha256": digest(__file__),
        "limitations": [
            "These are new splits, not the unavailable original CSCC train/eval payloads.",
            "Validation and test are holdouts from the upstream training split, not external benchmark test sets.",
            "Normalized text matching is not proof of semantic decontamination or freedom from pretraining overlap.",
            "Only supplied exclusion files are checked; supply every chosen external benchmark and prior-use set before splitting.",
            "Four external evaluation benchmarks have not been selected or downloaded by this script.",
            "No upstream dataset license is inferred or replaced by this code package's license.",
        ],
    }
    manifest = write_outputs(args.out, splits, metadata)
    print(json.dumps({"status": manifest["status"], "source_rows": len(records),
                      "eligible_pool": selection["eligible_pool"], "splits": manifest["splits"],
                      "manifest": str(args.out / "manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
