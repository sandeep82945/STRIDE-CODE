"""Offline checks for split integrity and acquisition provenance, using synthetic data."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("handoff_prepare_data", ROOT / "scripts" / "prepare_data.py")
data = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(data)
MODEL_SPEC = importlib.util.spec_from_file_location("handoff_download_model", ROOT / "scripts" / "download_model.py")
models = importlib.util.module_from_spec(MODEL_SPEC)
MODEL_SPEC.loader.exec_module(models)


def raw_rows(count=30):
    return [{"prompt": f"Calculate {index} + 10.", "solution": str(index + 10),
             "reward_model": {"ground_truth": str(index + 10)}, "extra_info": {"index": index}}
            for index in range(count)]


def fake_model_snapshot(**kwargs):
    """Tiny acquisition fixture, not loadable model weights."""
    out = Path(kwargs["local_dir"])
    (out / "config.json").write_text('{"model_type":"synthetic-test"}\n', encoding="utf-8")
    (out / "tokenizer_config.json").write_text('{}\n', encoding="utf-8")
    (out / "model.safetensors").write_bytes(b"synthetic fixture bytes")
    cache = out / ".cache" / "huggingface"
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "download.lock").write_text("fixture metadata", encoding="utf-8")


class DataTests(unittest.TestCase):
    def test_training_uses_clean_question_not_upstream_instruction_wrapper(self):
        import trainer
        from tests.test_actor_cpu_integration import AsciiTokenizer
        raw = raw_rows(1)
        prefix = ('Solve the following math problem step by step. The last line of your response '
                  'should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n')
        suffix = '\n\nRemember to put your answer on its own line after "Answer:".'
        raw[0]['source_prompt'] = [{'role': 'user', 'content': prefix + raw[0]['prompt'] + suffix}]
        row = data.source_records(raw)[0]
        tok = AsciiTokenizer()
        ids = trainer.render(tok, row)
        self.assertEqual(tok.decode(ids), trainer.INSTRUCTION + raw[0]['prompt'])
        self.assertNotIn(prefix, tok.decode(ids))
        self.assertNotIn(suffix, tok.decode(ids))
        self.assertEqual(trainer.render(tok, dict(row, ground_truth=999)), ids)
        self.assertEqual(set(row), {'id', 'question', 'ground_truth'})

    def test_original_questions_integer_answer_contract(self):
        raw = raw_rows(2)
        raw[0]["prompt"] = "  Question WITH original whitespace?\n"
        raw[0]["reward_model"]["ground_truth"] = " +010 "
        rows = data.source_records(raw)
        self.assertEqual(rows[0], {"id": "0", "question": raw[0]["prompt"], "ground_truth": 10})
        self.assertIs(type(rows[0]["ground_truth"]), int)

    def test_source_validation_rejects_conflicts_and_bad_answers(self):
        for mutation in (
            lambda rows: rows[0].update(solution="99"),
            lambda rows: rows[0]["reward_model"].update(ground_truth="3.5"),
            lambda rows: rows[0]["reward_model"].update(ground_truth=True),
            lambda rows: rows[0].update(prompt="  \n"),
            lambda rows: rows[0]["extra_info"].update(index=None),
            lambda rows: rows[1]["extra_info"].update(index=0),
        ):
            raw = raw_rows(2)
            mutation(raw)
            with self.assertRaises(ValueError):
                data.source_records(raw)

    def test_presentation_normalization(self):
        left = r"Find $\left( \dfrac{１}{2} \right)$ NOW."
        right = r"find (\frac{1}{2}) now."
        self.assertEqual(data.question_hash(left), data.question_hash(right))
        self.assertNotEqual(data.question_hash("Calculate 1 + 2"), data.question_hash("Calculate 1 - 2"))

    def test_deterministic_disjoint_splits_and_no_input_mutation(self):
        rows = data.source_records(raw_rows())
        original = copy.deepcopy(rows)
        first, stats = data.select_splits(rows, set(), set(), 12, 4, 5, 42)
        repeated, repeated_stats = data.select_splits(rows, set(), set(), 12, 4, 5, 42)
        changed, _ = data.select_splits(rows, set(), set(), 12, 4, 5, 43)
        self.assertEqual(first, repeated)
        self.assertEqual(stats, repeated_stats)
        self.assertNotEqual(first, changed)
        self.assertEqual(rows, original)
        self.assertEqual({name: len(part) for name, part in first.items()}, {"train": 12, "val": 4, "test": 5})
        self.assertEqual(len({row["id"] for part in first.values() for row in part}), 21)
        self.assertEqual(len({data.question_hash(row["question"]) for part in first.values() for row in part}), 21)

    def test_duplicates_and_exclusions_removed_before_split(self):
        rows = data.source_records(raw_rows())
        rows[1]["question"] = " CALCULATE 0 + 10. "
        excluded_hash = data.question_hash(rows[3]["question"])
        splits, stats = data.select_splits(rows, {"2"}, {excluded_hash}, 10, 4, 4, 42)
        selected_ids = {row["id"] for part in splits.values() for row in part}
        self.assertFalse(selected_ids & {"0", "1", "2", "3"})
        self.assertEqual(stats["eligible_pool"], 26)
        self.assertEqual(stats["rejected_rows"], {"duplicate_normalized_source_question": 2,
                         "excluded_source_id": 1, "excluded_normalized_question": 1})

    def test_insufficient_pool_and_invalid_counts_fail(self):
        rows = data.source_records(raw_rows(5))
        with self.assertRaises(ValueError):
            data.select_splits(rows, set(), set(), 3, 2, 1, 42)
        for count in (0, -1, True, 2.5):
            with self.assertRaises(ValueError):
                data.select_splits(rows, set(), set(), count, 1, 1, 42)

    def test_external_exclusions_do_not_confuse_foreign_numeric_ids(self):
        payload = [{"id": 2, "problem": "External question"},
                   {"source_id": 3}, {"dataset": "dapo", "id": "4"},
                   {"source_ids": ["5", 6]}, {"prompt": [{"role": "user", "content": "Chat question"}]}]
        ids, hashes = data.exclusions_from(payload)
        self.assertEqual(ids, {"3", "4", "5", "6"})
        self.assertEqual(hashes, {data.question_hash("External question"), data.question_hash("Chat question")})
        _, list_hashes = data.exclusions_from(["Question one", "Question two"])
        self.assertEqual(len(list_hashes), 2)
        _, nested_hashes = data.exclusions_from({"questions": ["Question one", "Question two"]})
        self.assertEqual(list_hashes, nested_hashes)

    def test_json_and_jsonl_exclusion_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a = root / "list.json"
            b = root / "list.jsonl"
            a.write_text('[{"question":"Q1"},{"problem":"Q2"}]', encoding="utf-8")
            b.write_text('{"question":"Q1"}\n\n{"problem":"Q2"}\n', encoding="utf-8")
            self.assertEqual(data.load_exclusions(a), data.load_exclusions(b))

    def test_frozen_outputs_have_verified_hashes_and_refuse_overwrite(self):
        splits, _ = data.select_splits(data.source_records(raw_rows()), set(), set(), 12, 4, 5, 42)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            manifest = data.write_outputs(root, splits, {"data_seed": 42})
            self.assertEqual(json.loads((root / "manifest.json").read_text()), manifest)
            before = {path.name: path.read_bytes() for path in root.iterdir()}
            for name in ("train", "val", "test"):
                self.assertEqual(manifest["splits"][name]["sha256"], data.digest(root / f"{name}.json"))
            with self.assertRaises(FileExistsError):
                data.write_outputs(root, splits, {"data_seed": 99})
            self.assertEqual(before, {path.name: path.read_bytes() for path in root.iterdir()})

    def test_pinned_sources_are_recorded(self):
        self.assertEqual(data.REVISION, "31dd309567e3da778038cc87d868b6097a3ccf68")
        self.assertEqual(data.SOURCE_SHA256, "40f672fe8b6dbeee953ae5acc75a38e99144e682450e07c625c87c734ec76285")
        self.assertEqual(models.MODELS["qwen4b"]["revision"], "cdbee75f17c01a7cc42f958dc650907174af0554")
        self.assertEqual(models.MODELS["qwen17b"]["revision"], "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")

    def test_cli_help_requires_no_site_packages(self):
        for script in ("prepare_data.py", "download_model.py"):
            result = subprocess.run([sys.executable, "-S", str(ROOT / "scripts" / script), "--help"],
                                    check=False, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("usage:", result.stdout)


class ModelDownloadTests(unittest.TestCase):
    def test_complete_download_records_request_and_asset_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "model"
            result = models.download_model("qwen4b", out, snapshot_fn=fake_model_snapshot)
            request = json.loads((out / ".download_request.json").read_text())
            self.assertEqual(request, {"alias": "qwen4b", **models.MODELS["qwen4b"]})
            self.assertEqual(result["status"], "download_complete")
            self.assertEqual({item["file"] for item in result["files"]},
                             {"config.json", "tokenizer_config.json", "model.safetensors"})
            for item in result["files"]:
                self.assertEqual(item["sha256"], models.digest(out / item["file"]))
            self.assertEqual(result, json.loads((out / "model_manifest.json").read_text()))

    def test_partial_download_resumes_only_with_matching_request(self):
        def interrupt(**kwargs):
            (Path(kwargs["local_dir"]) / "partial-marker").write_bytes(b"keep existing progress")
            raise OSError("simulated interrupted network")

        def resume(**kwargs):
            self.assertEqual((Path(kwargs["local_dir"]) / "partial-marker").read_bytes(), b"keep existing progress")
            fake_model_snapshot(**kwargs)

        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "model"
            with self.assertRaises(OSError):
                models.download_model("qwen4b", out, max_attempts=1, snapshot_fn=interrupt)
            self.assertTrue((out / ".download_request.json").is_file())
            self.assertFalse((out / "model_manifest.json").exists())
            with self.assertRaises(FileExistsError):
                models.download_model("qwen4b", out, snapshot_fn=resume)
            with self.assertRaises(ValueError):
                models.download_model("qwen17b", out, resume=True, snapshot_fn=resume)
            result = models.download_model("qwen4b", out, resume=True, snapshot_fn=resume)
            self.assertEqual(result["status"], "download_complete")

    def test_completed_resume_verifies_without_download_or_overwrite(self):
        def must_not_download(**kwargs):
            raise AssertionError("Completed resume must not call the network")

        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "model"
            models.download_model("qwen4b", out, snapshot_fn=fake_model_snapshot)
            before = {path.relative_to(out).as_posix(): path.read_bytes() for path in out.rglob("*") if path.is_file()}
            result = models.download_model("qwen4b", out, resume=True, snapshot_fn=must_not_download)
            self.assertEqual(result["status"], "verified_existing_download")
            after = {path.relative_to(out).as_posix(): path.read_bytes() for path in out.rglob("*") if path.is_file()}
            self.assertEqual(before, after)
            (out / "model.safetensors").write_bytes(b"altered asset")
            with self.assertRaises(ValueError):
                models.download_model("qwen4b", out, resume=True, snapshot_fn=must_not_download)
            self.assertEqual((out / "model.safetensors").read_bytes(), b"altered asset")

    def test_missing_or_changed_request_is_not_resumable(self):
        def must_not_download(**kwargs):
            raise AssertionError("Invalid request must not call the network")

        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "model"
            with self.assertRaises(ValueError):
                models.download_model("qwen4b", out, resume=True, snapshot_fn=must_not_download)
            out.mkdir()
            with self.assertRaises(ValueError):
                models.download_model("qwen4b", out, resume=True, snapshot_fn=must_not_download)
            request = {"alias": "qwen4b", **models.MODELS["qwen4b"], "revision": "wrong-revision"}
            (out / ".download_request.json").write_text(json.dumps(request), encoding="utf-8")
            with self.assertRaises(ValueError):
                models.download_model("qwen4b", out, resume=True, snapshot_fn=must_not_download)

    def test_bounded_retries_reuse_same_pinned_request(self):
        calls, waits = [], []

        def fail_twice(**kwargs):
            calls.append(dict(kwargs))
            if len(calls) < 3:
                raise OSError("simulated transient network failure")
            fake_model_snapshot(**kwargs)

        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "model"
            result = models.download_model("qwen4b", out, snapshot_fn=fail_twice, retry_wait=waits.append)
            self.assertEqual(result["status"], "download_complete")
            self.assertEqual(len(calls), 3)
            self.assertTrue(all(call == calls[0] for call in calls))
            self.assertEqual(waits, [2, 4])
            for count in (0, 6, True, 2.5):
                with self.assertRaises(ValueError):
                    models.download_model("qwen4b", Path(directory) / "other", max_attempts=count, snapshot_fn=fake_model_snapshot)

    def test_retries_stop_at_configured_bound(self):
        calls = []

        def always_fail(**kwargs):
            calls.append(kwargs)
            raise OSError("simulated persistent network failure")

        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "model"
            with self.assertRaises(OSError):
                models.download_model("qwen4b", out, max_attempts=2, snapshot_fn=always_fail, retry_wait=lambda _: None)
            self.assertEqual(len(calls), 2)
            self.assertFalse((out / "model_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
