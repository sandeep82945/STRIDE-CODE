"""CPU-only checks of held-out validation, immutable evidence and metric gates."""
from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training'))
import evaluate_async as driver
from evaluate import summarize


def dump(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')
    return path


def rows(prefix='eval', n=2):
    return [{'id': f'{prefix}:{i}', 'question': f'{prefix} problem number {i}', 'ground_truth': i + 1}
            for i in range(n)]


def make_adapter(root, model_digest='a' * 64):
    checkpoint = root / 'adapter_001'
    checkpoint.mkdir()
    dump(checkpoint / 'adapter_config.json', dict(peft_type='LORA', r=16, lora_alpha=32,
         lora_dropout=0, base_model_name_or_path='/an/old/machine/model'))
    (checkpoint / 'adapter_model.safetensors').write_bytes(b'fixture weights, not a model')
    manifest = {p.name: driver.file_sha256(p) for p in checkpoint.iterdir()}
    dump(root / 'adapter_001_manifest.json', manifest)
    dump(root / 'started.json', dict(model_identity={'content_sha256': model_digest},
                                   args={'model': 'qwen4b', 'finetune_mode': 'lora'}))
    return checkpoint


def fake_request(tok, batch, identifier, adapter, adapter_id, seed, cap, n):
    return dict(request_id=identifier, prompts=[dict(id=r['id'], prompt_token_ids=[11, 12], seed=seed+i)
                for i, r in enumerate(batch)], adapter_path=adapter, adapter_id=adapter_id,
                n=n, context_limit=cap)


class FakeSampler:
    def __init__(self, corrupt=None):
        self.requests = []
        self.closed = False
        self.corrupt = corrupt

    def generate(self, request, **kwargs):
        self.requests.append(copy.deepcopy(request))
        generated_rows = []
        for prompt in request['prompts']:
            samples = [dict(sample_index=i, token_ids=[13, 14, 15], text='A final explanation.',
                       finish_reason='stop', stop_reason=None, generated_tokens=3)
                       for i in range(request['n'])]
            generated_rows.append(dict(prompt, max_tokens=8190, samples=samples))
        result = dict(ok=True, rows=generated_rows, request_id=request['request_id'], context_limit=8192)
        if self.corrupt:
            self.corrupt(result)
        return result

    def close(self):
        self.closed = True


def fake_annotate(tok, batch, generated):
    groups = []
    for group in generated['rows']:
        records = []
        for sample in group['samples']:
            correct = sample['sample_index'] == 0
            score = dict(response=sample['text'], generated_tokens=len(sample['token_ids']),
                         step_eligible=True, strict_answer_correct=correct, relaxed_answer_correct=correct,
                         strict={'answer_correct': correct, 'reason': 'correct' if correct else 'wrong'},
                         relaxed={'answer_correct': correct, 'reason': 'correct' if correct else 'wrong'},
                         region={'valid': True, 'final': (0, len(sample['text']))})
            records.append(dict(sample, score=score))
        groups.append(dict(id=group['id'], records=records,
                           prompt_token_ids=group['prompt_token_ids'], features=None, advantages=object()))
    return groups


class PortableEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = dump(self.root / 'test.json', rows(n=3))
        self.train = dump(self.root / 'train.json', rows(prefix='train', n=2))
        self.model = self.root / 'model'
        self.model.mkdir()
        self.model_digest = {'content_sha256': 'a' * 64}

    def argv(self, *extra):
        return ['--model', 'qwen4b', '--model-path', str(self.model), '--data', str(self.data),
                '--train-data', str(self.train), '--out', str(self.root / 'eval-run'), '--wandb-mode', 'disabled', *extra]

    def components(self, sampler):
        return lambda: (lambda *a, **k: object(), lambda *a, **k: sampler,
                        fake_request, fake_annotate, summarize)

    def run_fixture(self, args=None, sampler=None, components=None):
        sampler = sampler or FakeSampler()
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': 'GPU-fixture'}), contextlib.redirect_stdout(io.StringIO()):
            result = driver.run_evaluation(args or driver.parse_args(self.argv()),
                       model_info_fn=lambda args: (self.model, self.model_digest),
                       components_fn=components or self.components(sampler))
        return result, sampler

    def test_help_uses_only_standard_library(self):
        result = subprocess.run([sys.executable, '-S', str(ROOT / 'training/evaluate_async.py'), '--help'],
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--train-data', result.stdout)
        self.assertIn('--sampler-eager', result.stdout)

    def test_frozen_evaluator_imports_without_source_repository_on_path(self):
        # Real launches execute under .cache/source/<hash>, not training/.
        # Do not let repository cwd/PYTHONPATH hide missing snapshot imports.
        spec = importlib.util.spec_from_file_location('handoff_launcher_for_eval_test', ROOT / 'run.py')
        launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launcher)
        snapshot = launcher.freeze_source(source=ROOT / 'training', cache=self.root / 'cache')
        result = subprocess.run(
            [sys.executable, '-E', '-S', str(snapshot / 'evaluate_async.py'), '--help'],
            cwd=self.root, text=True, capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--train-data', result.stdout)
        self.assertNotIn(str(ROOT / 'scripts'), result.stderr)

    def test_input_count_is_not_frozen_to_128(self):
        self.assertEqual(len(driver.load_data(self.data)), 3)
        self.assertEqual(len(driver.load_data(self.data, limit=1)), 1)
        with self.assertRaises(ValueError):
            driver.load_data(self.data, limit=4)

    def test_hash_pin_fails_before_runtime(self):
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            driver.load_data(self.data, expected_sha256='0' * 64)
        self.assertEqual(len(driver.load_data(self.data, expected_sha256=driver.file_sha256(self.data))), 3)

    def test_reference_and_record_validation(self):
        invalid = [None, {}, [], [dict(id='x', question='q')],
                   [dict(id=True, question='q', ground_truth=1)],
                   [dict(id='', question='q', ground_truth=1)],
                   [dict(id='x', question='', ground_truth=1)],
                   [dict(id='x', question='$$', ground_truth=1)],
                   [dict(id='x', question='q', ground_truth='1/2')],
                   [dict(id='x', question='q', ground_truth=1.0)],
                   [dict(id='x', question='q', ground_truth=True)]]
        for payload in invalid:
            with self.subTest(payload=payload):
                dump(self.data, payload)
                with self.assertRaises(ValueError):
                    driver.load_data(self.data)

    def test_duplicate_canonical_id_and_question(self):
        for payload in [
            [dict(id=1, question='a', ground_truth=1), dict(id='1', question='b', ground_truth=2)],
            [dict(id='a', question='Compute $x+1$', ground_truth=1),
             dict(id='b', question=' COMPUTE \\(x + 1\\) ', ground_truth=1)],
        ]:
            dump(self.data, payload)
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                driver.load_data(self.data)

    def test_overlap_in_unselected_tail_is_rejected(self):
        payload = rows(n=3)
        payload[2]['question'] = rows(prefix='train')[0]['question']
        dump(self.data, payload)
        with self.assertRaisesRegex(ValueError, 'overlap'):
            self.run_fixture(driver.parse_args(self.argv('--limit', '1')))
        self.assertFalse((self.root / 'eval-run').exists())

    def test_same_id_is_rejected_even_when_question_changed(self):
        eval_rows = rows()
        train_rows = rows(prefix='train')
        eval_rows[0]['id'] = train_rows[0]['id']
        with self.assertRaisesRegex(ValueError, 'overlap'):
            driver.check_overlap(eval_rows, train_rows)

    def test_argument_bounds(self):
        for pair in [('--rollouts', '0'), ('--rollouts', '65'), ('--batch-size', '0'),
                     ('--seed', '-1'), ('--limit', '0'), ('--generation-timeout', '0'), ('--data-sha256', 'bad')]:
            with self.subTest(pair=pair), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                driver.parse_args(self.argv(*pair))

    def test_checkpoint_model_mismatch_and_mutation(self):
        adapter = make_adapter(self.root)
        identity = driver.checkpoint_identity(adapter, model_identity=self.model_digest, model_alias='qwen4b')
        self.assertTrue(identity['source_model_identity_verified'])
        with self.assertRaisesRegex(ValueError, 'base-model content identity'):
            driver.checkpoint_identity(adapter, model_identity={'content_sha256': 'b' * 64}, model_alias='qwen4b')
        (adapter / 'adapter_model.safetensors').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'manifest'):
            driver.checkpoint_identity(adapter)

    def test_adapter_training_data_identity_is_enforced(self):
        adapter = make_adapter(self.root)
        started = self.root / 'started.json'
        metadata = json.loads(started.read_text())
        metadata['data_sha256'] = 'b' * 64
        dump(started, metadata)
        with self.assertRaisesRegex(ValueError, 'checkpoint training dataset hash'):
            self.run_fixture(driver.parse_args(self.argv('--adapter', str(adapter))))
        self.assertFalse((self.root / 'eval-run').exists())

    def test_checkpoint_requires_manifest_and_disallows_extra_files(self):
        adapter = make_adapter(self.root)
        (adapter / 'unexpected').write_text('extra')
        with self.assertRaisesRegex(ValueError, 'inventory'):
            driver.checkpoint_identity(adapter)
        (adapter / 'unexpected').unlink()
        (self.root / 'adapter_001_manifest.json').unlink()
        with self.assertRaisesRegex(ValueError, 'manifest'):
            driver.checkpoint_identity(adapter)

    def test_fixed_question_seeds_and_exact_metrics(self):
        args = driver.parse_args(self.argv('--seed', '7', '--rollouts', '8', '--batch-size', '2'))
        summary, sampler = self.run_fixture(args)
        self.assertEqual(summary['questions'], 3)
        self.assertEqual(summary['responses'], 24)
        self.assertEqual(summary['relaxed']['pass1'], 1 / 8)
        self.assertEqual(summary['relaxed']['pass8'], 1)
        self.assertEqual([p['seed'] for r in sampler.requests for p in r['prompts']], [7, 8, 9])
        self.assertTrue(all(r['adapter_path'] is None and r['adapter_id'] == 0 for r in sampler.requests))
        self.assertTrue(sampler.closed)
        out = self.root / 'eval-run'
        done = json.loads((out / 'done.json').read_text())
        self.assertEqual(done['summary_sha256'], driver.file_sha256(out / 'summary.json'))
        self.assertTrue(done['complete'])
        self.assertNotIn('advantages', json.loads((out / 'groups_000000.json').read_text())[0])
        self.assertFalse((out / 'failed.json').exists())

    def test_pass8_is_null_when_k_is_four(self):
        summary, _ = self.run_fixture(driver.parse_args(self.argv('--rollouts', '4')))
        self.assertIsNone(summary['strict']['pass8'])
        self.assertIsNone(summary['relaxed']['pass8'])
        self.assertEqual(summary['relaxed']['pass1'], .25)

    def test_existing_output_is_never_overwritten(self):
        target = self.root / 'eval-run'
        target.mkdir()
        (target / 'sentinel').write_text('keep')
        with self.assertRaises(FileExistsError):
            self.run_fixture()
        self.assertEqual((target / 'sentinel').read_text(), 'keep')

    def test_changed_rollout_budget_fails_without_done(self):
        sampler = FakeSampler(corrupt=lambda result: result['rows'][0]['samples'].pop())
        with self.assertRaisesRegex(ValueError, 'rollout count'):
            self.run_fixture(sampler=sampler)
        out = self.root / 'eval-run'
        self.assertTrue((out / 'sampled_000000.json').exists())
        self.assertTrue((out / 'failed.json').exists())
        self.assertFalse((out / 'done.json').exists())
        self.assertTrue(sampler.closed)

    def test_dependency_failure_is_not_reported_as_completed_evaluation(self):
        def missing():
            raise ImportError('fixture missing vllm')
        with self.assertRaises(ImportError):
            self.run_fixture(components=missing)
        out = self.root / 'eval-run'
        self.assertTrue((out / 'eval_spec.json').exists())
        self.assertTrue((out / 'failed.json').exists())
        self.assertFalse((out / 'done.json').exists())


if __name__ == '__main__':
    unittest.main()
