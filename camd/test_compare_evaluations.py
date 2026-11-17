"""Synthetic evidence tests; no GPUs, model inference, credentials or network."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('camd_compare_test_subject', HERE/'compare_evaluations.py')
comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)


def write(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True) + '\n')


def fixture(root):
    gate, aggregation = comparison.dependencies()
    training = root/'train.json'
    validation = root/'val.json'
    write(training, [dict(id=f'train-{i}', question=f'Train problem {i}', ground_truth=1) for i in range(100)])
    write(validation, [dict(id=f'val-{i}', question=f'Validation problem {i}', ground_truth=1) for i in range(100)])
    code = {path.relative_to(gate.TRAINING).as_posix(): comparison.sha(path)
            for path in gate.TRAINING.rglob('*.py')}
    model_files = {'model.safetensors': hashlib.sha256(b'synthetic base weights').hexdigest()}
    model = dict(files=model_files, content_sha256=hashlib.sha256(
        json.dumps(model_files, sort_keys=True, separators=(',', ':')).encode()).hexdigest())
    train_hash, val_hash = comparison.sha(training), comparison.sha(validation)
    manifest = dict(version=1, baseline='base', arms={arm: {} for arm in comparison.ARMS})
    for arm in ('base', *comparison.ARMS):
        if arm != 'base':
            run = root/f'train-{arm}'
            run.mkdir()
            contract = dict(version=6, mode='train', model='qwen4b', finetune_mode='lora', arm=arm,
                seed=42, epochs=4, K=8, batch_size=64, training_questions=100, world_size=4,
                updates_per_epoch=2, batch_schedule='epoch_shuffle_keep_partial',
                collective_padding='zero_loss_no_generated_samples', model_identity=model,
                data_sha256=train_hash, code_sha256=code)
            write(run/'started.json', dict(training_contract=contract, code_sha256=code,
                                          data_sha256=train_hash, model_identity=model))
        for epoch in (None,) if arm == 'base' else (4,):
            output = root/('base' if epoch is None else f'{arm}-e{epoch}')
            output.mkdir()
            checkpoint = None
            if epoch is not None:
                manifest['arms'][arm][str(epoch)] = output.name
                adapter = run/f'adapter_{epoch*2:03d}'
                adapter.mkdir()
                (adapter/'adapter_model.safetensors').write_bytes(f'synthetic {arm} {epoch}'.encode())
                write(adapter/'adapter_config.json', dict(peft_type='LORA', r=16, lora_alpha=32, lora_dropout=0))
                files = {file.name: comparison.sha(file) for file in adapter.iterdir()}
                sibling = adapter.with_name(adapter.name+'_manifest.json')
                write(sibling, files)
                checkpoint = dict(path=str(adapter), manifest_path=str(sibling),
                    manifest_sha256=comparison.sha(sibling), files=files, finetune_mode='lora',
                    source_metadata_sha256=comparison.sha(run/'started.json'),
                    source_model_content_sha256=model['content_sha256'], source_model_identity_verified=True,
                    source_training_data_sha256=train_hash)
            spec = dict(version=2, evaluation_only=True, model='qwen4b', model_identity=model,
                checkpoint=checkpoint, finetune_mode='base' if epoch is None else 'lora',
                data=dict(path=str(validation), sha256=val_hash, total_questions=100,
                          evaluated_questions=100, prefix_limit=None),
                train_data=dict(path=str(training), sha256=train_hash),
                overlap=dict(checked=True, id_overlap=0, question_overlap=0),
                evaluation_seed=42, sampler_engine_seed=42, python_hash_seed=42,
                rollouts_per_question=8, prompt_batch=16, context_limit=8192, temperature=1.,
                sampler_max_num_seqs=32, sampler_eager=False, generation_timeout_seconds=3600,
                grader='existing strict and relaxed signed-integer answer graders',
                diversity_scope='lexical final-explanation diagnostics only; no verified reasoning diversity',
                structural_reward=0, training_step=epoch*2 if epoch else None, epoch=epoch, code_sha256=code)
            groups = []
            for index in range(100):
                gain = 0 if epoch is None else epoch if arm == 'local_positive' else max(0, epoch-1)
                correct = min(8, index % 9 + gain)
                records = []
                for sample in range(8):
                    truncated = (index+sample) % 29 == 0
                    relaxed = sample < correct and not truncated
                    strict = sample < max(0, correct-(index % 3 == 0)) and not truncated
                    text = 'Step 1: Add.\nStep 2: Count.\nFinal Answer: \\boxed{1}'
                    token_ids = [1, 2, 3, 4][:1+(index+sample) % 4]
                    score = dict(response=text, generated_tokens=len(token_ids), truncated=truncated,
                        strict_answer_correct=strict, relaxed_answer_correct=relaxed,
                        strict=dict(answer_correct=strict, reason='synthetic'),
                        relaxed=dict(answer_correct=relaxed, reason='synthetic'),
                        step_eligible=not truncated, parse_valid=not truncated, mapping_verified=True,
                        region=dict(valid=True, final=[0, len(text)]))
                    records.append(dict(sample_index=sample, token_ids=token_ids, text=text,
                                        finish_reason='length' if truncated else 'stop', score=score))
                groups.append(dict(id=f'val-{index}', prompt_token_ids=[5, index+10], records=records))
            for start in range(0, 100, 16):
                write(output/f'groups_{start:06d}.json', groups[start:start+16])
            write(output/'eval_spec.json', spec)
            write(output/'summary.json', aggregation.summarize(groups))
            write(output/'done.json', dict(complete=True, evaluation_only=True, questions=100, responses=800,
                checkpoint=checkpoint, eval_spec_sha256=comparison.sha(output/'eval_spec.json'),
                summary_sha256=comparison.sha(output/'summary.json')))
    write(root/'manifest.json', manifest)
    return root/'manifest.json'


def rehash(output, recompute_summary=False):
    if recompute_summary:
        _, aggregation = comparison.dependencies()
        groups = [group for file in sorted(output.glob('groups_*.json')) for group in comparison.read(file)]
        write(output/'summary.json', aggregation.summarize(groups))
    done = comparison.read(output/'done.json')
    done['eval_spec_sha256'] = comparison.sha(output/'eval_spec.json')
    done['summary_sha256'] = comparison.sha(output/'summary.json')
    write(output/'done.json', done)


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='stride-comparison-fixture-')
        self.root = Path(self.temporary.name)
        self.manifest = fixture(self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def test_isolated_stdlib_cli_all_outputs_and_intervals(self):
        output = self.root/'comparison'
        command = [sys.executable, '-I', '-S', str(HERE/'compare_evaluations.py'),
                   '--manifest', str(self.manifest), '--output', str(output)]
        run = subprocess.run(command, capture_output=True, text=True, timeout=90)
        self.assertEqual(run.returncode, 0, run.stderr)
        report = comparison.read(output/'comparison.json')
        self.assertEqual((report['questions'], report['rollouts_per_question']), (100, 8))
        self.assertEqual(len(report['evaluations']), 5)
        self.assertEqual(len(report['contrasts']), 8)
        self.assertEqual(report['bootstrap']['replicates'], 2000)
        self.assertFalse(report['checkpoint_selected'])
        self.assertEqual(report['scope'], 'validation_pilot')
        for item in report['contrasts']:
            self.assertEqual(set(item['metrics']), set(comparison.METRICS))
            if item['arm'] == item['reference']:
                self.assertTrue(all(value['delta'] == 0 and value['ci95'] == [0., 0.]
                                    for value in item['metrics'].values()))
        self.assertTrue(any(item['metrics']['relaxed/pass1']['ci95'][0] > 0
                            for item in report['contrasts'] if item['arm'] == 'local_positive'
                            and item['reference'] == 'base'))
        done = comparison.read(output/'done.json')
        self.assertEqual(done['comparison_sha256'], comparison.sha(output/'comparison.json'))
        self.assertEqual(done['markdown_sha256'], comparison.sha(output/'comparison.md'))
        repeated = subprocess.run(command, capture_output=True, text=True, timeout=10)
        self.assertNotEqual(repeated.returncode, 0)
        self.assertIn('fresh comparison', repeated.stderr)

    def test_missing_output_reused_slot_and_wrong_arm_rejected(self):
        original = comparison.read(self.manifest)
        for kind in ('missing', 'reused', 'wrong_arm', 'earlier_epoch'):
            value = copy.deepcopy(original)
            if kind == 'missing':
                value['arms']['grpo']['4'] = 'absent'
            elif kind == 'reused':
                value['arms']['grpo']['4'] = value['arms']['local_positive']['4']
            elif kind == 'wrong_arm':
                value['arms']['grpo']['4'], value['arms']['local_positive']['4'] = (
                    value['arms']['local_positive']['4'], value['arms']['grpo']['4'])
            else:
                value['arms']['grpo']['3'] = value['arms']['grpo'].pop('4')
            write(self.manifest, value)
            with self.subTest(kind=kind), self.assertRaises((ValueError, FileNotFoundError)):
                comparison.compare(self.manifest)

    def test_mixed_evaluation_config_rejected(self):
        output = self.root/'local_positive-e4'
        spec = comparison.read(output/'eval_spec.json')
        spec['sampler_eager'] = True
        write(output/'eval_spec.json', spec)
        rehash(output)
        with self.assertRaisesRegex(ValueError, 'Mixed data, model, source or evaluation configuration'):
            comparison.compare(self.manifest)

    def test_mixed_model_source_and_training_configuration_rejected(self):
        output = self.root/'local_positive-e4'
        source_path = self.root/'train-local_positive/started.json'
        original_source = comparison.read(source_path)
        original_spec = comparison.read(output/'eval_spec.json')
        for kind in ('model', 'source', 'training'):
            spec = copy.deepcopy(original_spec)
            source = copy.deepcopy(original_source)
            if kind == 'model':
                spec['model_identity']['content_sha256'] = 'b'*64
            elif kind == 'source':
                spec['code_sha256']['trainer.py'] = 'b'*64
            else:
                source['training_contract']['K'] = 4
                write(source_path, source)
                spec['checkpoint']['source_metadata_sha256'] = comparison.sha(source_path)
            write(output/'eval_spec.json', spec)
            done = comparison.read(output/'done.json')
            done['checkpoint'] = spec['checkpoint']
            write(output/'done.json', done)
            rehash(output)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                comparison.compare(self.manifest)
            write(source_path, original_source)

    def test_raw_summary_mismatch_rejected_even_when_done_hashes_match(self):
        output = self.root/'base'
        summary = comparison.read(output/'summary.json')
        summary['strict']['pass1'] = .999
        write(output/'summary.json', summary)
        rehash(output)
        with self.assertRaisesRegex(ValueError, 'summary differs'):
            comparison.compare(self.manifest)

    def test_changed_prompt_tokens_rejected_even_with_consistent_summary(self):
        output = self.root/'local_positive-e4'
        groups = comparison.read(output/'groups_000000.json')
        groups[0]['prompt_token_ids'][0] = 999
        write(output/'groups_000000.json', groups)
        rehash(output, recompute_summary=True)
        with self.assertRaisesRegex(ValueError, 'rendered prompt token IDs differ'):
            comparison.compare(self.manifest)

    def test_question_ids_and_missing_groups_rejected(self):
        output = self.root/'base'
        file = output/'groups_000000.json'
        original = comparison.read(file)
        damaged = copy.deepcopy(original)
        damaged[0]['id'] = 'foreign-question'
        write(file, damaged)
        with self.assertRaisesRegex(ValueError, 'questions exactly'):
            comparison.compare(self.manifest)
        write(file, original)
        file.unlink()
        with self.assertRaises(ValueError):
            comparison.compare(self.manifest)

    def test_paired_question_bootstrap_retains_identical_pairs(self):
        rows = [dict(id=i, **{metric: (i % 9)/8 for metric in comparison.METRICS}) for i in range(100)]
        treatment = dict(arm='local_positive', epoch=4, per_question=rows)
        reference = dict(arm='grpo', epoch=4, per_question=copy.deepcopy(rows))
        weights = comparison.bootstrap_weights(replicates=100)
        report = comparison.paired_contrast(treatment, reference, weights)
        self.assertEqual(weights, comparison.bootstrap_weights(replicates=100))
        for metric in report['metrics'].values():
            self.assertEqual(metric['delta'], 0)
            self.assertEqual(metric['ci95'], [0, 0])
            self.assertFalse(metric['ci_excludes_zero'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
