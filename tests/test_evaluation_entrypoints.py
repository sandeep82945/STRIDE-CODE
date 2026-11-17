"""CPU entrypoint integration: real CLI, hashes, request, parser and metrics.

A deterministic sampler and tokenizer replace model inference. No GPU report,
model weight loading, or throughput conclusion is made by these tests.
"""
from __future__ import annotations

from contextlib import redirect_stdout, redirect_stderr
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from unittest.mock import MagicMock
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'training'))
import evaluate_async as driver
import runtime_config
import sampler_worker
import trainer
from tests.test_parser_grading_extensive import ByteLevelFixture, CANONICAL


class TemplateTokenizer(ByteLevelFixture):
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        assert not tokenize and add_generation_prompt
        return 'User: ' + messages[0]['content'] + '\nAssistant: '

    def __call__(self, text, add_special_tokens=False):
        return {'input_ids': self.encode(text, add_special_tokens=add_special_tokens)}


class DeterministicSampler:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.requests = []
        self.closed = False

    def generate(self, request, timeout):
        self.requests.append(request)
        rows = []
        for prompt in request['prompts']:
            samples = []
            for index in range(request['n']):
                text = CANONICAL if index % 2 == 0 else CANONICAL.replace(r'\boxed{4}', r'\boxed{5}')
                ids = self.tokenizer.encode(text)
                samples.append(dict(sample_index=index, token_ids=ids, text=text,
                                    generated_tokens=len(ids), finish_reason='stop', stop_reason=None))
            rows.append(dict(id=prompt['id'], prompt_token_ids=prompt['prompt_token_ids'],
                             max_tokens=8192-len(prompt['prompt_token_ids']), samples=samples))
        return dict(ok=True, request_id=request['request_id'], context_limit=request['context_limit'], rows=rows)

    def close(self):
        self.closed = True


class EvaluationEntrypointTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.model = self.root / 'model'
        self.model.mkdir()
        self.write(self.model / 'config.json', {'model_type': 'qwen3'})
        # Only content identity is read here; these bytes are never model-loaded.
        (self.model / 'model.safetensors').write_bytes(b'explicit synthetic identity fixture')
        self.write(self.model / 'tokenizer_config.json', {'fixture': True})
        self.data = self.root / 'eval.json'
        self.train = self.root / 'train.json'
        self.write(self.data, [dict(id='eval:' + str(i), question='Evaluate ' + str(i), ground_truth=4) for i in range(3)])
        self.write(self.train, [dict(id='train:0', question='Disjoint training question', ground_truth=7)])

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value), encoding='utf-8')

    def argv(self, **kwargs):
        values = dict(model='qwen4b', model_path=str(self.model), data=str(self.data),
                      train_data=str(self.train), out=str(self.root / 'evaluation'), rollouts=2,
                      batch_size=2, seed=11, wandb_mode='disabled')
        values.update(kwargs)
        return [part for key, value in values.items() for part in ('--' + key.replace('_', '-'), str(value))]

    def test_default_model_info_uses_real_content_hashes_and_detects_replacement(self):
        args = driver.parse_args(self.argv())
        path, identity = driver._default_model_info(args)
        self.assertEqual(path, self.model.resolve())
        self.assertEqual(identity, runtime_config.model_identity(self.model))
        self.assertEqual(identity['files']['model.safetensors'], hashlib.sha256(b'explicit synthetic identity fixture').hexdigest())
        self.assertIsNone(identity['preparation_manifest_sha256'])
        (self.model / 'model.safetensors').write_bytes(b'changed synthetic fixture')
        self.assertNotEqual(driver._default_model_info(args)[1]['content_sha256'], identity['content_sha256'])

    def test_default_model_info_checks_downloaded_alias_and_repo(self):
        manifest = self.model / 'model_manifest.json'
        args = driver.parse_args(self.argv())
        self.write(manifest, dict(alias='qwen4b', repo_id=driver.MODEL_REPOS['qwen4b']))
        _, identity = driver._default_model_info(args)
        self.assertEqual(identity['preparation_manifest_sha256'], driver.file_sha256(manifest))
        self.write(manifest, dict(alias='qwen17b', repo_id=driver.MODEL_REPOS['qwen17b']))
        with self.assertRaisesRegex(ValueError, 'label disagrees|alias differs'):
            driver._default_model_info(args)
        self.write(manifest, dict(alias='qwen4b', repo_id=driver.MODEL_REPOS['qwen17b']))
        with self.assertRaisesRegex(ValueError, 'repository differs'):
            driver._default_model_info(args)
        # Explicit None and absent fields do not manufacture a repository pin.
        self.write(manifest, dict(alias=None, repo_id=None))
        self.assertEqual(driver._default_model_info(args)[0], self.model)

    def test_runtime_components_resolve_the_actual_shared_implementations(self):
        from transformers import AutoTokenizer
        import evaluate
        loader, launcher, request, annotate, summary = driver._runtime_components()
        self.assertEqual(loader, AutoTokenizer.from_pretrained)
        self.assertEqual(launcher, sampler_worker.SamplerClient.launch)
        self.assertIs(request, trainer.request)
        self.assertIs(annotate, trainer.annotate_groups)
        self.assertIs(summary, evaluate.summarize)

    def test_main_runs_real_validation_request_scoring_and_metrics(self):
        from transformers import AutoTokenizer
        tokenizer = TemplateTokenizer()
        sampler = DeterministicSampler(tokenizer)
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': 'CPU-test-allocation-label'}), \
             patch.object(AutoTokenizer, 'from_pretrained', return_value=tokenizer) as loader, \
             patch.object(sampler_worker.SamplerClient, 'launch', return_value=sampler) as launch, \
             redirect_stdout(io.StringIO()):
            result = driver.main(self.argv())
        loader.assert_called_once_with(str(self.model), local_files_only=True, use_fast=True)
        self.assertEqual(launch.call_args.kwargs['sampler_device'], 'CPU-test-allocation-label')
        self.assertEqual(launch.call_args.kwargs['max_num_seqs'], 16)
        self.assertEqual(launch.call_args.kwargs['engine_seed'], 11)
        self.assertEqual(launch.call_args.kwargs['finetune_mode'], 'full')
        self.assertEqual(launch.call_args.kwargs['tokenizer_path'], str(self.model))
        self.assertEqual(launch.call_args.kwargs['model_version'], 0)
        self.assertFalse(launch.call_args.kwargs['eager'])
        self.assertEqual(result['questions'], 3)
        self.assertEqual(result['responses'], 6)
        self.assertEqual(result['relaxed']['pass1'], .5)
        self.assertIsNone(result['relaxed']['pass8'])
        self.assertEqual(result['syntax']['step_eligible_responses'], 6)
        self.assertEqual([p['seed'] for request in sampler.requests for p in request['prompts']], [11, 12, 13])
        self.assertTrue(sampler.closed)
        output = self.root / 'evaluation'
        spec = json.loads((output / 'eval_spec.json').read_text())
        self.assertEqual(spec['structural_reward'], 0)
        self.assertEqual(spec['finetune_mode'], 'base')
        self.assertEqual(spec['model_identity'], runtime_config.model_identity(self.model))
        self.assertEqual(spec['data']['sha256'], driver.file_sha256(self.data))
        self.assertEqual(spec['train_data']['sha256'], driver.file_sha256(self.train))
        done = json.loads((output / 'done.json').read_text())
        self.assertTrue(done['complete'])
        self.assertEqual(done['summary_sha256'], driver.file_sha256(output / 'summary.json'))
        self.assertFalse((output / 'failed.json').exists())

    def test_large_evaluation_seed_keeps_sampling_bits_and_bounds_engine_seed(self):
        from transformers import AutoTokenizer
        tokenizer = TemplateTokenizer()
        sampler = DeterministicSampler(tokenizer)
        seed = 2**40 + 43
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES='CPU-test-allocation-label'), \
             patch.object(AutoTokenizer, 'from_pretrained', return_value=tokenizer), \
             patch.object(sampler_worker.SamplerClient, 'launch', return_value=sampler) as launch, \
             redirect_stdout(io.StringIO()):
            driver.main(self.argv(seed=seed))
        self.assertEqual(launch.call_args.kwargs['engine_seed'], 43)
        self.assertEqual([p['seed'] for req in sampler.requests for p in req['prompts']],
                         [seed, seed + 1, seed + 2])
        spec = json.loads((self.root / 'evaluation' / 'eval_spec.json').read_text())
        self.assertEqual(spec['evaluation_seed'], seed)
        self.assertEqual(spec['sampler_engine_seed'], 43)
        self.assertEqual(spec['python_hash_seed'], 43)

    def test_main_rejects_cli_or_bad_model_before_loading_runtime(self):
        with patch.object(driver, '_runtime_components') as components:
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                driver.main(self.argv(rollouts=0))
            components.assert_not_called()
            (self.model / 'model.safetensors').unlink()
            with self.assertRaisesRegex(ValueError, 'No local model weights'):
                driver.main(self.argv())
            components.assert_not_called()
        self.assertFalse((self.root / 'evaluation').exists())

    @staticmethod
    def fake_wandb():
        run = MagicMock()
        run.id, run.project, run.entity, run.url = 'eval-id', 'test-project', 'test-team', 'https://example.invalid/eval-id'
        return types.SimpleNamespace(init=MagicMock(return_value=run), Settings=MagicMock()), run

    def test_wandb_evaluation_is_distinct_metrics_only_and_records_checkpoint_axis(self):
        from transformers import AutoTokenizer
        from tests.test_portable_eval import make_adapter
        tokenizer, sdk_and_run = TemplateTokenizer(), self.fake_wandb()
        sdk, tracking_run = sdk_and_run
        sampler = DeterministicSampler(tokenizer)
        checkpoint = make_adapter(self.root, runtime_config.model_identity(self.model)['content_sha256'])
        metadata = json.loads((self.root / 'started.json').read_text())
        metadata['training_contract'] = dict(arm='local_positive', finetune_mode='lora')
        self.write(self.root / 'started.json', metadata)
        events = []
        sdk.init.side_effect = lambda **kwargs: (events.append('wandb'), tracking_run)[1]
        def launch(*args, **kwargs):
            events.append('sampler')
            return sampler
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': 'CPU-test-allocation-label',
                                     'WANDB_RUN_ID': 'active-training-id', 'WANDB_RESUME': 'must',
                                     'WANDB_API_KEY': 'test-secret-not-for-config'}), \
             patch.dict(sys.modules, {'wandb': sdk}), \
             patch.object(AutoTokenizer, 'from_pretrained', return_value=tokenizer), \
             patch.object(sampler_worker.SamplerClient, 'launch', side_effect=launch), \
             redirect_stdout(io.StringIO()):
            summary = driver.main(self.argv(wandb_mode='online', adapter=checkpoint,
                training_step=64, epoch=2, wandb_name='qwen4b-local-positive-epoch2',
                wandb_project='test-project', wandb_group='comparison-42', wandb_entity='test-team'))
        self.assertEqual(events, ['wandb', 'sampler'])
        options = sdk.init.call_args.kwargs
        self.assertNotEqual(options['id'], 'active-training-id')
        self.assertEqual(options['resume'], 'never')
        self.assertTrue(options['reinit'])
        self.assertEqual(options['job_type'], 'evaluation')
        self.assertEqual(options['group'], 'comparison-42')
        self.assertEqual(options['name'], 'qwen4b-local-positive-epoch2')
        self.assertEqual(options['config']['method'], 'local_positive')
        self.assertEqual(options['config']['finetune_mode'], 'lora')
        self.assertEqual(options['config']['checkpoint_manifest_sha256'],
                         driver.file_sha256(self.root / 'adapter_001_manifest.json'))
        self.assertNotIn('test-secret', json.dumps(options['config']))
        self.assertNotIn('question', options['config'])
        self.assertFalse(options['save_code'])
        sdk.Settings.assert_called_once_with(console='off', disable_code=True, disable_git=True)
        tracking_run.define_metric.assert_any_call('eval/*', step_metric='training_step')
        expected = dict(driver.scalar_metrics(summary), training_step=64, epoch=2)
        tracking_run.log.assert_called_once_with(expected)
        self.assertTrue(all(isinstance(value, (int, float)) for value in expected.values()))
        tracking_run.finish.assert_called_once_with()
        local = json.loads((self.root / 'evaluation' / 'wandb.json').read_text())
        self.assertEqual(local['id'], tracking_run.id)
        self.assertEqual(local['url'], tracking_run.url)
        self.assertEqual(local['epoch'], 2)

    def test_wandb_authentication_fails_before_inference_without_leaking_key(self):
        sdk, _ = self.fake_wandb()
        sdk.init.side_effect = RuntimeError('Rejected credential test-secret-123')
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES='CPU-test-allocation-label'), \
             patch.dict(sys.modules, {'wandb': sdk}), \
             patch.object(driver, '_runtime_components') as runtime, \
             self.assertRaisesRegex(RuntimeError, 'W&B evaluation initialization failed') as caught:
            driver.main(self.argv(wandb_mode='online'))
        runtime.assert_not_called()
        self.assertNotIn('test-secret-123', str(caught.exception))
        failure = json.loads((self.root / 'evaluation' / 'failed.json').read_text())
        self.assertNotIn('test-secret-123', failure['error'])
        self.assertFalse((self.root / 'evaluation' / 'done.json').exists())

    def test_wandb_log_failure_keeps_local_results_and_is_explicit(self):
        from transformers import AutoTokenizer
        sdk, tracking_run = self.fake_wandb()
        tracking_run.log.side_effect = RuntimeError('Rejected credential test-secret-123')
        tokenizer = TemplateTokenizer()
        sampler = DeterministicSampler(tokenizer)
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES='CPU-test-allocation-label'), \
             patch.dict(sys.modules, {'wandb': sdk}), \
             patch.object(AutoTokenizer, 'from_pretrained', return_value=tokenizer), \
             patch.object(sampler_worker.SamplerClient, 'launch', return_value=sampler), \
             self.assertRaisesRegex(RuntimeError, 'W&B evaluation logging failed'):
            driver.main(self.argv(wandb_mode='offline'))
        output = self.root / 'evaluation'
        self.assertTrue((output / 'summary.json').exists())
        self.assertTrue(sampler.closed)
        self.assertFalse((output / 'done.json').exists())
        self.assertNotIn('test-secret-123', (output / 'failed.json').read_text())
        tracking_run.finish.assert_called_once_with(exit_code=1)

    def test_wandb_environment_defaults_and_checkpoint_axis_arguments(self):
        with patch.dict(os.environ, {'WANDB_PROJECT': 'env-project', 'WANDB_ENTITY': 'env-team',
                                     'WANDB_RUN_GROUP': 'env-group', 'WANDB_MODE': 'offline'}):
            # Override argv helper's explicitly disabled mode only in this test.
            args = driver.parse_args(self.argv()[:-2])
        self.assertEqual((args.wandb_project, args.wandb_entity, args.wandb_group, args.wandb_mode),
                         ('env-project', 'env-team', 'env-group', 'offline'))
        for extra in ({'training_step': -1}, {'epoch': 0}, {'wandb_project': '', 'wandb_mode': 'online'}):
            with self.subTest(extra=extra), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                driver.parse_args(self.argv(**extra))
        args = driver.parse_args(self.argv(training_step=0, epoch=1))
        self.assertEqual((args.training_step, args.epoch), (0, 1))

    def test_deprecated_fixed_pilot_entrypoint_rejects_without_training(self):
        with patch.object(trainer, 'load_actor') as loader, self.assertRaisesRegex(SystemExit, 'trainer_resumable.py train'):
            trainer.main()
        loader.assert_not_called()


if __name__ == '__main__':
    unittest.main()
