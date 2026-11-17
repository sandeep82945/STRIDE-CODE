"""Portable configuration/scheduling/provenance checks; no GPU needed."""
import argparse
from contextlib import redirect_stderr
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training'))
import runtime_config as config
import resumable_checkpoint as checkpoints
from trainer_distributed import allocation, epoch_batch, rank_batch


def arguments(mode='train', *extra):
    parser = argparse.ArgumentParser()
    config.add_training_arguments(parser, resumable=True)
    argv = [mode, '--model-path', '/prepared/model', '--data', '/prepared/train.json',
            '--data-sha256', 'a' * 64, '--out', '/fresh/attempt']
    return parser.parse_args(argv + list(extra))


class TrainingConfigurationTests(unittest.TestCase):
    def test_latest_epoch_state_retention_requires_keep_all_lora_and_extension_requires_resume(self):
        args = arguments('train', '--resume-state-retention', 'latest-epoch')
        with self.assertRaisesRegex(ValueError, 'requires LoRA'):
            config.validate_options(args, resumable=True)
        args.finetune_mode = 'lora'
        with self.assertRaisesRegex(ValueError, 'keep-all-checkpoints'):
            config.validate_options(args, resumable=True)
        args.keep_all_checkpoints = True
        config.validate_options(args, resumable=True)
        args.allow_epoch_extension = True
        with self.assertRaisesRegex(ValueError, 'requires a verified'):
            config.validate_options(args, resumable=True)
        args.resume_from = '/verified/state_128'
        config.validate_options(args, resumable=True)

    def test_epoch_extension_changes_only_horizon_and_rejects_equal_or_smaller_targets(self):
        from trainer_resumable import initial_training_state
        args = arguments('train', '--finetune-mode', 'lora')
        config.validate_options(args, resumable=True)
        with patch('trainer_resumable.version', return_value='test'):
            original = initial_training_state(args, 4, 2048, {'content_sha256': 'fixture'})[0]
            source = dict(completed_step=128, contract=original)
            args.resume_from, args.allow_epoch_extension, args.epochs = '/old/state_128', True, 6
            with patch('trainer_resumable.checkpoints.inspect_checkpoint', return_value=source) as inspect:
                contract, resumed, start, steps = initial_training_state(args, 4, 2048, {'content_sha256': 'fixture'})
                inspect.assert_called_once_with('/old/state_128', None, 4)
                self.assertEqual((start, steps, contract['epochs']), (128, 192, 6))
                self.assertEqual(resumed['contract']['epochs'], 4)
                for epochs in (3, 4):
                    args.epochs = epochs
                    with self.assertRaisesRegex(ValueError, 'strictly larger'):
                        initial_training_state(args, 4, 2048, {'content_sha256': 'fixture'})
                args.epochs = 6
            for field, value in [('code_sha256', {'trainer.py': 'changed'}), ('seed', 43),
                                 ('K', 4), ('optimizer', {'lr': .2}), ('data_sha256', 'b' * 64),
                                 ('world_size', 8), ('learner_microbatch_size', 2), ('versions', {'torch': 'changed'})]:
                altered = copy.deepcopy(source)
                altered['contract'][field] = value
                with self.subTest(field=field), patch('trainer_resumable.checkpoints.inspect_checkpoint', return_value=altered):
                    with self.assertRaisesRegex(ValueError, 'only epochs'):
                        initial_training_state(args, 4, 2048, {'content_sha256': 'fixture'})

    def test_keep_all_checkpoints_is_opt_in_and_preserves_scientific_contract(self):
        default = arguments('train', '--epochs', '10', '--finetune-mode', 'lora')
        retained = arguments('train', '--epochs', '10', '--finetune-mode', 'lora', '--keep-all-checkpoints')
        self.assertFalse(default.keep_all_checkpoints)
        self.assertTrue(retained.keep_all_checkpoints)
        for args in (default, retained):
            config.validate_options(args, resumable=True)
            self.assertEqual(config.schedule_steps(args, 2048), 320)
        identity = {'content_sha256': 'fixture'}
        self.assertEqual(config.scientific_contract(default, world_size=4, training_questions=2048, identity=identity),
                         config.scientific_contract(retained, world_size=4, training_questions=2048, identity=identity))

    def test_four_arm_configuration_and_new_source_identity(self):
        from trainer_resumable import initial_training_state
        self.assertEqual(set(config.ARMS), {'local_positive', 'single_step', 'token_uniform', 'grpo'})
        for arm in config.ARMS:
            args = arguments('train', '--arm', arm)
            config.validate_options(args, resumable=True)
            with patch('trainer_resumable.version', return_value='test'):
                contract, resume, start, steps = initial_training_state(args, 4, 2048, {'content_sha256': 'fixture'})
            self.assertEqual(contract['arm'], arm)
            self.assertEqual(contract['initialization'], 'base_model_full')
            self.assertIsNone(resume)
            self.assertEqual((start, steps), (0, 128))
            self.assertFalse(hasattr(args, 'adapter'))
            for name in ('experiment_credit.py', 'feature_pooling.py'):
                self.assertEqual(contract['code_sha256'][name], config.digest(ROOT/'training'/name))

    def test_help_without_torch_or_cluster_environment(self):
        for name in ('trainer_distributed.py', 'trainer_resumable.py'):
            run = subprocess.run([sys.executable, str(ROOT / 'training' / name), '--help'],
                                 capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            for flag in ('--model-path', '--data-sha256', '--rollouts', '--alpha', '--learning-rate', '--finetune-mode', '--warmup-steps', '--kl-coefficient'):
                self.assertIn(flag, run.stdout)
            for flag in ('--adapter', '--format-coefficient'):
                self.assertNotIn(flag, run.stdout)

    def test_default_and_sweep_budgets(self):
        args = arguments()
        config.validate_options(args, resumable=True)
        self.assertEqual((args.arm, args.rollouts, args.alpha, args.wandb_mode),
                         ('local_positive', 8, 1., 'online'))
        self.assertEqual((args.validation_mode, args.validation_every_epochs, args.validation_rollouts),
                         ('slurm', 1, 8))
        self.assertEqual(args.finetune_mode, 'full')
        self.assertEqual(args.learning_rate, 1e-6)
        lora = arguments('train', '--finetune-mode', 'lora')
        config.validate_options(lora, resumable=True)
        self.assertEqual(lora.learning_rate, 1e-4)
        self.assertEqual(config.schedule_steps(args, 2048), 128)
        for k in (4, 8, 16):
            args.rollouts = k
            self.assertEqual(config.schedule_steps(args, 2048), 128)
        args.mode = 'smoke'
        self.assertEqual(config.schedule_steps(args, 2048), 1)

    def test_invalid_or_ambiguous_experiment_rejected(self):
        for key, value in [('rollouts', 1), ('rollouts', 65), ('seed', -1), ('seed', 2**32),
                           ('alpha', -1), ('alpha', float('nan')),
                           ('learning_rate', 0), ('learning_rate', float('inf')),
                           ('batch_size', 3), ('epochs', 0),
                           ('smoke_cap', 8193), ('data_sha256', 'short')]:
            with self.subTest(key=key, value=value):
                args = arguments()
                setattr(args, key, value)
                with self.assertRaises(ValueError):
                    config.validate_options(args, resumable=True)
        args = arguments()
        args.resume_from = '/older/state_005'
        config.validate_options(args, resumable=True)
        with self.assertRaises(ValueError):
            config.validate_options(args, resumable=False)

    def test_removed_standalone_warmup_and_adapter_arguments_are_rejected(self):
        for resumable in (False, True):
            parser = argparse.ArgumentParser()
            config.add_training_arguments(parser, resumable=resumable)
            common = ['--model-path', 'model', '--data', 'train.json',
                      '--data-sha256', 'a'*64, '--out', 'run']
            for mode, extra in [('warmup', []),
                                ('train', ['--format-coefficient', '.1']),
                                ('train', ['--adapter', 'warmup/adapter_004'])]:
                with self.subTest(resumable=resumable, mode=mode, extra=extra):
                    with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                        parser.parse_args([mode, *common, *extra])
        for legacy_name, value in [('adapter', None), ('format_coefficient', 0.)]:
            args = arguments()
            setattr(args, legacy_name, value)
            with self.assertRaisesRegex(ValueError, 'removed'):
                config.validate_options(args, resumable=True)

    def test_every_question_once_per_epoch_and_whole_groups_per_rank(self):
        rows = [dict(id=i, question=f'Question {i}', ground_truth=i) for i in range(128)]
        visits = []
        for step in range(4):
            batch = epoch_batch(rows, step, 64, seed=42, epoch_offset=0)
            chunks = [rank_batch(batch, rank, 4) for rank in range(4)]
            self.assertEqual([r for chunk in chunks for r in chunk], batch)
            self.assertTrue(all(len(chunk) == 16 for chunk in chunks))
            visits += [r['id'] for r in batch]
        self.assertEqual(sorted(visits[:128]), list(range(128)))
        self.assertEqual(sorted(visits[128:]), list(range(128)))
        self.assertNotEqual(visits[:128], visits[128:])
        self.assertEqual(allocation('0,1,2,3,4,5,6,7', 2, 4), ('2', '6'))
        with self.assertRaises(ValueError):
            allocation('0,1,2,3', 2, 4)

    def test_frozen_data_content_checked(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'train.json'
            rows = [dict(id=str(i), question=f'Question {i}', ground_truth=i) for i in range(64)]
            path.write_text(json.dumps(rows))
            args = arguments()
            args.data, args.data_sha256 = str(path), config.digest(path)
            self.assertEqual(config.load_training_data(args)[1], rows)
            rows[0]['ground_truth'] += 1
            path.write_text(json.dumps(rows))
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                config.load_training_data(args)

    def test_invalid_ground_truth_and_duplicate_questions_fail_before_gpu(self):
        valid = dict(id='one', question=r'Solve $x=1$', ground_truth=1)
        for invalid in (1.5, float('nan'), float('inf'), True, '1/2', 'x', ''):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                config.validate_rows([dict(valid, ground_truth=invalid)])
        with self.assertRaises(ValueError):
            config.validate_rows([valid, dict(valid, id='two', question=' SOLVE x = 1 ')])
        with self.assertRaises(ValueError):
            config.validate_rows([dict(valid, id=1), dict(valid, id='1', question='Other question')])

    def test_model_identity_tracks_weights_and_manifest_not_path(self):
        with tempfile.TemporaryDirectory() as folder:
            left, right = Path(folder) / 'left', Path(folder) / 'right'
            for path in (left, right):
                path.mkdir()
                (path / 'config.json').write_text('{"model_type":"qwen3"}')
                (path / 'model.safetensors').write_bytes(b'weights-a')
                (path / 'tokenizer.json').write_text('{"tokenizer":"a"}')
                (path / 'model_manifest.json').write_text('{"alias":"qwen4b","status":"download_complete"}')
            original = config.model_identity(left)
            self.assertEqual(config.model_identity(right), original)
            (left / 'model.safetensors').write_bytes(b'weights-b')
            self.assertNotEqual(config.model_identity(left)['content_sha256'], original['content_sha256'])
            with self.assertRaises(ValueError):
                config.resolve_model_path(left, 'qwen17b')
            (right / 'model_manifest.json').write_text('{"alias":"qwen4b","changed":true}')
            changed = config.model_identity(right)
            self.assertEqual(changed['content_sha256'], original['content_sha256'])
            self.assertNotEqual(changed['preparation_manifest_sha256'], original['preparation_manifest_sha256'])

    def test_all_arms_default_to_no_format_reward_or_warmup_schedule(self):
        for arm in config.ARMS:
            args = arguments('train', '--arm', arm)
            contract = config.scientific_contract(args, world_size=4, training_questions=2048,
                                                  identity={'content_sha256': 'model'})
            self.assertEqual(contract['version'], 7)
            self.assertEqual(contract['sampler_engine_seed'], args.seed)
            self.assertEqual(contract['python_hash_seed'], args.seed)
            self.assertEqual(contract['format_coefficient'], 0.)
            self.assertEqual(contract['warmup_steps'], 0)
            self.assertEqual(contract['epoch_offset'], 0)
            self.assertEqual(contract['diversity_alpha_schedule'], 'constant')
            self.assertNotIn('warmup_adapter', contract)

    def test_resume_rejects_changed_scientific_settings_and_model_identity(self):
        args = arguments('train', '--finetune-mode', 'lora')
        config.validate_options(args, resumable=True)
        base = config.scientific_contract(args, world_size=4, training_questions=2048,
                                         identity={'content_sha256': 'model-a', 'preparation_manifest_sha256': 'manifest-a'})
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            adapter = out / 'adapter_003'
            adapter.mkdir()
            (adapter / 'adapter_model.safetensors').write_bytes(b'adapter')
            (adapter / 'adapter_config.json').write_text('{}')
            checkpoints.write_json(out / 'adapter_003_manifest.json',
                                   {p.name: checkpoints.digest(p) for p in adapter.iterdir()})
            stage = checkpoints.prepare_state(out, 3)
            for name in ('optimizer.pt',) + tuple(f'rng_rank_{rank}.pt' for rank in range(4)):
                (stage / name).write_bytes(b'opaque fixture; not loaded as torch')
            state = checkpoints.commit_state(out, 3, 4, base)
            self.assertEqual(checkpoints.inspect_checkpoint(state, base, 4)['completed_step'], 3)
            for field, value in [('alpha', .5), ('K', 16), ('data_sha256', 'b'*64), ('seed', 43),
                                 ('model_identity', {'content_sha256': 'model-b'}),
                                 ('version', 2), ('initialization', 'warmup_adapter'),
                                 ('epoch_offset', 1), ('format_coefficient', .1),
                                 ('optimizer', dict(base['optimizer'], lr=2e-4))]:
                with self.subTest(field=field):
                    changed = copy.deepcopy(base)
                    changed[field] = value
                    with self.assertRaisesRegex(ValueError, 'contract changed'):
                        checkpoints.inspect_checkpoint(state, changed, 4)

    def test_evaluation_adapter_still_requires_matching_base_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            adapter = root / 'adapter_004'
            adapter.mkdir()
            identity = {'content_sha256': 'base-a'}
            started = {'model_identity': identity, 'args': {'mode': 'train'}}
            (root / 'started.json').write_text(json.dumps(started))
            config.validate_adapter_origin(adapter, identity)
            with self.assertRaises(ValueError):
                config.validate_adapter_origin(adapter, {'content_sha256': 'base-b'})
            (root / 'started.json').unlink()
            with self.assertRaises(ValueError):
                config.validate_adapter_origin(adapter, identity)

    def test_fresh_and_resumed_state_use_same_contract_and_different_start(self):
        from trainer_resumable import initial_training_state
        args = arguments()
        identity = {'content_sha256': 'model'}
        with patch('trainer_resumable.version', return_value='test'):
            fresh = initial_training_state(args, 4, 2048, identity)
            args.resume_from = '/previous/state_032'
            verified_state = dict(completed_step=32, adapter_path='/previous/adapter_032',
                                  state_path=args.resume_from, contract=fresh[0])
            with patch('trainer_resumable.checkpoints.inspect_checkpoint', return_value=verified_state) as inspect:
                continued = initial_training_state(args, 4, 2048, identity)
                inspect.assert_called_once_with(args.resume_from, fresh[0], 4)
            self.assertEqual(continued[0], fresh[0])
            self.assertEqual(continued[1], verified_state)
            self.assertEqual(continued[2:], (32, 128))
            with patch('trainer_resumable.checkpoints.inspect_checkpoint', return_value={'completed_step': 129}):
                with self.assertRaisesRegex(ValueError, 'outside'):
                    initial_training_state(args, 4, 2048, identity)


if __name__ == '__main__':
    unittest.main()
