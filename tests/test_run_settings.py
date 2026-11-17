"""Root settings merge and launcher behavior, without GPU or service access."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import env_config
import run
import run_settings


class RunSettingsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.path = self.root / 'run_settings.toml'
        self.path.write_text('')

    def resolve(self, stage='train', args=(), env=None, contents=''):
        self.path.write_text(contents)
        environ = dict(env or {})
        result = run_settings.settings_arguments(stage, list(args), run_settings.load_settings(self.path), environ=environ)
        return env_config.runtime_arguments(stage, result, environ=environ, require_key=False)

    def value(self, result, name):
        return run.option(result, '--' + name)

    def test_shipped_file_matches_stable_defaults(self):
        loaded = run_settings.load_settings()
        self.assertEqual(loaded.values, run_settings.DEFAULTS)
        resolved = self.resolve()
        for flag, value in dict(model='qwen4b', arm='local_positive', seed='42',
                                epochs='4', rollouts='8', alpha='1.0').items():
            self.assertEqual(self.value(resolved, flag), value)
        self.assertEqual(self.value(resolved, 'finetune-mode'), 'full')
        self.assertEqual(self.value(resolved, 'batch-size'), '64')
        self.assertEqual(self.value(resolved, 'out'), str(self.root/'runs/qwen4b/full/comparison_seed42/local_positive'))
        self.assertEqual(self.value(resolved, 'wandb-mode'), 'online')
        self.assertEqual(self.value(resolved, 'validation-mode'), 'slurm')

    def test_alternate_settings_paths_use_config_directory(self):
        results = self.resolve(contents='[training]\nmodel_path="weights/{model}"\ndata="split.json"\nout="results/{arm}"\n[validation]\ndata="validation.json"')
        for flag, suffix in (('model-path', 'weights/qwen4b'), ('data', 'split.json'),
                             ('out', 'results/local_positive'), ('validation-data', 'validation.json')):
            self.assertEqual(self.value(results, flag), str(self.root/suffix))

    def test_all_arms_and_models_modes_keep_data_and_budget_identical(self):
        expected_data = str(self.root/'data/train.json')
        for model in ('qwen4b', 'qwen17b'):
            for mode in ('full', 'lora'):
                for arm in ('local_positive', 'single_step', 'token_uniform', 'grpo'):
                    with self.subTest(model=model, mode=mode, arm=arm):
                        result = self.resolve(env=dict(MODEL=model, FINETUNE_MODE=mode, ARM=arm, SEED='43'))
                        self.assertEqual(self.value(result, 'model-path'), str(self.root/'models'/model))
                        self.assertEqual(self.value(result, 'out'), str(self.root/f'runs/{model}/{mode}/comparison_seed43/{arm}'))
                        self.assertEqual(self.value(result, 'seed'), '43')
                        self.assertEqual(self.value(result, 'data'), expected_data)
                        self.assertEqual(self.value(result, 'batch-size'), '64')
                        self.assertIsNone(self.value(result, 'learning-rate'))

    def test_cli_over_environment_over_toml_and_paths_stay_literal(self):
        contents = '[training]\nmodel="qwen17b"\narm="single_step"\nseed=10\nfinetune_mode="full"\nepochs=2\n'
        env = dict(MODEL='qwen4b', ARM='grpo', SEED='11', FINETUNE_MODE='lora', EPOCHS='3',
                   WANDB_MODE='offline', WANDB_PROJECT='env-project', VALIDATION_MODE='off', VALIDATION_DATA='env-relative.json')
        cli = ['--seed=12', '--arm', 'token_uniform', '--epochs', '5', '--data', 'cli-relative.json',
               '--model-path', 'cli-model', '--wandb-project=cli-project', '--validation-rollouts=4']
        original = list(cli)
        result = self.resolve(args=cli, env=env, contents=contents)
        for flag, expected in (('model', 'qwen4b'), ('finetune-mode', 'lora'), ('arm', 'token_uniform'),
                               ('seed', '12'), ('epochs', '5'), ('data', 'cli-relative.json'),
                               ('model-path', 'cli-model'), ('wandb-project', 'cli-project'),
                               ('wandb-mode', 'offline'), ('validation-mode', 'off'),
                               ('validation-data', 'env-relative.json'), ('validation-rollouts', '4')):
            self.assertEqual(self.value(result, flag), expected)
        self.assertEqual(self.value(result, 'out'), str(self.root/'runs/qwen4b/lora/comparison_seed12/token_uniform'))
        self.assertEqual(cli, original)

    def test_dotenv_overrides_toml_but_exported_and_cli_win(self):
        envpath = self.root / '.env'
        envpath.write_text('WANDB_PROJECT=file-project\nWANDB_MODE=offline\nVALIDATION_ROLLOUTS=4\n')
        env = {'WANDB_PROJECT': 'shell-project'}
        env_config.load_env_file(envpath, environ=env)
        result = self.resolve(env=env, contents='[wandb]\nproject="toml-project"\n[validation]\nrollouts=2\n')
        self.assertEqual(self.value(result, 'wandb-project'), 'shell-project')
        self.assertEqual(self.value(result, 'wandb-mode'), 'offline')
        self.assertEqual(self.value(result, 'validation-rollouts'), '4')
        result = self.resolve(env=env, args=['--wandb-project', 'cli-project'])
        self.assertEqual(self.value(result, 'wandb-project'), 'cli-project')

    def test_eval_inherits_only_wandb_configuration(self):
        result = self.resolve(stage='eval', env=dict(MODEL='qwen17b', ARM='grpo', SEED='99', VALIDATION_MODE='slurm'),
                              contents='[training]\nmodel="qwen17b"\n[wandb]\nmode="offline"\nproject="eval-project"')
        self.assertEqual(self.value(result, 'wandb-project'), 'eval-project')
        self.assertEqual(self.value(result, 'wandb-mode'), 'offline')
        self.assertTrue(all(not arg.startswith(('--validation-', '--model', '--arm', '--data', '--out', '--seed', '--epochs')) for arg in result))

    def test_smoke_disables_observers_and_separates_default_output(self):
        result = self.resolve(stage='smoke')
        self.assertEqual(self.value(result, 'wandb-mode'), 'disabled')
        self.assertEqual(self.value(result, 'validation-mode'), 'off')
        self.assertTrue(self.value(result, 'out').endswith('/local_positive_smoke'))
        training_out = Path(self.value(self.resolve(), 'out'))
        smoke_out = Path(self.value(result, 'out'))
        self.assertEqual(smoke_out.parent, training_out.parent)
        self.assertFalse(smoke_out.is_relative_to(training_out))
        explicit = self.resolve(stage='smoke', args=['--out', 'explicit-smoke', '--wandb-mode', 'offline'])
        self.assertEqual(self.value(explicit, 'out'), 'explicit-smoke')
        self.assertEqual(self.value(explicit, 'wandb-mode'), 'offline')

    def test_learning_rate_auto_tracks_mode_and_numeric_override_survives(self):
        for mode, expected in (('full', 1e-6), ('lora', 1e-4)):
            result = self.resolve(args=['--finetune-mode', mode])
            self.assertIsNone(self.value(result, 'learning-rate'))
            # The existing driver resolves automatic LR after selecting mode.
            import sys
            with patch.object(sys, 'path', [str(run_settings.ROOT/'training'), *sys.path]):
                from runtime_config import DEFAULT_LEARNING_RATES
                self.assertEqual(DEFAULT_LEARNING_RATES[self.value(result, 'finetune-mode')], expected)
        self.assertEqual(self.value(self.resolve(contents='[training]\nlearning_rate=0.0002'), 'learning-rate'), '0.0002')
        self.assertEqual(self.value(self.resolve(env={'LEARNING_RATE': '0.0003'}, contents='[training]\nlearning_rate=0.0002'), 'learning-rate'), '0.0003')
        self.assertEqual(self.value(self.resolve(args=['--learning-rate', '0.0004'], env={'LEARNING_RATE': '0.0003'}), 'learning-rate'), '0.0004')
        self.assertIsNone(self.value(self.resolve(env={'LEARNING_RATE': 'auto'}, contents='[training]\nlearning_rate=0.0002'), 'learning-rate'))
        for cli in (['--learning-rate', 'auto'], ['--learning-rate=auto']):
            self.assertIsNone(self.value(self.resolve(args=cli, env={'LEARNING_RATE': '0.0003'}, contents='[training]\nlearning_rate=0.0002'), 'learning-rate'))

    def test_boolean_override_can_disable_eager_config(self):
        self.assertIn('--sampler-eager', self.resolve(contents='[training]\nsampler_eager=true'))
        result = self.resolve(args=['--no-sampler-eager'], contents='[training]\nsampler_eager=true')
        self.assertNotIn('--sampler-eager', result)
        self.assertNotIn('--no-sampler-eager', result)
        for args in (['--sampler-eager', '--no-sampler-eager'], ['--sampler-eager=false']):
            with self.assertRaises(ValueError):
                self.resolve(args=args)

    def test_unknown_keys_and_invalid_types_values_rejected_without_echo(self):
        examples = [
            '[other]\nthing="fixture-secret"', '[training]\napi_key="fixture-secret"',
            '[training]\nbatch_size=true', '[training]\nbatch_size=3', '[training]\nrollouts=1',
            '[training]\nseed=-1', '[training]\nseed=4294967296', '[training]\nepochs=0',
            '[training]\nalpha=nan', '[training]\nlearning_rate=inf', '[training]\nlearning_rate=0',
            '[training]\nmodel="bad"', '[training]\narm="centered"', '[training]\nfinetune_mode="adapter"',
            '[training]\nsampler_max_num_seqs=8', '[training]\nsampler_eager="false"',
            '[validation]\nevery_epochs=0', '[validation]\nrollouts=65', '[validation]\nmode="inline"',
            '[wandb]\nproject=""', '[wandb]\nmode="bad"', '[training]\ndata=""',
            '[training]\nout="{model.__class__}"', '[training]\nout="{seed:03}"',
            '[training]\nmodel="fixture-secret',
        ]
        for text in examples:
            with self.subTest(text=text), self.assertRaises(ValueError) as caught:
                self.resolve(contents=text)
            self.assertNotIn('fixture-secret', str(caught.exception))

    def test_invalid_environment_and_duplicates_rejected(self):
        for env in (dict(SEED=' 43'), dict(SEED='4.3'), dict(SEED='43; echo x'), dict(SEED='-1'),
                    dict(ARM='centered'), dict(FINETUNE_MODE='bad'), dict(MODEL='other'),
                    dict(BATCH_SIZE='31'), dict(ALPHA='nan'), dict(LEARNING_RATE='-0.1')):
            with self.subTest(env=env), self.assertRaises(ValueError):
                self.resolve(env=env)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.resolve(args=['--arm=grpo', '--arm', 'local_positive'])

    def test_os_environment_mapping_and_explicit_literal_paths(self):
        settings = run_settings.load_settings(self.path)
        with patch.dict(os.environ, {'MODEL': 'qwen17b', 'FINETUNE_MODE': 'lora'}, clear=True):
            result = run_settings.settings_arguments('train', ['--out', 'literal/{folder}'], settings)
        self.assertEqual(self.value(result, 'model'), 'qwen17b')
        self.assertEqual(self.value(result, 'finetune-mode'), 'lora')
        self.assertEqual(self.value(result, 'out'), 'literal/{folder}')

    def test_missing_symlink_invalid_toml_are_rejected(self):
        for path in (self.root, self.root/'missing'):
            with self.assertRaises(ValueError):
                run_settings.load_settings(path)
        link = self.root/'link.toml'; link.symlink_to(self.path)
        with self.assertRaises(ValueError):
            run_settings.load_settings(link)
        self.path.write_bytes(b'\xff')
        with self.assertRaises(ValueError):
            run_settings.load_settings(self.path)

    def invoke(self, *extra):
        with patch.dict(os.environ, {}, clear=True), patch.object(env_config, 'ROOT', self.root), \
                patch('sys.argv', ['run.py', 'train', '--dry-run', *extra]), \
                patch.object(run, 'execution_environment') as execution, \
                patch.object(run, 'freeze_source') as snapshot, \
                patch.object(run.subprocess, 'call') as call, contextlib.redirect_stdout(io.StringIO()) as out:
            result = run.main()
        execution.assert_not_called(); snapshot.assert_not_called(); call.assert_not_called()
        self.assertEqual(result, 0)
        return json.loads(out.getvalue())['command']

    def test_dry_run_alternate_file_works_without_key_or_gpu(self):
        (self.root/'split.json').write_text('[]')
        self.path.write_text('[training]\nfinetune_mode="lora"\ndata="split.json"\n')
        command = self.invoke('--settings', str(self.path))
        self.assertEqual(self.value(command, 'data'), str(self.root/'split.json'))
        self.assertEqual(self.value(command, 'finetune-mode'), 'lora')
        self.assertNotIn('--settings', command)
        self.assertIn('--data-sha256', command)

    def test_no_settings_retains_explicit_driver_launch(self):
        data = self.root/'split.json'; data.write_text('[]')
        command = self.invoke('--no-settings', '--data', str(data), '--model-path', 'm', '--out', 'o')
        self.assertEqual(self.value(command, 'model-path'), 'm')
        self.assertEqual(self.value(command, 'out'), 'o')
        self.assertIsNone(self.value(command, 'finetune-mode'))
        self.assertNotIn('--no-settings', command)


if __name__ == '__main__':
    unittest.main()
