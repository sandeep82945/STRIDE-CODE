"""CPU-only configuration and launch tests; no credentials or external calls."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import env_config
import run


class EnvConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_literal_values_and_existing_exports_take_precedence(self):
        path = self.root / '.env'
        path.write_text(' # ignored\nexport WANDB_PROJECT=from-file\n'
                        'WANDB_API_KEY="fixture#123" # comment\n'
                        'WANDB_RUN_GROUP="$(touch NOT_EXECUTED) `uname` ${SECRET}"\n'
                        'WANDB_ENTITY=\n')
        target = {'WANDB_PROJECT': 'from-shell'}
        result = env_config.load_env_file(path, environ=target)
        self.assertEqual(result['WANDB_PROJECT'], 'from-file')
        self.assertEqual(target['WANDB_PROJECT'], 'from-shell')
        self.assertEqual(target['WANDB_API_KEY'], 'fixture#123')
        self.assertEqual(target['WANDB_RUN_GROUP'], '$(touch NOT_EXECUTED) `uname` ${SECRET}')
        self.assertEqual(target['WANDB_ENTITY'], '')
        self.assertFalse((self.root / 'NOT_EXECUTED').exists())

    def test_invalid_file_does_not_partially_mutate_or_reveal_values(self):
        for suffix in ('UNKNOWN=x', 'WANDB_API_KEY=second', 'WANDB_RUN_GROUP="unclosed',
                       'WANDB_RUN_GROUP=two words', 'WANDB_MODE'):
            path = self.root / '.env'
            path.write_text('WANDB_API_KEY=fixture_super_secret\n' + suffix + '\n')
            target = {}
            with self.subTest(suffix=suffix), self.assertRaises(ValueError) as caught:
                env_config.load_env_file(path, environ=target)
            self.assertEqual(target, {})
            self.assertNotIn('fixture_super_secret', str(caught.exception))

    def test_missing_default_optional_explicit_and_symlinks_rejected(self):
        with patch.object(env_config, 'ROOT', self.root / 'missing'):
            self.assertEqual(env_config.load_env_file(environ={}), {})
        with self.assertRaises(ValueError):
            env_config.load_env_file(self.root / 'missing', environ={})
        real = self.root / 'real'; real.write_text('WANDB_MODE=offline')
        link = self.root / 'linked'; link.symlink_to(real)
        dangling = self.root / 'dangling'; dangling.symlink_to(self.root / 'absent')
        for path in (link, dangling, self.root):
            with self.subTest(path=path), self.assertRaises(ValueError):
                env_config.load_env_file(path, environ={})
        invalid = self.root / 'invalid'; invalid.write_bytes(b'\xff')
        with self.assertRaisesRegex(ValueError, 'Unable to read'):
            env_config.load_env_file(invalid, environ={})

    def test_train_defaults_enable_online_and_epoch_validation(self):
        env = {'WANDB_API_KEY': 'fixture_not_a_real_key', 'WANDB_DISABLE_CODE': 'false'}
        result = env_config.runtime_arguments('train', [], environ=env)
        for flag, expected in (('--wandb-project', 'stride-iclr2027'), ('--wandb-mode', 'online'),
                               ('--wandb-group', 'stride-comparison'), ('--validation-mode', 'slurm'),
                               ('--validation-every-epochs', '1'), ('--validation-rollouts', '8')):
            self.assertEqual(run.option(result, flag), expected)
        self.assertEqual(run.option(result, '--validation-data'), str(env_config.ROOT / 'data/val.json'))
        self.assertNotIn('fixture_not_a_real_key', ' '.join(result))
        self.assertEqual(env['WANDB_DISABLE_CODE'], 'true')

    def test_cli_options_win_over_environment(self):
        env = dict(WANDB_API_KEY='fixture_not_a_real_key', WANDB_MODE='online',
                   WANDB_PROJECT='environment-project', WANDB_RUN_GROUP='environment-group',
                   WANDB_ENTITY='environment-entity', VALIDATION_MODE='slurm',
                   VALIDATION_DATA='environment-data', VALIDATION_ROLLOUTS='8',
                   VALIDATION_EVERY_EPOCHS='1', VALIDATION_PARTITION='environment-partition')
        arguments = ['--wandb-mode=offline', '--wandb-project', 'cli-project',
                     '--wandb-group', 'cli-group', '--wandb-entity', 'cli-entity',
                     '--validation-mode=off', '--validation-data', 'cli-data',
                     '--validation-rollouts=4', '--validation-every-epochs=2',
                     '--validation-partition=cli-partition']
        before = list(arguments)
        result = env_config.runtime_arguments('train', arguments, environ=env)
        self.assertEqual(result, before)
        self.assertEqual(arguments, before)
        self.assertEqual(env['WANDB_MODE'], 'offline')

    def test_eval_does_not_get_training_validation_flags(self):
        result = env_config.runtime_arguments('eval', [], environ={}, require_key=False)
        self.assertEqual(run.option(result, '--wandb-mode'), 'online')
        self.assertFalse(any(value.startswith('--validation-') for value in result))

    def test_smoke_defaults_disable_validation_and_wandb(self):
        result = env_config.runtime_arguments('smoke', [], environ={})
        self.assertEqual(run.option(result, '--wandb-mode'), 'disabled')
        self.assertEqual(run.option(result, '--validation-mode'), 'off')

    def test_online_key_required_but_not_for_dry_run_offline_or_disabled(self):
        for key in ('', 'short', 'your_wandb_api_key', 'paste_your_key', 'white space key', 'fixture\x00secret'):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'Set WANDB_API_KEY'):
                env_config.runtime_arguments('train', [], environ={'WANDB_API_KEY': key})
        env_config.runtime_arguments('train', [], environ={}, require_key=False)
        for mode in ('offline', 'disabled'):
            env_config.runtime_arguments('train', ['--wandb-mode', mode], environ={})

    def test_no_secret_in_arguments_metadata_or_errors(self):
        secret = 'fixture_super_secret'
        base = {'WANDB_API_KEY': secret}
        for key in ('WANDB_PROJECT', 'WANDB_ENTITY', 'WANDB_RUN_GROUP', 'VALIDATION_DATA', 'VALIDATION_ACCOUNT'):
            with self.subTest(key=key), self.assertRaises(ValueError) as caught:
                env_config.runtime_arguments('train', [], environ={**base, key: 'contains-' + secret})
            self.assertNotIn(secret, str(caught.exception))
        for args in (['--out', '/some/' + secret], ['--wandb-project=' + secret],
                     ['--wandb-api-key', secret], ['--api-key=unknown-secret']):
            with self.subTest(args=args), self.assertRaises(ValueError) as caught:
                env_config.runtime_arguments('train', args, environ=base)
            self.assertNotIn(secret, str(caught.exception))

    def test_hub_token_never_forwarded_to_arguments_or_runtime_metadata(self):
        secret = 'hf_fixture_super_secret'
        env = {'HF_TOKEN': secret, 'WANDB_MODE': 'disabled'}
        for args in (['--out', '/some/' + secret], ['--wandb-project=' + secret],
                     ['--hf-token', 'fixture_other_secret'], ['--hf-token=fixture_other_secret']):
            with self.subTest(args=args), self.assertRaises(ValueError) as caught:
                env_config.runtime_arguments('train', args, environ=env)
            self.assertNotIn(secret, str(caught.exception))
        with self.assertRaises(ValueError):
            env_config.runtime_arguments('train', [], environ={**env, 'VALIDATION_ACCOUNT': secret})

    def test_configuration_constraints_fail_before_gpu_work(self):
        for flag, bad in (('--wandb-mode', 'bad'), ('--wandb-project', ''),
                          ('--wandb-group', 'line\nbreak'), ('--validation-mode', 'bad'),
                          ('--validation-data', ''), ('--validation-every-epochs', '0'),
                          ('--validation-rollouts', '65'), ('--validation-rollouts', 'two')):
            with self.subTest(flag=flag, bad=bad), self.assertRaises(ValueError):
                env_config.runtime_arguments('train', [flag, bad], environ={}, require_key=False)
        for args in (['--wandb-mode'], ['--wandb-mode', '--out', 'x'],
                     ['--wandb-mode=online', '--wandb-mode=offline']):
            with self.assertRaises(ValueError):
                env_config.runtime_arguments('train', args, environ={}, require_key=False)

    def launch(self, *, dry_run=False, extra=(), key='fixture_not_a_real_key', stage='train'):
        data = self.root / 'data.json'; data.write_text('[]')
        envfile = self.root / '.env'
        envfile.write_text(f'WANDB_API_KEY={key}\nWANDB_PROJECT=from-file\n')
        arguments = ['run', stage, '--env-file', str(envfile), '--hf-upload', 'off', '--model-path', '/fixture',
                     '--data', str(data), '--out', str(self.root / 'run'), *extra]
        if dry_run:
            arguments += ['--dry-run']
        with patch.dict(os.environ, {}, clear=True), \
                patch.object(run, 'execution_environment', side_effect=lambda stage, seed, environ: environ) as execution, \
                patch.object(run, 'freeze_source', return_value=self.root / 'snapshot') as snapshot, \
                patch.object(run.subprocess, 'call', return_value=17) as called, \
                patch('sys.argv', arguments), contextlib.redirect_stdout(io.StringIO()) as output:
            status = run.main()
        return status, called, execution, snapshot, output.getvalue()

    def test_launcher_loads_dotenv_without_key_on_command_line(self):
        status, called, execution, snapshot, output = self.launch(extra=('--wandb-project', 'explicit'))
        self.assertEqual(status, 17)
        self.assertEqual(called.call_count, 1)
        command = called.call_args.args[0]
        self.assertEqual(run.option(command, '--wandb-project'), 'explicit')
        self.assertNotIn('--env-file', command)
        self.assertNotIn('fixture_not_a_real_key', ' '.join(command) + output)
        self.assertEqual(called.call_args.kwargs['env']['WANDB_API_KEY'], 'fixture_not_a_real_key')
        execution.assert_called_once(); snapshot.assert_called_once()

    def test_dry_run_prints_settings_without_secrets_or_side_effects(self):
        for key in ('', 'fixture_not_a_real_key'):
            status, called, execution, snapshot, output = self.launch(dry_run=True, key=key)
            self.assertEqual(status, 0)
            called.assert_not_called(); execution.assert_not_called(); snapshot.assert_not_called()
            self.assertEqual(run.option(json.loads(output)['command'], '--validation-mode'), 'slurm')
            self.assertNotIn('fixture_not_a_real_key', output)

    def test_missing_key_rejected_before_gpu_allocation_or_launch(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            self.launch(key='')
        self.assertEqual(caught.exception.code, 2)

    def test_release_includes_template_but_never_local_secrets(self):
        module_path = Path(run.__file__).parent / 'scripts/package_release.py'
        spec = importlib.util.spec_from_file_location('env_release_test', module_path)
        release = importlib.util.module_from_spec(spec); spec.loader.exec_module(release)
        for name in ('.env', '.env.example', '.env.production', 'env_config.py',
                     'training/.env', 'training/.env.example', 'docs/.env.txt'):
            path = self.root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('fixture')
        files = {p.relative_to(self.root).as_posix() for p in release.source_files(self.root)}
        self.assertEqual(files, {'.env.example', 'env_config.py'})


if __name__ == '__main__':
    unittest.main()
