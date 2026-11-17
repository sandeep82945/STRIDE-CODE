"""Mocked launcher lifecycle tests: no GPU work, Hub calls, or real secrets."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run
import upload_checkpoints


HF_KEY = 'hf_fixture_upload_token_not_real'
WANDB_KEY = 'fixture_wandb_key_not_real'


class HuggingFaceLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.settings = self.root / 'run_settings.toml'
        self.settings.write_text('')
        self.data = self.root / 'train.json'
        self.data.write_text('[]')
        self.envfile = self.root / '.env'
        self.out = self.root / 'completed_run'
        self.out.mkdir()
        (self.out / 'done.json').write_text('{}')
        self.frozen = self.root / 'source_snapshot'

    def environment_file(self, *, hf_key=HF_KEY, namespace='fixture-team'):
        self.envfile.write_text(f'WANDB_API_KEY={WANDB_KEY}\nHF_TOKEN={hf_key}\nHF_USERNAME={namespace}\n')

    def invoke(self, *, stage='train', extra=(), statuses=(0, 0, 0), dry_run=False,
               hf_key=HF_KEY, namespace='fixture-team', exported=None):
        self.environment_file(hf_key=hf_key, namespace=namespace)
        argv = ['run.py', stage, '--settings', str(self.settings), '--env-file', str(self.envfile),
                '--model-path', '/fixture/model', '--data', str(self.data), '--out', str(self.out), *extra]
        if dry_run:
            argv.append('--dry-run')
        env = {'CUDA_VISIBLE_DEVICES': '0' if stage == 'eval' else '0,1,2,3,4,5,6,7',
               'HF_HUB_OFFLINE': '1', **(exported or {})}
        with patch.dict(os.environ, env, clear=True), patch('sys.argv', argv), \
                patch.object(run, 'freeze_source', return_value=self.frozen) as freeze, \
                patch.object(run, 'execution_environment', wraps=run.execution_environment) as execution, \
                patch.object(run.subprocess, 'call', side_effect=statuses) as call, \
                contextlib.redirect_stdout(io.StringIO()) as stdout, \
                contextlib.redirect_stderr(io.StringIO()) as stderr:
            parser_error = False
            try:
                code = run.main()
            except SystemExit as exc:
                code, parser_error = exc.code, True
        return dict(code=code, parser_error=parser_error, call=call, freeze=freeze,
                    execution=execution, stdout=stdout.getvalue(), stderr=stderr.getvalue())

    def assert_secret_free_output_and_commands(self, result):
        printed = result['stdout'] + result['stderr']
        for item in result['call'].call_args_list:
            printed += ' '.join(item.args[0])
        self.assertNotIn(HF_KEY, printed)
        self.assertNotIn(WANDB_KEY, printed)

    def test_success_authenticates_then_trains_then_uploads_selected_epochs(self):
        result = self.invoke(extra=('--hf-epochs', '2,4', '--hf-repo-prefix', 'paper-study',
                                    '--hf-description', 'Four epoch comparison', '--hf-retries', '2'))
        self.assertEqual(result['code'], 0)
        calls = result['call'].call_args_list
        self.assertEqual(len(calls), 3)
        preflight, train, publish = [item.args[0] for item in calls]
        self.assertIn('--preflight', preflight)
        self.assertEqual(preflight[2], str(self.frozen / 'hf_publish.py'))
        self.assertIn('torch.distributed.run', train)
        self.assertIn(str(self.frozen / 'trainer_resumable.py'), train)
        self.assertEqual(publish[2], str(self.frozen / 'hf_publish.py'))
        self.assertNotIn('--preflight', publish)
        self.assertEqual(run.option(publish, '--epochs'), '2,4')
        self.assertEqual(run.option(publish, '--repo-prefix'), 'paper-study')
        self.assertEqual(run.option(publish, '--description'), 'Four epoch comparison')
        self.assertEqual(run.option(publish, '--retries'), '2')
        self.assertEqual(run.option(publish, '--run-dir'), str(self.out))
        self.assertEqual(run.option(preflight, '--namespace'), 'fixture-team')
        self.assertFalse(any(arg.startswith('--hf-') for arg in train))
        self.assertNotIn('--env-file', train)
        self.assert_secret_free_output_and_commands(result)

    def test_publication_and_training_have_separate_credentials_and_network_modes(self):
        result = self.invoke(exported={'RANK': '9', 'LOCAL_RANK': '1', 'MASTER_PORT': '5678'})
        preflight, train, publish = [call.kwargs['env'] for call in result['call'].call_args_list]
        self.assertNotIn('HF_TOKEN', train)
        self.assertEqual(train['WANDB_API_KEY'], WANDB_KEY)
        self.assertEqual(train['HF_HUB_OFFLINE'], '1')
        self.assertEqual(train['TRANSFORMERS_OFFLINE'], '1')
        self.assertEqual(train['CUDA_VISIBLE_DEVICES'], '0,1,2,3,4,5,6,7')
        for env in (preflight, publish):
            self.assertEqual(env['HF_TOKEN'], HF_KEY)
            self.assertNotIn('WANDB_API_KEY', env)
            self.assertEqual(env['HF_HUB_OFFLINE'], '0')
            self.assertEqual(env['HF_ENDPOINT'], 'https://huggingface.co')
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '')
            self.assertEqual(env['WANDB_MODE'], 'disabled')
            for name in ('RANK', 'LOCAL_RANK', 'MASTER_PORT'):
                self.assertNotIn(name, env)
        self.assert_secret_free_output_and_commands(result)

    def test_failed_preflight_never_starts_training_or_upload(self):
        result = self.invoke(statuses=(9,))
        self.assertEqual(result['code'], 9)
        result['call'].assert_called_once()
        self.assertIn('--preflight', result['call'].call_args.args[0])
        self.assertIn('training was not started', result['stderr'])
        self.assert_secret_free_output_and_commands(result)

    def test_failed_training_never_uploads(self):
        result = self.invoke(statuses=(0, 17))
        self.assertEqual(result['code'], 17)
        self.assertEqual(result['call'].call_count, 2)
        self.assertIn('torch.distributed.run', result['call'].call_args.args[0])
        self.assertNotIn('Training completed', result['stderr'])

    def test_upload_failure_preserves_nonzero_status_and_explains_retry(self):
        result = self.invoke(statuses=(0, 0, 6))
        self.assertEqual(result['code'], 6)
        self.assertEqual(result['call'].call_count, 3)
        self.assertIn('Training completed', result['stderr'])
        self.assertIn('upload_checkpoints.py', result['stderr'])
        self.assertIn('without retraining', result['stderr'])

    def test_dry_run_without_hub_credentials_has_no_launch_or_snapshot(self):
        result = self.invoke(dry_run=True, hf_key='', namespace='')
        self.assertEqual(result['code'], 0)
        for name in ('call', 'freeze', 'execution'):
            result[name].assert_not_called()
        payload = json.loads(result['stdout'])
        self.assertTrue(payload['huggingface']['enabled'])
        self.assertEqual(payload['huggingface']['epochs'], [2, 3, 4])
        self.assertEqual(payload['huggingface']['namespace'], '')
        self.assertEqual(payload['required_gpus'], 8)
        self.assert_secret_free_output_and_commands(result)

    def test_missing_credentials_fail_before_gpu_checks_or_snapshot(self):
        for credentials in (dict(hf_key=''), dict(namespace='')):
            with self.subTest(credentials=credentials):
                result = self.invoke(**credentials)
                self.assertEqual(result['code'], 2)
                self.assertTrue(result['parser_error'])
                for name in ('call', 'freeze', 'execution'):
                    result[name].assert_not_called()
                self.assert_secret_free_output_and_commands(result)

    def test_no_reached_epoch_fails_before_gpu_checks_or_snapshot(self):
        result = self.invoke(extra=('--epochs', '1'))
        self.assertEqual(result['code'], 2)
        self.assertIn('No selected Hugging Face epoch', result['stderr'])
        for name in ('call', 'freeze', 'execution'):
            result[name].assert_not_called()

    def test_cli_can_turn_upload_off_without_hub_credentials(self):
        result = self.invoke(extra=('--hf-upload', 'off'), hf_key='', namespace='', statuses=(0,))
        self.assertEqual(result['code'], 0)
        result['call'].assert_called_once()
        self.assertIn('torch.distributed.run', result['call'].call_args.args[0])

    def test_smoke_and_eval_ignore_explicit_upload_enablement(self):
        for stage in ('smoke', 'eval'):
            with self.subTest(stage=stage):
                result = self.invoke(stage=stage, extra=('--hf-upload', 'on'), hf_key='', namespace='', statuses=(0,))
                self.assertEqual(result['code'], 0)
                result['call'].assert_called_once()
                command = result['call'].call_args.args[0]
                self.assertNotIn('--preflight', command)
                self.assertNotIn(str(self.frozen / 'hf_publish.py'), command)
                self.assert_secret_free_output_and_commands(result)

    def test_hub_token_embedded_in_public_config_fails_before_launch(self):
        for extra in (('--hf-description', HF_KEY), ('--out', str(self.root / HF_KEY)),
                      ('--wandb-project', HF_KEY)):
            with self.subTest(extra=extra):
                result = self.invoke(extra=extra)
                self.assertEqual(result['code'], 2)
                for name in ('call', 'freeze', 'execution'):
                    result[name].assert_not_called()
                self.assert_secret_free_output_and_commands(result)

    def retry(self, *, extra=(), statuses=(0,), dry_run=False, credentials=True):
        self.environment_file(hf_key=HF_KEY if credentials else '', namespace='fixture-team' if credentials else '')
        argv = ['upload_checkpoints.py', '--run-dir', str(self.out), '--settings', str(self.settings),
                '--env-file', str(self.envfile), *extra]
        if dry_run:
            argv.append('--dry-run')
        with patch.dict(os.environ, {}, clear=True), patch('sys.argv', argv), \
                patch.object(upload_checkpoints, 'freeze_source', return_value=self.frozen) as freeze, \
                patch.object(run, 'execution_environment') as execution, \
                patch.object(upload_checkpoints.subprocess, 'call', side_effect=statuses) as call, \
                contextlib.redirect_stdout(io.StringIO()) as stdout, \
                contextlib.redirect_stderr(io.StringIO()) as stderr:
            parser_error = False
            try:
                code = upload_checkpoints.main()
            except SystemExit as exc:
                code, parser_error = exc.code, True
        return dict(code=code, parser_error=parser_error, call=call, freeze=freeze,
                    execution=execution, stdout=stdout.getvalue(), stderr=stderr.getvalue())

    def test_standalone_retry_only_uploads_and_never_allocates_gpu(self):
        result = self.retry(extra=('--hf-epochs', '2,3', '--hf-retries', '2'))
        self.assertEqual(result['code'], 0)
        result['execution'].assert_not_called()
        result['call'].assert_called_once()
        result['freeze'].assert_called_once()
        command = result['call'].call_args.args[0]
        self.assertEqual(command[2], str(self.frozen / 'hf_publish.py'))
        self.assertNotIn('torch.distributed.run', command)
        self.assertNotIn('--preflight', command)
        self.assertEqual(run.option(command, '--epochs'), '2,3')
        self.assertEqual(run.option(command, '--retries'), '2')
        env = result['call'].call_args.kwargs['env']
        self.assertEqual(env['HF_TOKEN'], HF_KEY)
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '')
        self.assertNotIn('WANDB_API_KEY', env)
        self.assert_secret_free_output_and_commands(result)

    def test_standalone_dry_run_needs_no_credentials_and_has_no_side_effects(self):
        result = self.retry(dry_run=True, credentials=False)
        self.assertEqual(result['code'], 0)
        for name in ('call', 'freeze', 'execution'):
            result[name].assert_not_called()
        payload = json.loads(result['stdout'])
        self.assertTrue(payload['configuration_only'])
        self.assertEqual(payload['network_calls'], 0)
        self.assert_secret_free_output_and_commands(result)

    def test_standalone_retry_requires_completed_run_and_credentials(self):
        result = self.retry(credentials=False)
        self.assertEqual(result['code'], 2)
        for name in ('call', 'freeze', 'execution'):
            result[name].assert_not_called()
        (self.out / 'done.json').unlink()
        result = self.retry()
        self.assertEqual(result['code'], 2)
        self.assertIn('completed training run', result['stderr'])
        for name in ('call', 'freeze', 'execution'):
            result[name].assert_not_called()

    def test_standalone_retry_propagates_upload_failure(self):
        result = self.retry(statuses=(4,))
        self.assertEqual(result['code'], 4)
        result['call'].assert_called_once()
        result['execution'].assert_not_called()

    def test_description_starting_with_dashes_reaches_real_worker_parser(self):
        import sys
        root = Path(run.__file__).parent
        with patch.object(sys, 'path', [str(root / 'training'), *sys.path]):
            import hf_publish
        config = dict(namespace='fixture-team', epochs=[2, 3, 4],
                      repo_prefix='stride-test', description='--careful experiment', retries=2)
        command = run.hf_upload_command(config, root / 'training/hf_publish.py', run_dir=self.out)
        with patch.dict(os.environ, {}, clear=True), \
                patch.object(hf_publish, 'upload_run', return_value={'complete': True}) as upload, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(hf_publish.main(command[3:]), 0)
        self.assertEqual(upload.call_args.kwargs['description'], '--careful experiment')


if __name__ == '__main__':
    unittest.main()
