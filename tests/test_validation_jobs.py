"""CPU mocks only: async validation never allocates or waits for a real GPU."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training'))
import validation_jobs as jobs
import evaluate_async


def write_json(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ValidationJobsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.rows = [dict(id='train:1', question='Compute two plus three', ground_truth=5),
                     dict(id='train:2', question='Compute seven minus four', ground_truth=3)]
        self.val_rows = [dict(id='val:1', question='Find the square of six', ground_truth=36)]
        train, val = self.root / 'train.json', self.root / 'val.json'
        train_sha = write_json(train, self.rows)
        write_json(val, self.val_rows)
        self.source = self.root / 'frozen source'
        self.source.mkdir()
        (self.source / 'evaluate_async.py').write_text('# Frozen evaluator fixture\n')
        self.args = SimpleNamespace(mode='train', validation_mode='slurm', validation_data=str(val),
            validation_every_epochs=1, validation_rollouts=8, validation_partition='gpu',
            validation_account='research', validation_qos=None, out=str(self.root / 'run'),
            data=str(train), data_sha256=train_sha, seed=42, model='qwen4b',
            model_path=str(self.root / 'base model'), finetune_mode='full', arm='local_positive',
            generation_timeout=3600, wandb_mode='online', wandb_project='stride-iclr2027',
            wandb_entity=None, wandb_group='stride-comparison', sampler_eager=False)
        checker = patch.object(jobs.shutil, 'which', return_value='/usr/bin/sbatch')
        checker.start()
        self.addCleanup(checker.stop)
        self.submit = Mock(return_value=SimpleNamespace(returncode=0, stdout='12345;testcluster\n', stderr=''))

    def scheduler(self):
        metadata = jobs.prepare_validation(self.args, self.rows)
        return jobs.ValidationScheduler(self.args, metadata, source_root=self.source,
                     python='/test env/bin/python', submit=self.submit)

    def checkpoint(self, step=2):
        prefix = 'model' if self.args.finetune_mode == 'full' else 'adapter'
        path = Path(self.args.out) / f'{prefix}_{step:03d}'
        path.mkdir(parents=True)
        if prefix == 'model':
            write_json(path / 'config.json', {'model_type': 'qwen3'})
            (path / 'model.safetensors').write_bytes(b'not-real-weights')
        else:
            write_json(path / 'adapter_config.json', {'peft_type': 'LORA'})
            (path / 'adapter_model.safetensors').write_bytes(b'not-real-adapter')
        write_json(path.with_name(path.name + '_manifest.json'),
                   {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in path.iterdir()})
        return path

    def test_prepare_pins_exact_bytes_hashes_and_checks_overlap(self):
        metadata = jobs.prepare_validation(self.args, self.rows)
        self.assertEqual(Path(metadata['data']).read_bytes(), Path(self.args.validation_data).read_bytes())
        self.assertEqual(metadata['training_data_sha256'], self.args.data_sha256)
        self.assertEqual(metadata['overlap']['id_overlap'], 0)
        self.assertEqual(metadata['seed'], 42)
        Path(self.args.validation_data).write_text('changed after preparation')
        self.assertEqual(json.loads(Path(metadata['data']).read_text()), self.val_rows)

    def test_all_held_out_ids_and_normalized_questions_must_be_disjoint(self):
        for row in [dict(self.val_rows[0], id='train:1'),
                    dict(self.val_rows[0], question=' Compute  two plus three '),
                    dict(self.val_rows[0], question='Compute two plus three')]:
            with self.subTest(row=row):
                write_json(Path(self.args.validation_data), [row])
                with self.assertRaisesRegex(ValueError, 'overlap'):
                    jobs.prepare_validation(self.args, self.rows)

    def test_prepare_rejects_wrong_hash_or_partial_training_rows(self):
        self.args.data_sha256 = 'a' * 64
        with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
            jobs.prepare_validation(self.args, self.rows)
        self.args.data_sha256 = hashlib.sha256(Path(self.args.data).read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError, 'same complete training rows'):
            jobs.prepare_validation(self.args, self.rows[:1])
        self.args.validation_data_sha256 = 'b' * 64
        with self.assertRaisesRegex(ValueError, 'Validation dataset SHA256 mismatch'):
            jobs.prepare_validation(self.args, self.rows)

    def test_disabled_or_smoke_needs_no_files_or_scheduler(self):
        for mode, stage in [('off', 'train'), ('slurm', 'smoke')]:
            self.args.validation_mode, self.args.mode = mode, stage
            self.args.validation_data = None
            with patch.object(jobs.shutil, 'which', return_value=None):
                self.assertEqual(jobs.prepare_validation(self.args, None), {'mode': 'off'})
            scheduler = jobs.ValidationScheduler(self.args, {'mode': 'off'}, submit=self.submit)
            self.assertIsNone(scheduler.submit_checkpoint(2, '/missing', 2))
        self.submit.assert_not_called()

    def test_slurm_unavailable_fails_before_preparation(self):
        with patch.object(jobs.shutil, 'which', return_value=None):
            with self.assertRaisesRegex(ValueError, 'sbatch on PATH'):
                jobs.prepare_validation(self.args, self.rows)
        self.assertFalse(Path(self.args.out).exists())

    def test_full_epoch_job_is_independent_same_seed_and_source(self):
        scheduler = self.scheduler()
        result = scheduler.submit_checkpoint(2, self.checkpoint(), 2)
        self.assertEqual(result['status'], 'submitted')
        self.assertEqual(result['job_id'], '12345')
        command = self.submit.call_args.args[0]
        kwargs = self.submit.call_args.kwargs
        self.assertIn('--gres=gpu:1', command)
        self.assertIn('--export=ALL', command)
        self.assertNotIn('--wait', command)
        self.assertEqual(kwargs['timeout'], 30)
        self.assertFalse(kwargs['check'])
        self.assertNotIn('shell', kwargs)
        script = kwargs['input']
        evaluation = shlex.split(script.split('exec ', 1)[1].strip())
        self.assertEqual(evaluation[:3], ['/test env/bin/python', '-u', str(self.source.resolve() / 'evaluate_async.py')])
        for flag, value in [('--seed', '42'), ('--rollouts', '8'), ('--training-step', '2'),
                            ('--epoch', '1'), ('--wandb-project', 'stride-iclr2027')]:
            self.assertEqual(evaluation[evaluation.index(flag) + 1], value)
        self.assertIn('--checkpoint', evaluation)
        self.assertNotIn('--adapter', evaluation)
        parsed = evaluate_async.parse_args(evaluation[3:])
        self.assertEqual(parsed.training_step, 2)
        self.assertEqual(parsed.epoch, 1)
        self.assertEqual(parsed.wandb_project, 'stride-iclr2027')
        self.assertFalse(Path(result['output']).exists())
        self.assertFalse(scheduler.summary()['completion_checked'])

    def test_lora_uses_adapter_option_and_same_validation_budget(self):
        self.args.finetune_mode = 'lora'
        self.args.sampler_eager = True
        scheduler = self.scheduler()
        self.assertEqual(scheduler.submit_checkpoint(2, self.checkpoint(), 2)['status'], 'submitted')
        command = shlex.split(self.submit.call_args.kwargs['input'].split('exec ', 1)[1])
        self.assertIn('--adapter', command)
        self.assertNotIn('--checkpoint', command)
        self.assertIn('--sampler-eager', command)
        self.assertEqual(command[command.index('--rollouts') + 1], '8')
        self.assertIsNotNone(evaluate_async.parse_args(command[3:]).adapter)

    def test_submission_inherits_auth_only_in_memory_and_drops_training_allocation(self):
        original = {'WANDB_API_KEY': 'not-a-real-secret', 'WANDB_PROJECT': 'project', 'PATH': '/bin',
                    'CUDA_VISIBLE_DEVICES': '0,1,2,3,4,5,6,7', 'SLURM_JOB_ID': '99',
                    'RANK': '0', 'WORLD_SIZE': '4', 'TORCHELASTIC_RUN_ID': 'abc', 'SBATCH_GRES': 'gpu:8'}
        cleaned = jobs.submission_environment(original)
        self.assertEqual(cleaned, {k: original[k] for k in ('WANDB_API_KEY', 'WANDB_PROJECT', 'PATH')})
        scheduler = self.scheduler()
        with patch.dict(os.environ, original, clear=True):
            scheduler.submit_checkpoint(2, self.checkpoint(), 2)
        kwargs = self.submit.call_args.kwargs
        self.assertEqual(kwargs['env']['WANDB_API_KEY'], 'not-a-real-secret')
        self.assertNotIn('not-a-real-secret', kwargs['input'])
        for file in Path(self.args.out).rglob('*'):
            if file.is_file():
                self.assertNotIn('not-a-real-secret', file.read_text())

    def test_script_shell_quotes_group_and_model_paths(self):
        self.args.wandb_group = "study; $(never execute) ' quotation"
        scheduler = self.scheduler()
        scheduler.submit_checkpoint(2, self.checkpoint(), 2)
        command = shlex.split(self.submit.call_args.kwargs['input'].split('exec ', 1)[1])
        self.assertEqual(command[command.index('--wandb-group') + 1], self.args.wandb_group)
        self.assertEqual(command[command.index('--model-path') + 1], str(Path(self.args.model_path).resolve()))

    def test_submits_only_requested_completed_epochs_and_prevents_duplicates(self):
        self.args.validation_every_epochs = 2
        scheduler = self.scheduler()
        self.assertIsNone(scheduler.submit_checkpoint(1, '/missing', 2))
        self.assertIsNone(scheduler.submit_checkpoint(2, '/missing', 2))
        checkpoint = self.checkpoint(4)
        self.assertEqual(scheduler.submit_checkpoint(4, checkpoint, 2)['status'], 'submitted')
        self.assertEqual(scheduler.submit_checkpoint(4, checkpoint, 2)['status'], 'submission_failed')
        self.assertEqual(self.submit.call_count, 1)

    def test_scheduler_rejection_never_leaks_output_or_stops_training(self):
        scheduler = self.scheduler()
        self.submit.return_value = SimpleNamespace(returncode=1, stdout='secret-output', stderr='secret-api-key')
        result = scheduler.submit_checkpoint(2, self.checkpoint(), 2)
        self.assertEqual(result['status'], 'submission_failed')
        self.assertNotIn('secret', json.dumps(result))
        self.assertEqual(scheduler.summary()['submission_failures'], 1)
        self.assertTrue((Path(self.args.out) / 'validation_jobs/epoch_001/submission_failed.json').is_file())

    def test_submission_timeout_and_ambiguous_job_id_never_retry(self):
        for outcome in [subprocess.TimeoutExpired('sbatch', 30),
                        SimpleNamespace(returncode=0, stdout='not a job ID', stderr='')]:
            with self.subTest(outcome=type(outcome).__name__):
                self.args.out = str(self.root / type(outcome).__name__)
                scheduler = self.scheduler()
                self.submit.reset_mock(side_effect=True)
                if isinstance(outcome, Exception):
                    self.submit.side_effect = outcome
                else:
                    self.submit.return_value = outcome
                result = scheduler.submit_checkpoint(2, self.checkpoint(), 2)
                self.assertEqual(result['status'], 'submission_failed')
                self.assertEqual(self.submit.call_count, 1)

    def test_incomplete_checkpoint_and_changed_inputs_are_not_submitted(self):
        scheduler = self.scheduler()
        checkpoint = self.checkpoint()
        checkpoint.with_name(checkpoint.name + '_manifest.json').unlink()
        self.assertEqual(scheduler.submit_checkpoint(2, checkpoint, 2)['status'], 'submission_failed')
        checkpoint = self.checkpoint(4)
        Path(scheduler.metadata['data']).write_text('changed')
        self.assertEqual(scheduler.submit_checkpoint(4, checkpoint, 2)['status'], 'submission_failed')
        self.submit.assert_not_called()

    def test_symlink_checkpoint_and_wrong_step_are_rejected(self):
        scheduler = self.scheduler()
        checkpoint = self.checkpoint(4)
        link = Path(self.args.out) / 'model_002'
        link.symlink_to(checkpoint, target_is_directory=True)
        self.assertEqual(scheduler.submit_checkpoint(2, link, 2)['status'], 'submission_failed')
        self.assertEqual(scheduler.submit_checkpoint(6, checkpoint, 2)['status'], 'submission_failed')
        self.submit.assert_not_called()


if __name__ == '__main__':
    unittest.main()
