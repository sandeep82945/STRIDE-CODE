import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import run


class LauncherTests(unittest.TestCase):
    def test_keep_all_checkpoints_is_forwarded_to_resumable_trainer(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / 'data.json'
            data.write_text('[]')
            command = run.build_command('train', ['--model-path', '/model', '--data', str(data),
                                                 '--out', '/fresh', '--keep-all-checkpoints'])
            self.assertEqual(command.count('--keep-all-checkpoints'), 1)
            self.assertTrue(any(item.endswith('/trainer_resumable.py') for item in command))

    def test_seed_limits_defaults_and_duplicate_validation(self):
        for stage in ('train', 'smoke', 'eval'):
            self.assertEqual(run.experiment_seed(stage, []), 42)
            self.assertEqual(run.experiment_seed(stage, ['--seed=43']), 43)
            self.assertEqual(run.experiment_seed(stage, ['--seed', '0']), 0)
            maximum = 2**63 - 1 if stage == 'eval' else 2**32 - 1
            self.assertEqual(run.experiment_seed(stage, ['--seed', str(maximum)]), maximum)
            for value in ('-1', str(maximum + 1), 'random', ''):
                with self.subTest(stage=stage, value=value), self.assertRaises(ValueError):
                    run.experiment_seed(stage, ['--seed=' + value])
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                run.experiment_seed(stage, ['--seed', '42', '--seed=43'])

    def test_hash_seed_is_applied_before_python_process_start(self):
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES='GPU-fixture', PYTHONHASHSEED='random'):
            env = run.execution_environment('eval', seed=2**40 + 43)
        self.assertEqual(env['PYTHONHASHSEED'], '43')
        command = [sys.executable, '-c',
                   'import os; print(os.environ["PYTHONHASHSEED"], hash("STRIDE shared question order"))']
        first = subprocess.check_output(command, env=env, text=True)
        second = subprocess.check_output(command, env=env, text=True)
        self.assertEqual(first, second)
        self.assertTrue(first.startswith('43 '))

    def test_exact_data_hash_and_argument_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / 'data with spaces.json'
            data.write_text('[]')
            arguments = ['--model-path', '/model with spaces', '--data', str(data), '--out', '/new run']
            command = run.build_command('train', arguments, python='/python with spaces')
            self.assertEqual(command[0], '/python with spaces')
            self.assertIn('/model with spaces', command)
            self.assertIn('--nproc_per_node=4', command)
            self.assertEqual(run.option(command, '--seed'), '42')
            self.assertEqual(command[-1], hashlib.sha256(b'[]').hexdigest())
            self.assertNotIn('--data-sha256', arguments)

    def test_mismatched_data_hash_rejected_before_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / 'data.json'
            data.write_text('[]')
            with self.assertRaises(ValueError):
                run.build_command('train', ['--data', str(data), '--model-path', '/m', '--data-sha256', '0'*64])

    def test_scheduler_visibility_not_reassigned(self):
        visible = ','.join(f'GPU-{i}' for i in range(8))
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': visible, 'SLURM_JOB_ID': 'test'}, clear=True):
            self.assertEqual(run.execution_environment('train')['CUDA_VISIBLE_DEVICES'], visible)
            with self.assertRaises(ValueError):
                run.execution_environment('eval')

    def test_missing_scheduler_visibility_is_not_guessed(self):
        with patch.dict(os.environ, {'SLURM_JOB_ID': 'test'}, clear=True), self.assertRaises(ValueError):
            run.execution_environment('train')

    def test_duplicate_option_rejected(self):
        with self.assertRaises(ValueError):
            run.option(['--data', 'a', '--data=b'], '--data')

    def test_source_snapshot_tracks_edits_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, cache = Path(tmp)/'source', Path(tmp)/'cache'
            source.mkdir()
            (source/'a.py').write_text('x = 1\n')
            first = run.freeze_source(source, cache)
            self.assertEqual(first, run.freeze_source(source, cache))
            (source/'a.py').write_text('x = 2\n')
            second = run.freeze_source(source, cache)
            self.assertNotEqual(first, second)
            self.assertEqual((first/'a.py').read_text(), 'x = 1\n')
            (second/'a.py').write_text('x = 99\n')
            with self.assertRaises(ValueError):
                run.freeze_source(source, cache)


if __name__ == '__main__':
    unittest.main()
