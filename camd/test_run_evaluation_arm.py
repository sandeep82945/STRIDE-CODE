"""Validate the single-arm controller's scope and fail-closed stage ordering."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import run_evaluation_arm as controller


class EvaluationArmTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        (self.root / 'results').mkdir()
        (self.root / 'checkpoints').mkdir()
        self.submission = {'complete': True, 'training_runs': {
            arm: str(self.root / 'checkpoints' / f'{arm}-train-{number}')
            for number, arm in enumerate(controller.ARMS, 1)}}
        for run in self.submission['training_runs'].values():
            Path(run).mkdir()
        (self.root / 'results/submission.json').write_text(json.dumps(self.submission))

    def test_one_arm_uses_its_own_final_checkpoint(self):
        for arm in controller.ARMS:
            with self.subTest(arm=arm):
                command = controller.evaluation_command(arm, self.submission, self.root)
                self.assertEqual(command[-7:], ['eval', '--arm', arm, '--run-dir',
                                               self.submission['training_runs'][arm], '--epoch', '4'])
                self.assertIn('--gres=gpu:1', command)
        base = controller.evaluation_command('base', self.submission, self.root)
        self.assertEqual(base[-3:], ['eval', '--arm', 'base'])
        self.assertNotIn('--run-dir', base)

    def test_incomplete_manifest_and_unknown_arm_are_rejected(self):
        invalid = ({'complete': False, 'training_runs': self.submission['training_runs']},
                   {'complete': True, 'training_runs': {}},
                   {'complete': True, 'training_runs': []})
        for submission in invalid:
            with self.subTest(submission=submission), self.assertRaises(ValueError):
                controller.evaluation_command('grpo', submission, self.root)
        with self.assertRaises(ValueError):
            controller.evaluation_command('invented', self.submission, self.root)

    def test_foreign_run_and_symlink_are_rejected(self):
        original = self.submission['training_runs']['grpo']
        outside = self.root / 'elsewhere'
        outside.mkdir()
        link = self.root / 'checkpoints/grpo-train-99'
        link.symlink_to(outside, target_is_directory=True)
        for run in (outside, self.root / 'checkpoints/single_step-train-2',
                    self.root / 'checkpoints/grpo-train-arbitrary', link):
            with self.subTest(run=run), self.assertRaises(ValueError):
                self.submission['training_runs']['grpo'] = str(run)
                controller.evaluation_command('grpo', self.submission, self.root)
        self.submission['training_runs']['grpo'] = original

    def run_main(self, runner):
        with patch.object(controller, 'ROOT', self.root), patch.dict(controller.os.environ, {'SLURM_JOB_ID': '123'}), \
                patch.object(controller.subprocess, 'run', runner), contextlib.redirect_stdout(io.StringIO()) as output:
            controller.main(['grpo'])
        return output.getvalue()

    def test_gate_runs_only_after_successful_stage(self):
        from unittest.mock import Mock
        runner = Mock()
        output = self.run_main(runner)
        self.assertEqual(runner.call_count, 2)
        self.assertEqual(runner.call_args_list[0].args[0][-7:],
                         ['eval', '--arm', 'grpo', '--run-dir', self.submission['training_runs']['grpo'], '--epoch', '4'])
        self.assertEqual(runner.call_args_list[1].args[0][-3:], ['123', 'eval', 'grpo'])
        self.assertIn('EVALUATION_ARM_VERIFIED', output)

    def test_stage_failure_prevents_gate(self):
        from unittest.mock import Mock
        runner = Mock(side_effect=subprocess.CalledProcessError(1, ['srun']))
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_main(runner)
        self.assertEqual(runner.call_count, 1)

    def test_host_gate_failure_propagates(self):
        from unittest.mock import Mock
        runner = Mock(side_effect=[None, subprocess.CalledProcessError(1, ['host_gate.py'])])
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_main(runner)
        self.assertEqual(runner.call_count, 2)


if __name__ == '__main__':
    unittest.main()
