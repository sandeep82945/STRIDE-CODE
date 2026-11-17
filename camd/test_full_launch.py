"""Full launch orchestration through real argument parsers; no Slurm/GPU/network."""
import argparse
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

CAMD = Path(__file__).resolve().parent
CODE = CAMD.parent
sys.path[:0] = [str(CAMD), str(CODE), str(CODE / 'training')]
import full_stage as stage
import submit_full as submit
import runtime_config


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class StageParserTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        write(self.root / 'data/train.json', [dict(id='fixture', question='What is 1 + 1?', ground_truth=2)])
        self.env_file = self.root / 'config/wandb.env'
        self.env_file.parent.mkdir()
        self.env_file.write_text('WANDB_MODE=offline\n')
        self.environment = dict(SLURM_JOB_ID='999', SLURM_STEP_ID='0', SLURMD_NODENAME='fixture-node',
            STRIDE_RUNTIME='rocm', ROCR_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',
            HIP_VISIBLE_DEVICES='0,1,2,3,4,5,6,7', CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7')

    def exercise(self, kind, arm='local_positive', microbatch=2):
        root = self.root / f'{kind}-{arm}-{microbatch}'
        root.mkdir()
        (root / 'data').mkdir()
        shutil.copyfile(self.root / 'data/train.json', root / 'data/train.json')
        report_dir = root / 'results/job-999' / f'{kind}-{arm}'
        if kind == 'train':
            write(root / 'results/job-999' / f'smoke-{arm}/host_verification.json', {'passed': True})
        captured = []
        publisher = Mock(returncode=0)
        publisher.wait.return_value = 0

        def invocation(command, env):
            command = [str(value) for value in command]
            self.assertEqual(env['ROCR_VISIBLE_DEVICES'], self.environment['ROCR_VISIBLE_DEVICES'])
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '0,1,2,3,4,5,6,7')
            # Use actual run.py argument resolution in dry-run mode, so unsupported
            # flags cannot disappear in a mocked launcher or a hand-built parser.
            output = io.StringIO()
            with patch('sys.argv', [command[1], *command[2:], '--dry-run']), redirect_stdout(output):
                self.assertEqual(stage.launcher.main(), 0)
            planned = json.loads(output.getvalue())
            actual = planned['command']
            trainer_index = next(i for i, value in enumerate(actual) if value.endswith(
                'trainer_resumable.py' if kind == 'train' else 'trainer_distributed.py'))
            parser = argparse.ArgumentParser()
            runtime_config.add_training_arguments(parser, resumable=kind == 'train')
            args = parser.parse_args(actual[trainer_index + 1:])
            runtime_config.validate_options(args, resumable=kind == 'train')
            captured.append(args)
            write(Path(args.out) / 'done.json', {'complete': True})

        argv = ['full_stage.py', kind, '--arm', arm]
        if kind == 'train':
            argv += ['--smoke-output', str(root / 'checkpoints' / f'{arm}-smoke-999')]
        with patch.object(stage, 'ROOT', root), patch.object(stage, 'version', return_value='fixture'), \
             patch.object(stage, 'production_evidence', return_value={'files_checked': 1}), \
             patch.object(stage, 'cpu_evidence', return_value={'passed': True}), \
             patch.object(stage, 'dependencies'), patch.object(stage, 'preflight') as preflight, \
             patch.object(stage, 'tracking', return_value=(self.env_file, 'disabled' if kind == 'smoke' else 'offline', 'fixture')), \
             patch.object(stage, 'verify', return_value={'passed': True, 'arm': arm}) as smoke_gate, \
             patch.object(stage, 'select_microbatch', return_value=microbatch), \
             patch.object(stage, 'start_publisher', return_value=(publisher, report_dir / 'publisher_stop.json', 'fixture/repo')) as start_publisher, \
             patch.object(stage, 'invoke', side_effect=invocation), patch.dict(os.environ, self.environment, clear=True), \
             patch('sys.argv', argv), redirect_stdout(io.StringIO()):
            stage.main()
        self.assertEqual(len(captured), 1)
        args = captured[0]
        self.assertEqual((args.mode, args.arm, args.epochs, args.batch_size), (kind, arm, 4, 64))
        self.assertEqual((args.finetune_mode, args.learning_rate, args.validation_mode), ('lora', .00002, 'off'))
        self.assertEqual(args.warmup_steps, 10)
        self.assertEqual(args.kl_coefficient, .01)
        self.assertEqual(args.model, stage.MODEL_KEY)
        self.assertEqual(args.alpha, stage.ALPHA)
        self.assertIs(args.thinking_mode, False)
        self.assertEqual(args.sampler_max_num_seqs, 32)
        self.assertEqual(args.seed, 42)
        report = json.loads((report_dir / 'run.json').read_text())
        self.assertEqual(report['status'], 'FINISHED')
        if kind == 'smoke':
            self.assertEqual((args.rollouts, args.smoke_cap, args.sampler_eager), (2, 128, True))
            self.assertFalse(hasattr(args, 'keep_all_checkpoints'))
            preflight.assert_called_once()
            start_publisher.assert_not_called()
        else:
            self.assertEqual(args.rollouts, 8)
            self.assertFalse(args.sampler_eager)
            self.assertTrue(args.keep_all_checkpoints)
            self.assertEqual(args.resume_state_retention, "latest-epoch")
            self.assertFalse(args.allow_epoch_extension)
            self.assertEqual(args.learner_microbatch_size, microbatch)
            self.assertEqual(runtime_config.schedule_steps(args, 2048), 128)
            smoke_gate.assert_called_once()
            start_publisher.assert_called_once()
            self.assertFalse(report['publisher_needs_retry'])

    def test_actual_smoke_parser_accepts_all_four_methods(self):
        for arm in stage.ARMS:
            with self.subTest(arm=arm):
                self.exercise('smoke', arm)

    def test_actual_training_parser_accepts_all_four_methods_with_retention_and_pairs(self):
        for arm in stage.ARMS:
            with self.subTest(arm=arm):
                self.exercise('train', arm, 2)

    def test_conservative_microbatch_one_uses_identical_full_data_schedule(self):
        self.exercise('train', 'local_positive', 1)


class PublisherStartupTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.report_dir = self.root / 'results/job-999/train-grpo'
        self.report_dir.mkdir(parents=True)
        self.output = self.root / 'checkpoints/grpo-train-999'
        self.repo = 'fixture/public-grpo'
        write(self.root / 'results/hf_repositories.json', dict(repos={'grpo': self.repo}))
        self.environment = dict(SLURM_JOB_ID='999', ROCR_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',
            CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7', HIP_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',
            HF_TOKEN='inert-inherited-test-value', OPENAI_API_KEY='inert-inherited-test-value',
            WANDB_API_KEY='inert-inherited-test-value', HUGGING_FACE_HUB_TOKEN='inert-inherited-test-value')

    def test_publisher_must_report_matching_ready_repository_before_training(self):
        worker = Mock()
        worker.poll.return_value = None

        def popen(command, **kwargs):
            env = kwargs['env']
            self.assertEqual([env[name] for name in ('ROCR_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES',
                                                    'CUDA_VISIBLE_DEVICES')], ['-1', '-1', '-1'])
            for name in ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'OPENAI_API_KEY', 'WANDB_API_KEY'):
                self.assertNotIn(name, env)
            self.assertEqual(command[command.index('--arm') + 1], 'grpo')
            self.assertEqual(command[command.index('--repo-id') + 1], self.repo)
            self.assertEqual(command[command.index('--token-file') + 1], str(self.root / 'config/huggingface.env'))
            state_dir = Path(command[command.index('--state-dir') + 1])
            write(state_dir / 'status.json', dict(repository=self.repo, errors=[], stop_requested=False))
            return worker

        with patch.object(stage, 'ROOT', self.root), patch.dict(os.environ, self.environment, clear=True), \
             patch.object(stage.subprocess, 'Popen', side_effect=popen) as launch:
            result = stage.start_publisher('grpo', self.output, self.report_dir)
        self.assertEqual(result, (worker, self.report_dir / 'publisher_stop.json', self.repo))
        launch.assert_called_once()
        worker.terminate.assert_not_called()
        self.assertFalse(self.output.exists(), 'Publisher must be ready before the trainer creates its output')

    def test_early_publisher_exit_aborts_startup_and_reaps_process(self):
        worker = Mock()
        worker.poll.return_value = 2
        worker.wait.return_value = 2
        with patch.object(stage, 'ROOT', self.root), patch.dict(os.environ, self.environment, clear=True), \
             patch.object(stage.subprocess, 'Popen', return_value=worker), \
             self.assertRaisesRegex(RuntimeError, 'exited during startup'):
            stage.start_publisher('grpo', self.output, self.report_dir)
        worker.terminate.assert_called_once()
        worker.wait.assert_called_once_with(timeout=15)
        self.assertFalse(self.output.exists())


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.code = self.root / 'code/stride_handoff'
        self.code.mkdir(parents=True)
        (self.code / 'run.py').write_text('# inert tested source fixture\n')
        (self.code / 'camd').mkdir()
        shutil.copyfile(CAMD / 'full_job.sbatch', self.code / 'camd/full_job.sbatch')
        write(self.root / 'results/cpu_verified.json', dict(passed=True, source_stable=True, skipped=[],
            tests_run=454, source_sha256={'run.py': digest(self.code / 'run.py')}))
        relative = 'code/stride_handoff/camd/full_job.sbatch'
        write(self.root / 'results/production_manifest.json', dict(files={relative: digest(self.root / relative)}))
        write(self.root / 'results/hf_repositories.json', dict(verified_public=True,
            repos={arm: f'fixture/{arm}' for arm in submit.ARMS}))

    def invoke(self, effects, reference_error=None):
        with patch.object(submit, 'ROOT', self.root), patch.object(submit, 'CODE', self.code), \
             patch.object(submit.subprocess, 'check_output', side_effect=effects) as sbatch, \
             patch.object(submit, 'verify_credit_reference', return_value={'passed': True}, side_effect=reference_error), \
             patch('sys.argv', ['submit_full.py', '--submit']), redirect_stdout(io.StringIO()):
            submit.main()
        return sbatch

    def test_exact_family_independent_jobs_time_limit_and_no_base(self):
        sbatch = self.invoke([str(1000 + i) + '\n' for i in range(len(submit.ARMS))])
        self.assertEqual(sbatch.call_count, len(submit.ARMS))
        text = (self.code / 'camd/full_job.sbatch').read_text()
        for directive in (f'#SBATCH --time={submit.TIME_LIMIT}', '#SBATCH --gres=gpu:8',
                          f'#SBATCH --qos={submit.QOS}', '#SBATCH --account=faculty-acc'):
            self.assertIn(directive, text)
        self.assertNotIn('--dependency', text)
        record = json.loads((self.root / 'results/submission.json').read_text())
        self.assertTrue(record['complete'])
        self.assertFalse(record['base_job'])
        self.assertEqual(record['time_limit'], submit.TIME_LIMIT)
        self.assertEqual(record['qos'], submit.QOS)
        self.assertEqual(record['model'], submit.MODEL_KEY)
        self.assertEqual(record['alpha'], submit.ALPHA)
        self.assertIs(record['thinking_mode'], False)
        self.assertEqual(record['updates'], 128)
        self.assertEqual(set(record['training_runs']), set(submit.ARMS))
        self.assertEqual([job['arm'] for job in record['jobs']], submit.ARMS)
        for call, job in zip(sbatch.call_args_list, record['jobs']):
            self.assertEqual(job['dependencies'], [])
            self.assertEqual(call.args[0][-1], job['arm'])
            self.assertFalse(any(arg.startswith('--dependency') for arg in call.args[0]))

    def test_submission_receipt_blocks_accidental_duplicate_jobs(self):
        self.invoke([str(1000+i)+'\n' for i in range(len(submit.ARMS))])
        record = json.loads((self.root / 'results/submission.json').read_text())
        self.assertTrue(record['complete'])
        self.assertEqual(len(record['jobs']), len(submit.ARMS))
        with self.assertRaises(FileExistsError):
            self.invoke([AssertionError('Must not submit duplicate work')])

    def test_partial_family_receipt_blocks_duplicate_submission(self):
        completed = max(0, len(submit.ARMS) - 1)
        effects = [str(1000 + i) + '\n' for i in range(completed)]
        effects.append(subprocess.CalledProcessError(1, 'inert sbatch fixture'))
        with self.assertRaises(subprocess.CalledProcessError):
            self.invoke(effects)
        record = json.loads((self.root / 'results/submission.json').read_text())
        self.assertFalse(record['complete'])
        self.assertEqual([item['arm'] for item in record['jobs']], submit.ARMS[:completed])
        self.assertEqual(record['pending_submission']['arm'], submit.ARMS[completed])
        with self.assertRaises(FileExistsError):
            self.invoke([AssertionError('Partial family must never be resubmitted blindly')])

    def test_invalid_sbatch_output_leaves_uncertain_intent_and_blocks_retry(self):
        with self.assertRaisesRegex(ValueError, 'Unexpected sbatch response'):
            self.invoke(['unexpected scheduler output\n'])
        record = json.loads((self.root / 'results/submission.json').read_text())
        self.assertFalse(record['complete'])
        self.assertEqual(record['pending_submission']['arm'], submit.ARMS[0])
        with self.assertRaises(FileExistsError):
            self.invoke([AssertionError('Unknown scheduler state must not duplicate work')])

    def test_failed_submission_retains_uncertain_receipt_for_manual_recovery(self):
        with self.assertRaises(subprocess.CalledProcessError):
            self.invoke([subprocess.CalledProcessError(1, 'inert sbatch fixture')])
        record = json.loads((self.root / 'results/submission.json').read_text())
        self.assertFalse(record['complete'])
        self.assertEqual(record['jobs'], [])
        with self.assertRaises(FileExistsError):
            self.invoke([AssertionError('Must not retry an uncertain submission')])

    def test_changed_frozen_job_script_is_rejected_before_any_submission(self):
        with (self.code / 'camd/full_job.sbatch').open('a') as stream:
            stream.write('\n# changed after freezing\n')
        with self.assertRaisesRegex(ValueError, 'Frozen production file changed'):
            self.invoke([AssertionError('Must not submit unverified work')])


    def test_stale_credit_reference_stops_before_sbatch_and_receipt_creation(self):
        with self.assertRaisesRegex(ValueError, 'Stale CPU credit-reference'):
            self.invoke([AssertionError('Must not submit with a stale numerical reference')],
                        reference_error=ValueError('Stale CPU credit-reference source hashes'))
        self.assertFalse((self.root / 'results/submission.json').exists())


    def test_submission_help_runs_with_isolated_host_python_without_site_packages(self):
        result = subprocess.run([sys.executable, '-I', '-S', str(CAMD / 'submit_full.py'), '--help'],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--submit', result.stdout)


if __name__ == '__main__':
    unittest.main()
