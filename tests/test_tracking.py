"""CPU-only observer lifecycle tests with fake SDK, collectives and scheduler."""
from __future__ import annotations

from contextlib import redirect_stdout
import builtins
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training'))
import tracking
import validation_jobs


class TrackingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.out = Path(temporary.name)
        environment = patch.dict(os.environ, {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)

    @staticmethod
    def args(**overrides):
        values = dict(wandb_mode='online', wandb_project='stride-test', wandb_entity='test-team',
                      wandb_group='controlled-comparison', model='qwen4b', finetune_mode='full',
                      mode='train', arm='local_positive', seed=42, rollouts=8)
        values.update(overrides)
        return SimpleNamespace(**values)

    @staticmethod
    def fake_sdk():
        run = MagicMock()
        run.id, run.project, run.entity, run.url = 'eval-test-id', 'stride-test', 'test-team', 'https://example.invalid/test'
        run.summary = {}
        return SimpleNamespace(init=MagicMock(return_value=run)), run

    @staticmethod
    def fake_dist(rank=0, received=None):
        dist = MagicMock()
        dist.get_rank.return_value = rank
        dist.get_world_size.return_value = 4
        if received is not None:
            def receive(packet, src):
                packet[0] = received
            dist.broadcast_object_list.side_effect = receive
        return dist

    def start(self, args=None):
        return tracking.start_tracking(args or self.args(), self.out,
                                       world_size=4, training_questions=2048, updates=128)

    def test_disabled_does_not_import_sdk_or_require_key(self):
        real_import = builtins.__import__
        def guarded_import(name, *args, **kwargs):
            if name == 'wandb':
                self.fail('Disabled tracking imported W&B')
            return real_import(name, *args, **kwargs)
        with patch('builtins.__import__', side_effect=guarded_import):
            self.assertIsNone(self.start(self.args(wandb_mode='disabled')))
        self.assertFalse((self.out / 'wandb.json').exists())

    def test_online_missing_or_blank_key_fails_before_sdk(self):
        sdk, _ = self.fake_sdk()
        with patch.dict(sys.modules, {'wandb': sdk}):
            for key in ('', '   '):
                with self.subTest(key=key), patch.dict(os.environ, WANDB_API_KEY=key), \
                     self.assertRaisesRegex(ValueError, 'Set WANDB_API_KEY'):
                    self.start()
        sdk.init.assert_not_called()

    def test_online_initialization_logs_metadata_without_credentials_or_source(self):
        sdk, run = self.fake_sdk()
        with patch.dict(sys.modules, {'wandb': sdk}), patch.dict(os.environ, {
                'WANDB_API_KEY': 'test_secret_987', 'WANDB_RUN_ID': 'training-id-inherited',
                'WANDB_RESUME': 'must'}):
            actual = self.start()
            self.assertEqual(os.environ['WANDB_DISABLE_CODE'], 'true')
            self.assertEqual(os.environ['WANDB_CONSOLE'], 'off')
        self.assertIs(actual, run)
        options = sdk.init.call_args.kwargs
        self.assertEqual(options['project'], 'stride-test')
        self.assertEqual(options['group'], 'controlled-comparison')
        self.assertEqual(options['name'], 'qwen4b-full-train-local_positive-seed42')
        self.assertEqual(options['resume'], 'never')
        self.assertRegex(options['id'], r'^[0-9a-f]{32}$')
        self.assertNotEqual(options['id'], 'training-id-inherited')
        self.assertFalse(options['save_code'])
        self.assertNotIn('test_secret_987', json.dumps(options['config']))
        self.assertNotIn('WANDB_API_KEY', options['config'])
        self.assertEqual(options['config']['training_questions'], 2048)
        self.assertEqual(options['config']['updates'], 128)
        self.assertEqual(options['config']['world_size'], 4)
        self.assertEqual(options['config']['K'], 8)
        self.assertEqual(options['config']['context_limit'], 8192)
        self.assertEqual(options['config']['loss_normalization'], 'global_real_response_tokens')
        saved = json.loads((self.out / 'wandb.json').read_text())
        self.assertEqual(saved, dict(id=run.id, project=run.project, entity=run.entity,
                                    url=run.url, mode='online'))

    def test_offline_does_not_require_a_key_and_each_start_has_a_fresh_id(self):
        sdk, _ = self.fake_sdk()
        with patch.dict(sys.modules, {'wandb': sdk}):
            self.start(self.args(wandb_mode='offline'))
            self.start(self.args(wandb_mode='offline'))
        ids = [call.kwargs['id'] for call in sdk.init.call_args_list]
        self.assertEqual(len(ids), 2)
        self.assertNotEqual(ids[0], ids[1])

    def test_rejects_key_embedded_in_arbitrary_run_metadata(self):
        sdk, _ = self.fake_sdk()
        with patch.dict(sys.modules, {'wandb': sdk}), patch.dict(os.environ, WANDB_API_KEY='test_secret_987'), \
             self.assertRaisesRegex(ValueError, 'must not appear'):
            self.start(self.args(wandb_group='accidental-test_secret_987-group'))
        sdk.init.assert_not_called()

    def test_sdk_initialization_failure_is_sanitized(self):
        sdk, _ = self.fake_sdk()
        sdk.init.side_effect = RuntimeError('request rejected token=test_secret_987')
        with patch.dict(sys.modules, {'wandb': sdk}), patch.dict(os.environ, WANDB_API_KEY='test_secret_987'), \
             self.assertRaisesRegex(RuntimeError, 'W&B initialization failed') as caught:
            self.start()
        self.assertNotIn('test_secret_987', str(caught.exception))
        self.assertTrue(caught.exception.__suppress_context__)
        self.assertFalse((self.out / 'wandb.json').exists())

    def test_rank_zero_sets_up_plan_scheduler_and_tracking_once(self):
        args, rows = self.args(), [{'id': 'train:1'}, {'id': 'train:2'}]
        dist = self.fake_dist()
        plan = {'mode': 'slurm', 'questions': 128}
        scheduler, run = object(), object()
        with patch.object(validation_jobs, 'prepare_validation', return_value=plan) as prepare, \
             patch.object(validation_jobs, 'ValidationScheduler', return_value=scheduler) as construct, \
             patch.object(tracking, 'start_tracking', return_value=run) as start:
            result = tracking.setup_observers(args, rows, self.out, dist, updates=128)
        self.assertEqual(result, (run, scheduler))
        prepare.assert_called_once_with(args, rows)
        construct.assert_called_once_with(args, plan)
        start.assert_called_once_with(args, self.out, world_size=4, training_questions=2, updates=128)
        self.assertEqual(json.loads((self.out / 'validation_plan.json').read_text()), plan)
        dist.broadcast_object_list.assert_called_once_with([{'ok': True}], src=0)

    def test_sdk_returning_none_is_a_sanitized_startup_failure(self):
        sdk, _ = self.fake_sdk()
        sdk.init.return_value = None
        with patch.dict(sys.modules, {'wandb': sdk}), \
             self.assertRaisesRegex(RuntimeError, 'W&B initialization failed'):
            self.start(self.args(wandb_mode='offline'))

    def test_metadata_write_failure_closes_created_run_as_failed(self):
        sdk, run = self.fake_sdk()
        with patch.dict(sys.modules, {'wandb': sdk}), \
             patch('sampler_worker.atomic_json', side_effect=OSError('disk full')), \
             self.assertRaisesRegex(RuntimeError, 'W&B initialization failed'):
            self.start(self.args(wandb_mode='offline'))
        run.finish.assert_called_once_with(exit_code=1)

    def test_nonzero_rank_receives_success_without_sdk_or_scheduler_work(self):
        dist = self.fake_dist(rank=2, received={'ok': True})
        with patch.object(validation_jobs, 'prepare_validation') as prepare, \
             patch.object(validation_jobs, 'ValidationScheduler') as construct, \
             patch.object(tracking, 'start_tracking') as start:
            result = tracking.setup_observers(self.args(), [], self.out, dist, updates=128)
        self.assertEqual(result, (None, None))
        prepare.assert_not_called()
        construct.assert_not_called()
        start.assert_not_called()
        self.assertFalse((self.out / 'validation_plan.json').exists())
        dist.broadcast_object_list.assert_called_once()

    def test_rank_zero_sanitizes_setup_error_and_broadcasts_failure(self):
        dist = self.fake_dist()
        with patch.dict(os.environ, WANDB_API_KEY='test_secret_987'), \
             patch.object(validation_jobs, 'prepare_validation', side_effect=ValueError('bad test_secret_987 access')), \
             patch.object(tracking, 'start_tracking') as start, \
             self.assertRaisesRegex(RuntimeError, 'Validation/tracking setup failed') as caught:
            tracking.setup_observers(self.args(), [], self.out, dist, updates=128)
        start.assert_not_called()
        self.assertNotIn('test_secret_987', str(caught.exception))
        status = dist.broadcast_object_list.call_args.args[0][0]
        self.assertEqual(status, {'ok': False, 'error': 'bad [redacted] access'})

    def test_nonzero_rank_raises_broadcast_setup_failure(self):
        dist = self.fake_dist(rank=1, received={'ok': False, 'error': 'rank zero setup failed'})
        with patch.object(tracking, 'start_tracking') as start, \
             self.assertRaisesRegex(RuntimeError, 'rank zero setup failed'):
            tracking.setup_observers(self.args(), [], self.out, dist, updates=128)
        start.assert_not_called()

    def test_submit_validation_records_submission_status_without_claiming_completion(self):
        for status, submitted, failures in [('submitted', 2, 0), ('submission_failed', 1, 1)]:
            with self.subTest(status=status):
                scheduler, run = MagicMock(), SimpleNamespace(summary={})
                record = dict(status=status, epoch=2, training_step=64)
                scheduler.submit_checkpoint.return_value = record
                scheduler.summary.return_value = dict(submitted=submitted, submission_failures=failures,
                                                       completion_checked=False)
                output = io.StringIO()
                with redirect_stdout(output):
                    actual = tracking.submit_validation(scheduler, run, completed_step=64,
                                                         checkpoint='/pinned/model_064', per_epoch=32)
                self.assertIs(actual, record)
                scheduler.submit_checkpoint.assert_called_once_with(64, '/pinned/model_064', 32)
                self.assertEqual(json.loads(output.getvalue()), {'validation_submission': record})
                self.assertEqual(run.summary, {'validation_jobs/submitted': submitted,
                                               'validation_jobs/submission_failures': failures})
                self.assertNotIn('validation_jobs/completed', run.summary)
                self.assertFalse(any('pass' in key for key in run.summary))

    def test_skipped_validation_does_not_log_or_poll(self):
        scheduler, run = MagicMock(), SimpleNamespace(summary={})
        scheduler.submit_checkpoint.return_value = None
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertIsNone(tracking.submit_validation(scheduler, run, completed_step=1,
                                                          checkpoint='/irrelevant', per_epoch=32))
        self.assertEqual(output.getvalue(), '')
        self.assertEqual(run.summary, {})
        scheduler.summary.assert_not_called()

    def test_submission_with_tracking_disabled_still_reports_local_status(self):
        scheduler = MagicMock()
        record = {'status': 'submitted', 'epoch': 1, 'training_step': 32}
        scheduler.submit_checkpoint.return_value = record
        output = io.StringIO()
        with redirect_stdout(output):
            actual = tracking.submit_validation(scheduler, None, completed_step=32,
                                                 checkpoint='/pinned/model_032', per_epoch=32)
        self.assertIs(actual, record)
        self.assertEqual(json.loads(output.getvalue()), {'validation_submission': record})
        scheduler.summary.assert_not_called()


if __name__ == '__main__':
    unittest.main()
