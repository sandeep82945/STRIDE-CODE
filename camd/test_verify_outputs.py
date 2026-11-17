"""Inert filesystem fixtures for the host gate; no Torch, GPU or network."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('verify_outputs', Path(__file__).with_name('verify_outputs.py'))
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class HostGateTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / 'output'
        self.output.mkdir()
        self.train = self.root / 'train.json'
        write(self.train, [dict(id=f'train:{i}', question=f'Question {i}', ground_truth=i) for i in range(100)])
        self.ids = [f'train:{i}' for i in range(100)]
        self.hashes = {p.relative_to(gate.TRAINING).as_posix(): gate.sha(p) for p in gate.TRAINING.rglob('*.py')}
        self.args = dict(mode='train', model='qwen4b', finetune_mode='lora', seed=42, arm='grpo',
            batch_size=64, epochs=4, alpha=1., learning_rate=1e-4, validation_mode='off',
            rollouts=8, sampler_eager=False, data=str(self.train), data_sha256=gate.sha(self.train))
        self.started = dict(args=self.args, world_size=4, context_limit=8192,
            data_sha256=gate.sha(self.train), code_sha256=self.hashes, start_step=0, resume_from=None)
        self.cp = gate.checkpoint_module()

    def group(self, identity, count=8):
        return dict(id=identity, records=[dict(sample_index=i, token_ids=[1]) for i in range(count)])

    def policy(self, step, contract=None):
        target = self.output / f'adapter_{step:03d}'
        write(target / 'adapter_config.json', dict(r=16, lora_alpha=32, lora_dropout=0))
        (target / 'adapter_model.safetensors').write_bytes(b'inert fixture')
        write(target.with_name(target.name + '_manifest.json'), {p.name: gate.sha(p) for p in target.iterdir()})
        if contract:
            staging = self.cp.prepare_state(self.output, step)
            for name in self.cp._state_names(4, 'lora'):
                (staging / name).write_bytes(b'inert optimizer/RNG bytes; never deserialize')
            self.cp.commit_state(self.output, step, 4, contract, 'lora')

    def training_fixture(self, arm='grpo'):
        self.args['arm'] = arm
        contract = dict(version=6, mode='train', model='qwen4b', finetune_mode='lora', arm=arm,
            batch_schedule='epoch_shuffle_keep_partial', collective_padding='zero_loss_no_generated_samples',
            updates_per_epoch=2, training_questions=100, batch_size=64, epochs=4, world_size=4,
            K=8, seed=42, alpha=1., context_limit=8192, format_coefficient=0., warmup_steps=0,
            data_sha256=self.started['data_sha256'], code_sha256=self.hashes)
        self.started['training_contract'] = contract
        write(self.output / 'started.json', self.started)
        for epoch in range(4):
            order = list(self.ids)
            random.Random(42 + epoch).shuffle(order)
            for within, count in enumerate((64, 36)):
                step = epoch * 2 + within + 1
                batch = order[within * 64:within * 64 + count]
                for rank in range(4):
                    width, extra = divmod(count, 4)
                    offset = rank * width + min(rank, extra)
                    rows = batch[offset:offset + width + (rank < extra)]
                    write(self.output / f'rank_{rank}/rollouts_{step-1:03d}.json', [self.group(i) for i in rows])
                write(self.output / f'update_{step:03d}.json', dict(step=step, mode='train', arm=arm,
                    finetune_mode='lora', gradient_norm=.2, rollout_count=count * 8,
                    collective_padding_forwards=0,
                    totals=dict(groups=count, responses=count * 8, nonzero_advantage_tokens=4)))
            self.policy((epoch + 1) * 2, contract)
        write(self.output / 'done.json', dict(complete=True, finetune_mode='lora', steps=8, epochs=4,
            start_step=0, final_adapter=str(self.output / 'adapter_008'),
            final_state_checkpoint=str(self.output / 'state_008')))

    def evaluation_fixture(self):
        data = self.root / 'val.json'
        ids = [f'val:{i}' for i in range(100)]
        write(data, [dict(id=i) for i in ids])
        summary = dict(questions=100, responses=800, strict=dict(pass1=.5, pass8=1.), relaxed=dict(pass1=.5, pass8=1.))
        spec = dict(evaluation_only=True, model='qwen4b', rollouts_per_question=8, context_limit=8192,
            evaluation_seed=42, temperature=1., structural_reward=0, code_sha256=self.hashes,
            data=dict(path=str(data), sha256=gate.sha(data), evaluated_questions=100, prefix_limit=None),
            train_data=dict(path=str(self.train), sha256=gate.sha(self.train)),
            overlap=dict(checked=True, id_overlap=0, question_overlap=0), finetune_mode='base', checkpoint=None)
        write(self.output / 'eval_spec.json', spec)
        write(self.output / 'summary.json', summary)
        write(self.output / 'groups_000000.json', [self.group(i) for i in ids])
        write(self.output / 'done.json', dict(complete=True, evaluation_only=True, questions=100, responses=800,
            checkpoint=None, eval_spec_sha256=gate.sha(self.output / 'eval_spec.json'),
            summary_sha256=gate.sha(self.output / 'summary.json')))

    def test_complete_partial_training_and_real_epoch_checkpoints_pass(self):
        self.training_fixture()
        result = gate.verify('train', self.output)
        self.assertEqual(result['real_responses'], 3200)
        self.assertEqual(result['nonzero_updates'], 8)
        self.assertEqual(result['epoch_steps'], [2, 4, 6, 8])
        self.assertEqual([gate.read(self.output / f'update_{step:03d}.json')['rollout_count']
                          for step in range(1, 9)], [512, 288] * 4)
        self.assertTrue(result['optimizer_state_hashes_verified'])
        self.assertNotIn('torch', sys.modules)

    def test_all_four_arms_use_the_same_complete_100_question_budget(self):
        for arm in sorted(gate.ARMS):
            with self.subTest(arm=arm):
                self.output = self.root / arm
                self.output.mkdir()
                self.training_fixture(arm)
                result = gate.verify('train', self.output)
                self.assertEqual(result['arm'], arm)
                self.assertEqual(result['real_responses'], 3200)

    def test_previous_150_question_schedule_and_wrong_dataset_are_rejected(self):
        self.training_fixture()
        path = self.output / 'done.json'
        original = path.read_bytes()
        done = gate.read(path)
        done['steps'] = 12
        write(path, done)
        with self.assertRaisesRegex(ValueError, 'Incomplete or unexpected resumed schedule'):
            gate.verify('train', self.output)
        path.write_bytes(original)
        rows = gate.read(self.train)
        rows += [dict(id=f'extra:{i}') for i in range(50)]
        write(self.train, rows)
        self.started['data_sha256'] = self.args['data_sha256'] = gate.sha(self.train)
        write(self.output / 'started.json', self.started)
        with self.assertRaisesRegex(ValueError, 'exactly 100'):
            gate.verify('train', self.output)

    def test_dropped_tail_duplicates_padding_and_zero_signal_rejected(self):
        self.training_fixture()
        path = self.output / 'rank_3/rollouts_001.json'
        original = path.read_bytes()
        rows = gate.read(path)
        rows[-1] = copy.deepcopy(rows[0])
        write(path, rows)
        with self.assertRaisesRegex(ValueError, 'Dropped, repeated'):
            gate.verify('train', self.output)
        path.write_bytes(original)
        padding_path = self.output / 'update_002.json'
        original = padding_path.read_bytes()
        update = gate.read(padding_path)
        update['collective_padding_forwards'] = 16
        write(padding_path, update)
        with self.assertRaisesRegex(ValueError, 'collective padding'):
            gate.verify('train', self.output)
        padding_path.write_bytes(original)
        for step in range(1, 9):
            path = self.output / f'update_{step:03d}.json'
            item = gate.read(path)
            item['gradient_norm'] = 0
            write(path, item)
        with self.assertRaisesRegex(ValueError, 'No genuine update'):
            gate.verify('train', self.output)

    def test_native_false_success_checkpoint_tamper_and_nan_fail(self):
        self.training_fixture()
        path = self.output / 'state_008/optimizer.pt'
        original = path.read_bytes()
        path.write_bytes(b'corrupted checkpoint')
        with self.assertRaisesRegex(ValueError, 'state hash mismatch'):
            gate.verify('train', self.output)
        path.write_bytes(original)
        (self.output / 'update_001.json').write_text('{"gradient_norm":NaN}')
        with self.assertRaisesRegex(ValueError, 'Nonfinite JSON'):
            gate.verify('train', self.output)
        (self.output / 'done.json').unlink()
        completed = subprocess.run([sys.executable, '-I', '-S', str(Path(gate.__file__)), 'train',
                                   '--output', str(self.output)], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 1)
        self.assertFalse(json.loads(completed.stdout)['passed'])

    def test_smoke_requires_all_four_ranks_and_exact_current_source(self):
        self.args.update(mode='smoke', rollouts=2, smoke_cap=128, sampler_eager=True)
        write(self.output / 'started.json', self.started)
        write(self.output / 'done.json', dict(complete=True, steps=1, finetune_mode='lora',
              final_adapter=str(self.output / 'adapter_001')))
        write(self.output / 'update_001.json', dict(actor_behavior_abs_logp_mean=.001))
        self.policy(1)
        for rank in range(4):
            write(self.output / f'rank_{rank}/smoke_checks.json', dict(adapter_changed_logps=.01,
                fixed_reference_maxdiff=0., fresh_adapter_abs_logp_mean=.001,
                all_rank_trainable_weights_equal=True, trainable_sha256='a' * 64))
        self.assertTrue(gate.verify('smoke', self.output)['synthetic_update_only'])
        (self.output / 'rank_3/smoke_checks.json').unlink()
        with self.assertRaisesRegex(ValueError, 'Missing or unsafe'):
            gate.verify('smoke', self.output)
        self.started['code_sha256']['trainer.py'] = 'f' * 64
        write(self.output / 'started.json', self.started)
        with self.assertRaisesRegex(ValueError, 'source hashes'):
            gate.verify('smoke', self.output)

    def test_evaluation_hashes_exact_budget_and_failure_marker(self):
        self.evaluation_fixture()
        self.assertEqual(gate.verify('eval', self.output)['responses'], 800)
        path = self.output / 'summary.json'
        original = path.read_bytes()
        write(path, {'questions': 99})
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            gate.verify('eval', self.output)
        path.write_bytes(original)
        write(self.output / 'failed.json', {'complete': False})
        with self.assertRaisesRegex(ValueError, 'did not finish'):
            gate.verify('eval', self.output)

    def test_final_evaluation_requires_epoch_four_adapter_008(self):
        self.evaluation_fixture()
        spec_path, done_path = self.output / 'eval_spec.json', self.output / 'done.json'
        spec, done = gate.read(spec_path), gate.read(done_path)
        checkpoint = dict(finetune_mode='lora', path=str(self.root / 'trained' / 'adapter_008'))
        spec.update(finetune_mode='lora', checkpoint=checkpoint, epoch=4, training_step=8)
        done['checkpoint'] = checkpoint
        write(spec_path, spec)
        done['eval_spec_sha256'] = gate.sha(spec_path)
        write(done_path, done)
        self.assertEqual(gate.verify('eval', self.output)['training_step'], 8)
        spec['training_step'] = 12
        write(spec_path, spec)
        done['eval_spec_sha256'] = gate.sha(spec_path)
        write(done_path, done)
        with self.assertRaisesRegex(ValueError, 'completed epoch-four adapter_008'):
            gate.verify('eval', self.output)

    def test_upload_is_disabled_for_this_pilot_even_with_a_success_marker(self):
        write(self.output / 'hf_upload_status.json', dict(complete=True, private_all=True))
        with self.assertRaisesRegex(ValueError, 'publication is disabled'):
            gate.verify('upload', self.output)

    def test_cli_creates_fresh_report_and_never_overwrites(self):
        self.evaluation_fixture()
        report = self.root / 'verification.json'
        args = ['eval', '--output', str(self.output), '--report', str(report)]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(gate.main(args), 0)
            original = report.read_bytes()
            self.assertEqual(gate.main(args), 1)
        self.assertEqual(report.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
