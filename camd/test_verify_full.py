"""Inert host-gate fixtures; no Torch, GPU, network or paid API calls."""
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('verify_full', Path(__file__).with_name('verify_full.py'))
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@contextmanager
def changed(path, update):
    original = path.read_bytes()
    value = gate.read(path)
    update(value)
    write(path, value)
    try:
        yield
    finally:
        path.write_bytes(original)


class FullHostGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.train = cls.root / 'train.json'
        cls.ids = [f'train:{i}' for i in range(2048)]
        write(cls.train, [dict(id=i, question=f'Question {i}', ground_truth=n) for n, i in enumerate(cls.ids)])
        cls.sha_patch = patch.object(gate, 'EXPECTED_TRAIN_SHA256', gate.sha(cls.train))
        cls.sha_patch.start()
        cls.output = cls.root / 'training'
        cls.output.mkdir()
        cls.hashes = {p.relative_to(gate.TRAINING).as_posix(): gate.sha(p) for p in gate.TRAINING.rglob('*.py')}
        cls.args = dict(thinking_mode=False, mode='train', model=gate.MODEL_KEY, finetune_mode='lora', seed=42, arm='local_positive',
            batch_size=64, epochs=4, alpha=gate.ALPHA, learning_rate=2e-5, kl_coefficient=.01, warmup_steps=10, validation_mode='off',
            rollouts=8, sampler_eager=False, keep_all_checkpoints=True,
            resume_state_retention='latest-epoch',
            data=str(cls.train), data_sha256=gate.sha(cls.train))
        cls.started = dict(args=cls.args, world_size=4, context_limit=8192, checkpoint_retention='all',
            resume_state_retention='latest-epoch',
            data_sha256=gate.sha(cls.train), code_sha256=cls.hashes, start_step=0, resume_from=None)
        cls.contract = dict(thinking_mode=False, version=7, mode='train', model=gate.MODEL_KEY, finetune_mode='lora', arm='local_positive',
            batch_schedule='epoch_shuffle_keep_partial', collective_padding='zero_loss_no_generated_samples',
            updates_per_epoch=32, training_questions=2048, batch_size=64, epochs=4, world_size=4,
            K=8, seed=42, alpha=gate.ALPHA, context_limit=8192, format_coefficient=0., warmup_steps=10, kl_coefficient=.01,
        learning_rate_schedule='linear_warmup_then_constant', kl_reference='frozen_initial_base',
        kl_estimator='original_grpo_k3', kl_normalization='global_generated_tokens',
            data_sha256=cls.started['data_sha256'], code_sha256=cls.hashes)
        cls.started['training_contract'] = cls.contract
        write(cls.output / 'started.json', cls.started)
        cls.cp = gate.checkpoint_module()
        for step in range(129):
            cls.policy(cls.output, step, cls.contract)
            cls.cp.retain_latest_epoch_states(cls.output, step, 32)
        for epoch in range(4):
            order = list(cls.ids)
            random.Random(42 + epoch).shuffle(order)
            for within in range(32):
                step = epoch * 32 + within + 1
                batch = order[within * 64:(within + 1) * 64]
                for rank in range(4):
                    rows = batch[rank * 16:(rank + 1) * 16]
                    write(cls.output / f'rank_{rank}/rollouts_{step-1:03d}.json',
                          [dict(id=i, records=[dict(sample_index=k, token_ids=[1]) for k in range(8)]) for i in rows])
                write(cls.output / f'update_{step:03d}.json', dict(step=step, mode='train', arm='local_positive',
                    finetune_mode='lora', gradient_norm=.2, loss=.01, rollout_count=512, response_tokens=512,
                    collective_padding_forwards=0, learning_rate=2e-5 * min(step/10, 1.),
                    kl_coefficient=.01, sampled_kl_mean=.2, kl_loss=.002,
                    totals=dict(groups=64, responses=512, response_tokens=512, nonzero_advantage_tokens=4)))
                write(cls.output / f'checkpoint_{step:03d}.json', dict(completed_step=step,
                    state_checkpoint=str(cls.output / f'state_{step:03d}'), checkpoint_retention='all', pruned_steps=[],
                    resume_state_retention='latest-epoch',
                    pruned_state_steps=sorted({step - 1, (step - 1) // 32 * 32} - {step, step // 32 * 32})))
        write(cls.output / 'done.json', dict(complete=True, finetune_mode='lora', steps=128, epochs=4,
            start_step=0, checkpoint_retention='all', resume_state_retention='latest-epoch',
            final_adapter=str(cls.output / 'adapter_128'), final_state_checkpoint=str(cls.output / 'state_128')))
        cls.smoke = cls.root / 'smoke'
        cls.smoke.mkdir()
        smoke_args = dict(cls.args, mode='smoke', smoke_cap=128, rollouts=2, sampler_eager=True)
        smoke_args.pop('keep_all_checkpoints')
        write(cls.smoke / 'started.json', dict(cls.started, args=smoke_args))
        write(cls.smoke / 'done.json', dict(complete=True, steps=1, finetune_mode='lora',
                                           final_adapter=str(cls.smoke / 'adapter_001')))
        write(cls.smoke / 'update_001.json', dict(actor_behavior_abs_logp_mean=.001))
        cls.policy(cls.smoke, 1)
        for rank in range(4):
            write(cls.smoke / f'rank_{rank}/smoke_checks.json', dict(adapter_changed_logps=.01,
                fixed_reference_maxdiff=0., fresh_adapter_abs_logp_mean=.001,
                all_rank_trainable_weights_equal=True, trainable_sha256='a' * 64))

    @classmethod
    def tearDownClass(cls):
        cls.sha_patch.stop()
        cls.temp.cleanup()

    @classmethod
    def policy(cls, output, step, contract=None):
        target = output / f'adapter_{step:03d}'
        write(target / 'adapter_config.json', dict(peft_type='LORA', task_type='CAUSAL_LM',
            r=16, lora_alpha=32, lora_dropout=0, target_modules=sorted(gate.LORA_TARGETS)))
        (target / 'adapter_model.safetensors').write_bytes(b'inert fixture, not real model weights')
        write(target.with_name(target.name + '_manifest.json'), {p.name: gate.sha(p) for p in target.iterdir()})
        if contract:
            staging = cls.cp.prepare_state(output, step)
            for name in cls.cp._state_names(4, 'lora'):
                (staging / name).write_bytes(b'inert optimizer/RNG fixture')
            cls.cp.commit_state(output, step, 4, contract, 'lora')

    def test_full_schedule_verifies_129_adapters_one_resume_state_and_65536_responses(self):
        result = gate.verify('train', self.output)
        self.assertEqual(result['real_responses'], 65536)
        self.assertEqual(result['steps'], 128)
        self.assertEqual(result['epochs'], 4)
        self.assertEqual(result['retained_checkpoint_pairs'], 1)
        self.assertEqual(result['retained_adapter_checkpoints'], 129)
        self.assertEqual(result['archived_commit_receipts'], 129)
        self.assertEqual(result['nonzero_updates'], 128)
        self.assertEqual(result['epoch_steps'], list(range(32, 129, 32)))
        self.assertTrue(result['optimizer_state_hashes_verified'])
        self.assertFalse(result['optimizer_tensors_deserialized'])

    def test_frozen_dataset_and_full_settings_are_required(self):
        with patch.object(gate, 'EXPECTED_TRAIN_SHA256', '0' * 64):
            with self.assertRaisesRegex(ValueError, 'frozen 2048'):
                gate.verify('train', self.output)
        for field, value in [('epochs', 10), ('batch_size', 128), ('learning_rate', .001), ('arm', 'base'), ('warmup_steps', 0), ('kl_coefficient', 0.)]:
            with self.subTest(field=field), changed(self.output / 'started.json',
                    lambda x: x['args'].update({field: value})):
                with self.assertRaisesRegex(ValueError, 'settings differ'):
                    gate.verify('train', self.output)
        with changed(self.output / 'started.json', lambda x: x['args'].update(keep_all_checkpoints=False)):
            with self.assertRaisesRegex(ValueError, 'retain every checkpoint'):
                gate.verify('train', self.output)

    def test_thinking_model_alpha_and_stabilization_contract_cannot_drift(self):
        for field, value in [('model', 'qwen17b' if gate.MODEL_KEY == 'qwen4b' else 'qwen4b'),
                             ('alpha', 2. if gate.ALPHA == 1. else 1.)]:
            with self.subTest(field=field), changed(self.output / 'started.json',
                    lambda x: x['args'].update({field: value})):
                with self.assertRaisesRegex(ValueError, 'settings differ'):
                    gate.verify('train', self.output)
        for value in (True, None, 0):
            for kind, output in [('train', self.output), ('smoke', self.smoke)]:
                with self.subTest(kind=kind, value=value), changed(output / 'started.json',
                        lambda x: x['args'].update(thinking_mode=value)):
                    with self.assertRaisesRegex(ValueError, 'nonthinking arguments'):
                        gate.verify(kind, output)
            with changed(self.output / 'started.json',
                    lambda x: x['training_contract'].update(thinking_mode=value)):
                with self.assertRaisesRegex(ValueError, 'nonthinking training contract'):
                    gate.verify('train', self.output)
        for field, value in [('version', 6), ('warmup_steps', 0), ('kl_coefficient', 0.)]:
            with self.subTest(field=field), changed(self.output / 'started.json',
                    lambda x: x['training_contract'].update({field: value})):
                with self.assertRaisesRegex(ValueError, 'Training contract'):
                    gate.verify('train', self.output)

    def test_missing_initial_or_interior_receipts_are_rejected(self):
        for step in (0, 123):
            state = self.output / 'checkpoint_receipts' / f'state_{step:03d}'
            moved = self.root / 'held-state'
            state.rename(moved)
            try:
                with self.assertRaisesRegex(ValueError, 'commit receipts'):
                    gate.verify('train', self.output)
            finally:
                moved.rename(state)

    def test_final_optimizer_and_interior_adapter_corruption_are_detected(self):
        for path, message in ((self.output / 'state_128/optimizer.pt', 'state hash mismatch'),
                              (self.output / 'adapter_097/adapter_model.safetensors', 'Policy hash mismatch')):
            original = path.read_bytes()
            path.write_bytes(b'corrupted')
            try:
                with self.assertRaisesRegex(ValueError, message):
                    gate.verify('train', self.output)
            finally:
                path.write_bytes(original)

    def test_wrong_question_order_rollout_index_or_token_count_is_rejected(self):
        path = self.output / 'rank_2/rollouts_122.json'
        with changed(path, lambda x: x[1].update(id=x[0]['id'])):
            with self.assertRaisesRegex(ValueError, 'wrong-epoch'):
                gate.verify('train', self.output)
        with changed(path, lambda x: x[0]['records'][1].update(sample_index=0)):
            with self.assertRaisesRegex(ValueError, 'reordered generated'):
                gate.verify('train', self.output)
        with changed(self.output / 'update_123.json', lambda x: x.update(response_tokens=0)):
            with self.assertRaisesRegex(ValueError, 'token counts'):
                gate.verify('train', self.output)

    def test_empty_signal_and_nonfinite_loss_are_rejected(self):
        originals = {}
        try:
            for step in range(1, 129):
                path = self.output / f'update_{step:03d}.json'
                originals[path] = path.read_bytes()
                value = gate.read(path)
                value['gradient_norm'] = 0
                write(path, value)
            with self.assertRaisesRegex(ValueError, 'No genuine update'):
                gate.verify('train', self.output)
        finally:
            for path, payload in originals.items():
                path.write_bytes(payload)
        with changed(self.output / 'update_001.json', lambda x: x.update(loss=float('nan'))):
            with self.assertRaisesRegex(ValueError, 'Nonfinite JSON'):
                gate.verify('train', self.output)

    def test_checkpoint_retention_metadata_and_final_step_are_required(self):
        with changed(self.output / 'checkpoint_127.json', lambda x: x.update(pruned_steps=[125])):
            with self.assertRaisesRegex(ValueError, 'retention metadata'):
                gate.verify('train', self.output)
        with changed(self.output / 'done.json', lambda x: x.update(steps=127)):
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                gate.verify('train', self.output)

    def test_smoke_requires_actual_changed_adapter_fixed_reference_and_rank_agreement(self):
        self.assertTrue(gate.verify('smoke', self.smoke)['integrated_adapter_reload_checked'])
        for field, value in [('adapter_changed_logps', 0), ('fixed_reference_maxdiff', .001),
                              ('fresh_adapter_abs_logp_mean', .1), ('all_rank_trainable_weights_equal', False)]:
            with self.subTest(field=field), changed(self.smoke / 'rank_3/smoke_checks.json',
                                                    lambda x: x.update({field: value})):
                with self.assertRaisesRegex(ValueError, 'integrated smoke'):
                    gate.verify('smoke', self.smoke)
        with changed(self.smoke / 'rank_3/smoke_checks.json', lambda x: x.update(trainable_sha256='b' * 64)):
            with self.assertRaisesRegex(ValueError, 'Learner weights differ'):
                gate.verify('smoke', self.smoke)

    def test_adapter_architecture_and_current_source_identity_are_verified(self):
        with changed(self.smoke / 'started.json', lambda x: x['code_sha256'].update({'trainer.py': 'f' * 64})):
            with self.assertRaisesRegex(ValueError, 'source hashes'):
                gate.verify('smoke', self.smoke)
        with changed(self.smoke / 'adapter_001/adapter_config.json', lambda x: x.update(r=8)):
            with self.assertRaisesRegex(ValueError, 'LoRA architecture'):
                gate.policy_config(self.smoke / 'adapter_001')

    def test_cli_works_without_site_packages_and_reports_are_immutable(self):
        result = subprocess.run([sys.executable, '-I', '-S', str(Path(gate.__file__)), '--help'],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('smoke,train', result.stdout)
        report = self.root / 'smoke-verification.json'
        args = ['smoke', '--output', str(self.smoke), '--report', str(report)]
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(gate.main(args), 0)
            before = report.read_bytes()
            self.assertEqual(gate.main(args), 1)
        self.assertEqual(report.read_bytes(), before)
        with self.assertRaisesRegex(ValueError, 'Only smoke and training'):
            gate.verify('eval', self.output)


if __name__ == '__main__':
    unittest.main()
