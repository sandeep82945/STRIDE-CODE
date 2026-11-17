"""Analytic KL gradients, real frozen-base LoRA, and distributed parity checks."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

import stabilization
import trainer
import trainer_resumable
from tests.test_batched_learner import tiny_actor, fixture_groups


def tiny_lora():
    from peft import LoraConfig, get_peft_model
    actor = get_peft_model(tiny_actor(), LoraConfig(r=2, lora_alpha=4,
        lora_dropout=0., bias='none', task_type='CAUSAL_LM',
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']))
    # A nonzero departure is essential: a fresh zero-B LoRA has zero KL gradient.
    with torch.no_grad():
        for name, parameter in actor.named_parameters():
            if 'lora_B' in name:
                parameter.normal_(std=.07)
    return actor


def kl_distributed_worker(rank, init_file, result_dir):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + init_file, rank=rank, world_size=4)
    try:
        initial = tiny_lora()
        groups = fixture_groups(initial, rank)
        # Cover actual truncated records and all-zero advantages: KL must persist.
        for group in groups:
            group['advantages'].zero_()
            for record in group['records']:
                record['finish_reason'] = 'length'
        result = []
        for size in (1, 2):
            actor = DistributedDataParallel(copy.deepcopy(initial), broadcast_buffers=False)
            optimizer = torch.optim.SGD([p for p in actor.parameters() if p.requires_grad], lr=.03)
            stats = trainer_resumable.update_learner(actor, optimizer, groups, dist=dist,
                learner_microbatch_size=size, kl_coefficient=.01)
            result.append((stats, [p.detach().clone() for p in actor.parameters() if p.requires_grad],
                           [p.grad.detach().clone() for p in actor.parameters() if p.requires_grad]))
            assert all(p.grad is None for p in actor.parameters() if not p.requires_grad)
            assert stats['sampled_kl_mean'] > 0
            assert stats['gradient_norm'] > 0
            assert stats['policy_loss'] == 0
            assert abs(stats['loss'] - stats['kl_loss']) < 1e-8
        a, b = result
        error = max(float((x-y).abs().max()) for x, y in zip(a[1], b[1]))
        grad_error = max(float((x-y).abs().max()) for x, y in zip(a[2], b[2]))
        assert error < 1e-6, error
        assert grad_error < 2e-6, grad_error
        assert abs(a[0]['sampled_kl_mean'] - b[0]['sampled_kl_mean']) < 1e-7
        assert a[0]['response_tokens'] == b[0]['response_tokens']
        assert b[0]['collective_padding_forwards'] == 2
        Path(result_dir, f'rank_{rank}.json').write_text(json.dumps(dict(
            weight_error=error, gradient_error=grad_error, sampled_kl_mean=b[0]['sampled_kl_mean'])))
    finally:
        dist.destroy_process_group()


class StabilizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_sampled_k3_formula_gradient_and_detached_reference(self):
        new = torch.tensor([-2., -3., -4., -1.], requires_grad=True)
        reference = torch.tensor([-2., -2.7, -4.2, -1.0001], requires_grad=True)
        delta = reference.detach() - new.detach()
        actual = stabilization.sampled_k3_kl(new, reference)
        expected = torch.expm1(delta) - delta
        torch.testing.assert_close(actual, expected)
        self.assertTrue(bool((actual >= 0).all()))
        actual.sum().backward()
        torch.testing.assert_close(new.grad, -torch.expm1(delta), atol=1e-7, rtol=1e-5)
        self.assertIsNone(reference.grad)
        # Value under the policy distribution equals categorical reverse KL;
        # the implemented sampled gradient intentionally omits bias correction.
        p, q = torch.tensor([.2, .3, .5]), torch.tensor([.4, .4, .2])
        estimate = (p * stabilization.sampled_k3_kl(p.log(), q.log())).sum()
        torch.testing.assert_close(estimate, (p * (p.log() - q.log())).sum())

    def test_reference_is_fixed_detached_and_restores_adapters_and_training_flags(self):
        actor = tiny_lora().train()
        actor.base_model.model.model.layers[0].self_attn.eval()
        flags = [m.training for m in actor.modules()]
        sequences = [([1], [7]), ([2, 3, 4], [8, 9, 10, 11])]
        baseline = stabilization.reference_logps(actor, sequences)
        self.assertEqual([len(x) for x in baseline], [1, 4])
        self.assertTrue(all(not x.requires_grad and x.grad_fn is None for x in baseline))
        self.assertEqual(flags, [m.training for m in actor.modules()])
        self.assertFalse(any(getattr(m, 'disable_adapters', False) is True for m in actor.modules()))
        before_policy = trainer.selected_logps(actor, *sequences[1]).detach()
        with torch.no_grad():
            for name, parameter in actor.named_parameters():
                if 'lora_B' in name:
                    parameter.add_(.1)
        after = stabilization.reference_logps(actor, sequences)
        for a, b in zip(baseline, after):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertGreater(float((trainer.selected_logps(actor, *sequences[1]).detach() - before_policy).abs().max()), 0)
        # Nested usage must not re-enable an already disabled adapter.
        with actor.disable_adapter():
            stabilization.reference_logps(actor, sequences)
            self.assertTrue(any(getattr(m, 'disable_adapters', False) is True for m in actor.modules()))
        try:
            with stabilization.frozen_lora_reference(actor):
                raise RuntimeError('test failure')
        except RuntimeError:
            pass
        self.assertEqual(flags, [m.training for m in actor.modules()])
        self.assertFalse(any(getattr(m, 'disable_adapters', False) is True for m in actor.modules()))
        self.assertTrue(all(p.grad is None for p in actor.parameters()))

    def test_reference_rejects_a_trainable_base(self):
        actor = tiny_lora()
        actor.get_base_model().model.embed_tokens.weight.requires_grad_(True)
        with self.assertRaisesRegex(ValueError, 'base parameter'):
            stabilization.reference_logps(actor, [([1], [2])])

    def test_real_four_rank_kl_single_pair_parity_with_zero_reward_and_truncation(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(kl_distributed_worker, args=(str(Path(directory) / 'gloo'), directory), nprocs=4, join=True)
            self.assertEqual(len(list(Path(directory).glob('rank_*.json'))), 4)

    def test_warmup_absolute_update_schedule_and_adam_resume_parity(self):
        rates = [stabilization.learning_rate_at_update(2e-5, 10, step) for step in range(15)]
        self.assertAlmostEqual(rates[0], 2e-6)
        self.assertEqual(rates[9:], [2e-5] * 6)
        self.assertEqual(stabilization.learning_rate_at_update(2e-5, 0, 0), 2e-5)
        original = torch.nn.Parameter(torch.tensor([.3, -.7]))
        resumed = torch.nn.Parameter(original.detach().clone())
        uninterrupted = torch.optim.AdamW([original], lr=2e-5, weight_decay=0.)
        interrupted = torch.optim.AdamW([resumed], lr=2e-5, weight_decay=0.)
        for step in range(15):
            for parameter, optimizer in ((original, uninterrupted), (resumed, interrupted)):
                stabilization.set_update_learning_rate(optimizer, 2e-5, 10, step)
                optimizer.zero_grad(set_to_none=True)
                ((parameter - torch.tensor([.6, .1])).square().sum()).backward()
                optimizer.step()
            if step in (3, 11):
                state = copy.deepcopy(interrupted.state_dict())
                resumed = torch.nn.Parameter(resumed.detach().clone())
                interrupted = torch.optim.AdamW([resumed], lr=999., weight_decay=0.)
                interrupted.load_state_dict(state)
        torch.testing.assert_close(original, resumed, rtol=0, atol=0)
        self.assertEqual(uninterrupted.param_groups[0]['lr'], interrupted.param_groups[0]['lr'])
        for key in ('step', 'exp_avg', 'exp_avg_sq'):
            torch.testing.assert_close(uninterrupted.state[original][key], interrupted.state[resumed][key], rtol=0, atol=0)

    def test_contract_and_invalid_configuration(self):
        import runtime_config as config
        from tests.test_training_config import arguments
        args = arguments('train', '--finetune-mode', 'lora', '--learning-rate', '2e-5',
                         '--warmup-steps', '10', '--kl-coefficient', '.01')
        config.validate_options(args, resumable=True)
        with patch('trainer_resumable.version', return_value='test'):
            contract = trainer_resumable.training_contract(args, 4, 2048, {'content_sha256': 'fixture'})
        self.assertEqual(contract['warmup_steps'], 10)
        self.assertEqual(contract['kl_coefficient'], .01)
        self.assertEqual(contract['kl_reference'], 'frozen_initial_base')
        self.assertEqual(contract['kl_estimator'], 'original_grpo_k3')
        self.assertIn('stabilization.py', contract['code_sha256'])
        for key, value in [('warmup_steps', -1), ('warmup_steps', 1.5), ('warmup_steps', True),
                           ('kl_coefficient', -.1), ('kl_coefficient', float('nan')),
                           ('kl_coefficient', float('inf')), ('finetune_mode', 'full')]:
            invalid = copy.copy(args)
            setattr(invalid, key, value)
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                config.validate_options(invalid, resumable=True)


if __name__ == '__main__':
    unittest.main()
