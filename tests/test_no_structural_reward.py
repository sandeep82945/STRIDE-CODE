"""Actual scorer-to-credit checks: syntax eligibility is not a scalar reward.

Reference features are tiny supplied fixtures. Both actor-update functions are
exercised on CPU with a single-rank collective stub, not GPU or DDP serving.
"""
from contextlib import nullcontext
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from credit import standardize
import format_credit
import runtime_config
import scoring
import trainer
import trainer_distributed
import trainer_resumable
from tests.test_actor_cpu_integration import AsciiTokenizer


def formatted(answer):
    return ('Step 1: Add one and one.\nStep 2: State the result.\n'
            'Final Answer: \\boxed{' + str(answer) + '}')


class SingleRankCollectives:
    """A CPU identity-reduction stub; real distributed parity is tested elsewhere."""
    ReduceOp = torch.distributed.ReduceOp
    @staticmethod
    def get_world_size(): return 1
    @staticmethod
    def all_reduce(tensor, op=None): return None


class LogProbabilityActor(torch.nn.Module):
    def __init__(self, length):
        super().__init__()
        self.logps = torch.nn.Parameter(torch.zeros(length))
    def no_sync(self): return nullcontext()


class NoStructuralRewardTests(unittest.TestCase):
    def groups(self, texts, arm, *, extra_format_score=None):
        tok = AsciiTokenizer()
        rows = [dict(id='question', question='What is one plus one?', ground_truth=2)]
        prompt = trainer.render(tok, rows[0])
        generated = dict(rows=[dict(id='question', prompt_token_ids=prompt, max_tokens=8192-len(prompt),
            samples=[dict(sample_index=i, token_ids=tok.encode(text), text=text,
                          logprobs=[0.]*len(tok.encode(text)), finish_reason='stop')
                     for i, text in enumerate(texts)])])
        values = []
        for i, text in enumerate(texts):
            score = scoring.annotate(tok, prompt, tok.encode(text), 2, 'stop')
            steps = torch.eye(8)[2*i:2*i+len(score['spans'])]
            whole = torch.eye(8)[i]
            pooled = F.normalize(steps.sum(0), dim=0) if len(steps) else torch.zeros(8)
            values.append((whole, steps, pooled))
        original_annotate = scoring.annotate
        def annotated(*args):
            result = original_annotate(*args)
            if extra_format_score is not None:
                # These diagnostic names must not become extra scalar rewards.
                result.update(format_reward=extra_format_score, structural_reward=extra_format_score,
                              format_score=extra_format_score, reward=extra_format_score,
                              step_count=1000 if extra_format_score > 0 else 0)
            return result
        with patch('trainer.features', side_effect=values), patch('scoring.annotate', side_effect=annotated), \
                patch.object(format_credit, 'compute_format_credit', side_effect=AssertionError('Dormant format reward called')) as legacy:
            result = trainer.annotate_groups(tok, rows, generated, actor=None if arm == 'grpo' else object(), arm=arm)
            legacy.assert_not_called()
        return result

    def test_wrong_but_formatted_rollouts_receive_no_reward_or_actor_update(self):
        texts = [formatted(3), 'Final Answer: \\boxed{3}']
        for arm in runtime_config.ARMS:
            groups = self.groups(texts, arm)
            scores = [r['score'] for r in groups[0]['records']]
            self.assertEqual([s['step_eligible'] for s in scores], [True, False])
            self.assertEqual([s['answer_correct'] for s in scores], [False, False])
            self.assertFalse(bool(groups[0]['advantages'].any()), arm)
            for driver in (trainer_distributed, trainer_resumable):
                with self.subTest(arm=arm, driver=driver.__name__):
                    actor = LogProbabilityActor(groups[0]['advantages'].shape[1])
                    before = actor.logps.detach().clone()
                    optimizer = torch.optim.AdamW(actor.parameters(), lr=.01, weight_decay=0.)
                    with patch('trainer.selected_logps', side_effect=lambda actor, prompt, response: actor.logps[:len(response)]):
                        result = driver.update_distributed(actor, optimizer, groups,
                            dist=SingleRankCollectives(), force_signal=False)
                    self.assertEqual(result['loss'], 0.)
                    self.assertEqual(result['gradient_norm'], 0.)
                    self.assertTrue(torch.equal(actor.logps, before))

    def test_correct_unformatted_response_keeps_accuracy_credit(self):
        texts = ['Final Answer: \\boxed{2}', formatted(3)]
        expected = standardize(torch.tensor([1., 0.]))
        for arm in runtime_config.ARMS:
            with self.subTest(arm=arm):
                group = self.groups(texts, arm)[0]
                self.assertEqual([r['score']['step_eligible'] for r in group['records']], [False, True])
                self.assertEqual([r['score']['answer_correct'] for r in group['records']], [True, False])
                for i, record in enumerate(group['records']):
                    actual = group['advantages'][i, :len(record['token_ids'])]
                    self.assertTrue(torch.equal(actual, expected[i].expand_as(actual)))
                self.assertGreater(float(group['advantages'][0, 0]), 0.)
                self.assertLess(float(group['advantages'][1, 0]), 0.)

    def test_format_diagnostic_values_cannot_change_active_arm_credit(self):
        texts = [formatted(2), formatted(2), formatted(3)]
        for arm in runtime_config.ARMS:
            with self.subTest(arm=arm):
                original = self.groups(texts, arm)[0]
                if arm != 'grpo':
                    self.assertTrue(original['diagnostics']['gate_active'])
                    self.assertGreater(original['diagnostics']['arms'][arm]['aux_positive_tokens'], 0)
                for value in (-1e9, 1e9):
                    altered = self.groups(texts, arm, extra_format_score=value)[0]
                    self.assertTrue(torch.equal(original['advantages'], altered['advantages']))
                    # The wrong formatted rollout never receives auxiliary credit.
                    expected_wrong = standardize(torch.tensor([1., 1., 0.]))[2]
                    wrong = altered['records'][2]
                    actual = altered['advantages'][2, :len(wrong['token_ids'])]
                    self.assertTrue(torch.equal(actual, expected_wrong.expand_as(actual)))

    def test_nonzero_format_coefficient_is_rejected_for_every_public_arm(self):
        for arm in runtime_config.ARMS:
            with self.subTest(arm=arm), self.assertRaisesRegex(ValueError, 'Structural rewards are disabled'):
                trainer.annotate_groups(None, [], {'rows': []}, actor=None if arm == 'grpo' else object(),
                                        arm=arm, format_coefficient=.1)


if __name__ == '__main__':
    unittest.main()
