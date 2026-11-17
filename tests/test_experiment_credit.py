"""Mathematical and training-path checks for the four-setting comparison."""
import unittest
from unittest.mock import patch
import torch
import torch.nn.functional as F

from credit import compute_group_credit as original_credit
from experiment_credit import compute_group_credit
from feature_pooling import pool_reference_hidden
from runtime_config import ARMS
import trainer


def fixture():
    # Unequal lengths and deliberately large excluded tokens catch accidental
    # whole-response pooling, gap coverage and equal-step weighting.
    hidden = [torch.tensor([[10., 0, 0], [99., 99, 99], [0., 1, 0], [0., 1, 0], [0., 1, 0], [-99., 0, 0]]),
              torch.tensor([[10., 0, 0], [99., 99, 99], [0., 0, 1], [0., 0, 1], [0., 0, 1], [-99., 0, 0]]),
              torch.eye(6, 3)]
    spans = [[(0, 0), (2, 4)]] * 3
    values = [pool_reference_hidden(h, s, include_pooled=True) for h, s in zip(hidden, spans)]
    return dict(answer_correct=[1, 1, 0], step_embeddings=[v[1] for v in values],
                step_spans=spans, step_eligible=[True] * 3,
                whole_embeddings=torch.stack([v[0] for v in values]),
                response_mask=torch.ones(3, 6, dtype=torch.bool),
                pooled_step_embeddings=torch.stack([v[2] for v in values]))


class ExperimentCreditTests(unittest.TestCase):
    def test_structural_reward_cannot_be_enabled(self):
        for actor in (None, object()):
            with self.assertRaisesRegex(ValueError, 'Structural rewards are disabled'):
                trainer.annotate_groups(None, [], {'rows': []}, actor=actor,
                                        arm='grpo', format_coefficient=.1)

    def test_raw_token_pooling_excludes_gaps_and_preserves_original_features(self):
        hidden = torch.tensor([[10., 0.], [200., 200.], [0., 1.], [0., 1.], [0., 1.]])
        spans = [(0, 0), (2, 4)]
        whole, steps, pooled = pool_reference_hidden(hidden, spans, include_pooled=True)
        self.assertTrue(torch.equal(whole, F.normalize(hidden.mean(0), dim=0)))
        self.assertTrue(torch.equal(steps, torch.stack([F.normalize(hidden[a:b+1].mean(0), dim=0) for a, b in spans])))
        self.assertTrue(torch.allclose(pooled, F.normalize(torch.tensor([10., 3.]), dim=0)))
        self.assertFalse(torch.allclose(pooled, F.normalize((steps * torch.tensor([[1.], [3.]])).sum(0), dim=0)))
        self.assertFalse(torch.allclose(pooled, whole))
        split = [(0, 0), (2, 2), (3, 4)]
        self.assertTrue(torch.equal(pooled, pool_reference_hidden(hidden, split, include_pooled=True)[2]))

    def test_original_three_arm_results_are_bitwise_unchanged(self):
        x = fixture()
        for alpha in (0., .5, 1., 2.):
            old = original_credit(**{k: v for k, v in x.items() if k != 'pooled_step_embeddings'}, alpha=alpha)
            new = compute_group_credit(**x, alpha=alpha)
            self.assertEqual(set(new['advantages']), set(ARMS))
            for arm in ('grpo', 'local_positive', 'token_uniform'):
                self.assertTrue(torch.equal(old['advantages'][arm], new['advantages'][arm]), arm)
                self.assertTrue(torch.equal(old['auxiliary'][arm], new['auxiliary'][arm]), arm)

    def test_single_step_matches_independent_schur_complement(self):
        x = fixture()
        result = compute_group_credit(**x, alpha=.7)
        phi = F.normalize(x['pooled_step_embeddings'][:2].double(), dim=1)
        gram = phi @ phi.T
        eps = 1e-3
        for i in range(2):
            j = 1 - i
            # One other response: closed-form conditional variance, no Cholesky.
            u = (gram[i, i] - gram[i, j]**2 / (gram[j, j] + eps)).clamp(0, 1)
            u = 0. if u <= eps else float(u)
            actual = result['auxiliary']['single_step'][i, result['mapped_mask'][i]]
            self.assertTrue(torch.allclose(actual, torch.full_like(actual, .7*u), atol=2e-6, rtol=0))
        self.assertFalse(bool(result['auxiliary']['single_step'][~result['mapped_mask']].any()))
        self.assertFalse(torch.allclose(result['auxiliary']['single_step'], result['auxiliary']['token_uniform']))

    def test_same_gate_nonnegative_bonus_and_padding_support(self):
        x = fixture()
        x['response_mask'][:, -1] = False
        for alpha in (0., .5, 2.):
            result = compute_group_credit(**x, alpha=alpha)
            for arm in ARMS:
                aux = result['auxiliary'][arm]
                self.assertTrue(bool(((aux >= 0) & (aux <= alpha)).all()))
                self.assertFalse(bool(aux[~result['mapped_mask']].any()))
        x['step_eligible'][1] = False
        result = compute_group_credit(**x)
        for arm in ARMS:
            self.assertTrue(torch.equal(result['advantages'][arm], result['main']))

    def test_ineligible_embedding_cannot_influence_single_step(self):
        x = fixture()
        first = compute_group_credit(**x)['auxiliary']['single_step']
        x['pooled_step_embeddings'][2] = torch.tensor([123., -999., 3.])
        second = compute_group_credit(**x)['auxiliary']['single_step']
        self.assertTrue(torch.equal(first, second))

    def test_single_declared_step_reduces_to_local_positive(self):
        x = fixture()
        x['step_spans'] = [[(0, 4)]] * 3
        x['step_embeddings'] = [v[None] for v in x['pooled_step_embeddings']]
        result = compute_group_credit(**x)
        self.assertTrue(torch.equal(result['auxiliary']['single_step'], result['auxiliary']['local_positive']))

    def test_duplicates_and_nonfinite_inputs(self):
        x = fixture()
        x['pooled_step_embeddings'][1] = x['pooled_step_embeddings'][0]
        self.assertFalse(bool(compute_group_credit(**x)['auxiliary']['single_step'].any()))
        x['pooled_step_embeddings'][0, 0] = float('nan')
        with self.assertRaises(ValueError):
            compute_group_credit(**x)

    def test_annotation_selects_all_four_arms_and_records_pooled_features(self):
        x = fixture()
        rows = [dict(id='q', ground_truth=1)]
        generated = dict(rows=[dict(id='q', prompt_token_ids=[42], max_tokens=8,
                                   samples=[dict(token_ids=[i]*6, finish_reason='stop') for i in range(3)])])
        scores = [dict(answer_correct=x['answer_correct'][i], spans=x['step_spans'][i], step_eligible=True) for i in range(3)]
        values = [(x['whole_embeddings'][i], x['step_embeddings'][i], x['pooled_step_embeddings'][i]) for i in range(3)]
        expected = compute_group_credit(**x)
        for arm in ARMS:
            with patch('scoring.annotate', side_effect=scores), patch('trainer.features', side_effect=values) as feature:
                groups = trainer.annotate_groups(None, rows, generated, actor=object(), arm=arm)
            self.assertTrue(torch.equal(groups[0]['advantages'], expected['advantages'][arm]))
            self.assertEqual(groups[0]['features']['pooled_steps'], x['pooled_step_embeddings'].tolist())
            self.assertTrue(all(call.kwargs['include_pooled'] for call in feature.call_args_list))
        with patch('scoring.annotate', side_effect=scores), patch('trainer.features') as feature:
            groups = trainer.annotate_groups(None, rows, generated, actor=None, arm='grpo')
            feature.assert_not_called()
        self.assertTrue(torch.equal(groups[0]['advantages'], expected['advantages']['grpo']))
        with self.assertRaisesRegex(ValueError, 'requires reference features'):
            trainer.annotate_groups(None, rows, generated, actor=None, arm='single_step')


if __name__ == '__main__':
    unittest.main()
