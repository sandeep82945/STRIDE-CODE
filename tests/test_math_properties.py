"""Independent numerical oracles for STRIDE credit and the clipped actor loss.

Randomized tests use fixed seeds. They test supplied features, not whether an
embedding represents a genuinely distinct or correct mathematical strategy.
"""
import math
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

import credit
import production_stride as production
from experiment_credit import compute_group_credit
from feature_pooling import pool_reference_hidden
from trainer import loss_terms


def random_group(seed, k=5, dimension=9):
    rng = torch.Generator().manual_seed(seed)
    hidden = [torch.randn(15, dimension, generator=rng) for _ in range(k)]
    spans = [[(0, 1), (3, 6), (9, 11)] for _ in range(k)]
    pooled = [pool_reference_hidden(h, s, include_pooled=True)
              for h, s in zip(hidden, spans)]
    mask = torch.ones(k, 15, dtype=torch.bool)
    mask[:, 13:] = False
    return dict(answer_correct=[1] * (k-1) + [0],
                step_embeddings=[p[1] for p in pooled], step_spans=spans,
                step_eligible=[True] * k,
                whole_embeddings=torch.stack([p[0] for p in pooled]),
                response_mask=mask,
                pooled_step_embeddings=torch.stack([p[2] for p in pooled]))


def schur_residual(target, conditioning, eps):
    """Ridge residual via a linear solve, independent of the Cholesky routine."""
    if len(conditioning) == 0:
        return target.dot(target)
    covariance = conditioning @ conditioning.T
    cross = conditioning @ target
    return target.dot(target) - cross.dot(torch.linalg.solve(
        covariance + eps * torch.eye(len(conditioning), dtype=torch.float64), cross))


class KernelProperties(unittest.TestCase):
    def test_chain_matches_independent_schur_solves_and_eigen_logdet(self):
        for seed in range(12):
            with self.subTest(seed=seed):
                rng = torch.Generator().manual_seed(seed)
                embeddings = [torch.randn(n, 7, generator=rng, dtype=torch.float64)
                              for n in (2, 3, 1)]
                normalized = [x / torch.linalg.vector_norm(x, dim=1, keepdim=True)
                              for x in embeddings]
                result = production.exact_chain_credits(embeddings, [0, 1, 2])
                all_features = torch.cat(normalized)
                all_eigenvalues = torch.linalg.eigvalsh(all_features @ all_features.T)
                full_logdet = torch.log1p(all_eigenvalues / production.STRIDE_EPS).sum()
                for i in range(3):
                    others = torch.cat([x for j, x in enumerate(normalized) if j != i])
                    expected = []
                    for j, target in enumerate(normalized[i]):
                        conditioning = torch.cat([others, normalized[i][:j]])
                        residual = schur_residual(target, conditioning, production.STRIDE_EPS)
                        expected.append(torch.log1p(residual / production.STRIDE_EPS))
                    self.assertTrue(torch.allclose(result[i], torch.stack(expected), atol=1e-10, rtol=1e-10))
                    other_eigenvalues = torch.linalg.eigvalsh(others @ others.T)
                    other_logdet = torch.log1p(other_eigenvalues / production.STRIDE_EPS).sum()
                    self.assertAlmostEqual(float(result[i].sum()), float(full_logdet-other_logdet), places=9)

    def test_conditional_variance_decreases_with_more_conditioning(self):
        rng = torch.Generator().manual_seed(97)
        phi = F.normalize(torch.randn(8, 11, generator=rng, dtype=torch.float64), dim=1)
        previous = float('inf')
        for count in range(1, 8):
            ordered = torch.cat([phi[:count], phi[-1:]])
            gram = ordered @ ordered.T
            actual = production._conditional_variances(gram, [0, count, count+1], 2, 1, .001)[0]
            self.assertLessEqual(float(actual), previous + 1e-12)
            expected = .001 + schur_residual(phi[-1], phi[:count], .001)
            self.assertAlmostEqual(float(actual), float(expected), places=11)
            previous = float(actual)

    def test_rotation_scaling_and_rollout_permutation_invariance(self):
        rng = torch.Generator().manual_seed(83)
        features = [torch.randn(n, 9, generator=rng, dtype=torch.float64) for n in (2, 1, 3)]
        rotation, _ = torch.linalg.qr(torch.randn(9, 9, generator=rng, dtype=torch.float64))
        expected = production.exact_chain_credits(features, [0, 1, 2])
        transformed = [(x @ rotation) * (i+2) for i, x in enumerate(features)]
        actual = production.exact_chain_credits(transformed, [2, 0, 1])
        for i in range(3):
            self.assertTrue(torch.allclose(expected[i], actual[i], atol=1e-11, rtol=1e-11))

    def test_causal_step_order_changes_allocation_but_preserves_total(self):
        features = [torch.tensor([[1., 0, 0], [1., 1, 0]], dtype=torch.float64),
                    torch.tensor([[0., 0, 1]], dtype=torch.float64)]
        forward = production.exact_chain_credits(features, [0, 1])[0]
        swapped = [features[0].flip(0), features[1]]
        backward = production.exact_chain_credits(swapped, [0, 1])[0]
        self.assertAlmostEqual(float(forward.sum()), float(backward.sum()), places=11)
        self.assertFalse(torch.allclose(forward, backward.flip(0)))

    def test_zero_empty_identity_and_rank_one_logdet_and_rank(self):
        for n in (0, 1, 5):
            zero = torch.zeros(n, n)
            self.assertEqual(production.log_det_F(zero), 0.)
            self.assertEqual(production.effective_rank(zero), 0.)
            identity = torch.eye(n)
            self.assertAlmostEqual(production.log_det_F(identity), n*math.log1p(1000), places=10)
            self.assertAlmostEqual(production.effective_rank(identity), float(n), places=10)
        rank_one = torch.ones(7, 7, dtype=torch.float64)
        self.assertAlmostEqual(production.log_det_F(rank_one), math.log1p(7000), places=10)
        self.assertAlmostEqual(production.effective_rank(rank_one), 1., places=10)
        self.assertAlmostEqual(production.effective_rank(rank_one * 1e-10), 1., places=10)

    def test_empty_and_single_eligible_chain_are_inactive(self):
        features = [torch.empty(0, 3), torch.eye(3)]
        self.assertEqual(production.exact_chain_credits(features, []), {})
        self.assertEqual(production.exact_chain_credits(features, [0, 1]), {})

    def test_one_rollout_conditional_helper_and_normalization_dtype(self):
        embeddings = [torch.tensor([[3., 4.], [0., 0.]])]
        phi = production._normalized_phi(embeddings, [0], [2], torch.device('cpu'), dtype=torch.float64)
        self.assertEqual(phi.dtype, torch.float64)
        self.assertTrue(torch.equal(phi, torch.tensor([[.6, .8], [0., 0.]], dtype=torch.float64)))
        variances = production._conditional_variances(phi @ phi.T, [0, 2], 1, 0, .001)
        self.assertTrue(torch.allclose(variances, torch.tensor([1.001, .001], dtype=torch.float64)))

    def test_strict_and_nonstrict_span_handling_and_statistics(self):
        features = [torch.eye(3)[:2], torch.eye(3)[1:]]
        mask = torch.ones(2, 7, dtype=torch.bool)
        for bad_spans in ([(-1, 1), (2, 3)], [(0, 3), (3, 4)], [(0, 1), (3, 7)]):
            with self.subTest(spans=bad_spans):
                stats = {}
                with self.assertRaisesRegex(ValueError, 'invalid STRIDE'):
                    production.stride_group_diversity_advantage(features, [bad_spans, [(0, 1), (3, 4)]], [0, 1], mask)
                output = production.stride_group_diversity_advantage(features, [bad_spans, [(0, 1), (3, 4)]], [0, 1], mask, strict=False, stats=stats)
                self.assertFalse(bool(output.any()))
                self.assertEqual(stats['invalid_span'], 1.)
                self.assertEqual(stats['group_active'], 0.)
        stats = {}
        output = production.stride_group_diversity_advantage(features, [[(0, 1), (3, 4)]]*2, [0, 1], mask, stats=stats)
        self.assertAlmostEqual(float(output.sum()), 0., places=6)
        self.assertEqual(stats['group_active'], 1.)
        self.assertTrue(all(math.isfinite(value) for value in stats.values()))
        self.assertAlmostEqual(stats['token_coverage'], 8/14)

    def test_threshold_can_disable_every_novelty_bonus(self):
        stats = {}
        output = production.stride_group_diversity_advantage(
            [torch.eye(2)[:1], torch.eye(2)[1:]], [[(0, 1)]]*2, [0, 1],
            torch.ones(2, 2, dtype=torch.bool), tau=1., stats=stats)
        self.assertFalse(bool(output.any()))
        self.assertEqual(stats['u_zero_frac'], 1.)


class GroupCreditProperties(unittest.TestCase):
    def test_randomized_mass_support_sign_and_independent_step_credit(self):
        for seed in range(15):
            with self.subTest(seed=seed):
                x = random_group(seed)
                alpha = (0., .5, 1., 2., 3.)[seed % 5]
                output = compute_group_credit(**x, alpha=alpha)
                mask = output['mapped_mask']
                local = output['auxiliary']['local_positive']
                uniform = output['auxiliary']['token_uniform']
                self.assertTrue(torch.allclose(local.double().sum(1), uniform.double().sum(1), atol=5e-6, rtol=1e-6))
                for arm, aux in output['auxiliary'].items():
                    self.assertTrue(bool(torch.isfinite(aux).all()))
                    self.assertFalse(bool(aux[~mask].any()))
                    self.assertTrue(bool(((aux >= 0) & (aux <= alpha)).all()))
                    self.assertEqual(output['diagnostics']['arms'][arm]['main_opposed_tokens'], 0)
                normalized = [F.normalize(f.double(), dim=1) for f in x['step_embeddings'][:-1]]
                for i, own in enumerate(normalized):
                    others = torch.cat([v for j, v in enumerate(normalized) if i != j])
                    for j, target in enumerate(own):
                        residual = float(schur_residual(target, torch.cat([others, own[:j]]), .001))
                        expected = 0. if residual <= .001 else alpha * min(max(residual, 0.), 1.)
                        start, end = x['step_spans'][i][j]
                        # Production uses float32 Cholesky on a ridge matrix;
                        # the independent float64 solve is a higher-precision
                        # oracle. Allow 1% of the 1e-3 novelty threshold per alpha.
                        self.assertTrue(torch.allclose(local[i, start:end+1], torch.full_like(local[i, start:end+1], expected), atol=1e-5*max(alpha, 1.), rtol=0))

    def test_single_step_matches_multiresponse_schur_oracle(self):
        for seed in range(10):
            x = random_group(seed, k=6, dimension=4)
            output = compute_group_credit(**x, alpha=.7)
            phi = F.normalize(x['pooled_step_embeddings'][:-1].double(), dim=1)
            for i, target in enumerate(phi):
                condition = torch.cat([phi[:i], phi[i+1:]])
                residual = float(schur_residual(target, condition, .001))
                expected = 0. if residual <= .001 else .7*min(max(residual, 0.), 1.)
                actual = output['auxiliary']['single_step'][i, output['mapped_mask'][i]]
                self.assertTrue(torch.allclose(actual, torch.full_like(actual, expected), atol=1e-5, rtol=0))

    def test_all_correct_all_wrong_and_alpha_zero(self):
        for reward in (0, 1):
            x = random_group(37)
            x['answer_correct'] = [reward] * 5
            output = compute_group_credit(**x)
            self.assertFalse(bool(output['main'].any()))
            self.assertEqual(output['diagnostics']['gate_active'], bool(reward))
            for arm in ('local_positive', 'single_step', 'token_uniform'):
                self.assertEqual(bool(output['auxiliary'][arm].any()), bool(reward))
            zero = compute_group_credit(**x, alpha=0.)
            self.assertTrue(all(not bool(value.any()) for value in zero['advantages'].values()))

    def test_zero_duplicate_and_nearly_collinear_features_stay_finite(self):
        for mode in ('zero', 'duplicate', 'nearly_collinear'):
            x = random_group(4, k=8, dimension=12)
            common = torch.arange(1, 13, dtype=torch.float32)
            for i in range(8):
                f = torch.zeros(3, 12) if mode == 'zero' else common.repeat(3, 1)
                if mode == 'nearly_collinear':
                    f[i % 3, i] += 1e-4
                x['step_embeddings'][i] = f
                x['whole_embeddings'][i] = f.mean(0)
                x['pooled_step_embeddings'][i] = f.mean(0)
            output = compute_group_credit(**x)
            self.assertTrue(all(bool(torch.isfinite(v).all()) for v in output['advantages'].values()))
            for arm in ('local_positive', 'single_step', 'token_uniform'):
                self.assertFalse(bool(output['auxiliary'][arm].any()), mode)

    def test_ineligible_or_wrong_features_do_not_change_stride_credit(self):
        x = random_group(65)
        x['step_eligible'][1] = False
        expected = compute_group_credit(**x)
        for i in (1, 4):
            x['step_embeddings'][i] *= 0
            x['whole_embeddings'][i] *= 0
            x['pooled_step_embeddings'][i] *= 0
        actual = compute_group_credit(**x)
        for arm in actual['advantages']:
            self.assertTrue(torch.equal(actual['advantages'][arm], expected['advantages'][arm]), arm)

    def test_group_permutation_is_equivariant(self):
        x = random_group(121)
        expected = compute_group_credit(**x)
        order = [4, 2, 0, 3, 1]
        permuted = {key: value[order] if isinstance(value, torch.Tensor)
                    else [value[i] for i in order] for key, value in x.items()}
        actual = compute_group_credit(**permuted)
        for arm in actual['advantages']:
            self.assertTrue(torch.allclose(actual['advantages'][arm], expected['advantages'][arm][order], atol=3e-6, rtol=0), arm)

    def test_invalid_group_inputs_fail_before_training(self):
        cases = []
        for alpha in (-1., float('nan'), float('inf'), True):
            cases.append((f'alpha={alpha}', {}, alpha))
        cases.extend([
            ('float mask', {'response_mask': torch.ones(5, 15)}, 1.),
            ('rank-one mask', {'response_mask': torch.ones(5, dtype=torch.bool)}, 1.),
            ('zero width', {'response_mask': torch.ones(5, 0, dtype=torch.bool)}, 1.),
            ('dimension mismatch', {'answer_correct': [1, 1]}, 1.),
            ('nonbinary rewards', {'answer_correct': [1, .5, 1, 1, 0]}, 1.),
            ('nan rewards', {'answer_correct': [1, float('nan'), 1, 1, 0]}, 1.),
            ('bad pooled dimension', {'pooled_step_embeddings': torch.ones(5, 2)}, 1.),
            ('nonfinite pooled', {'pooled_step_embeddings': torch.full((5, 9), float('nan'))}, 1.),
            ('nonfinite whole', {'whole_embeddings': torch.full((5, 9), float('inf'))}, 1.),
        ])
        for name, changes, alpha in cases:
            with self.subTest(case=name):
                x = random_group(18)
                x.update(changes)
                with self.assertRaises(ValueError):
                    compute_group_credit(**x, alpha=alpha)
        for mutation in ('padded span', 'overlap', 'negative', 'float', 'boolean', 'empty', 'feature count', 'nonfinite feature', 'empty response'):
            with self.subTest(mutation=mutation):
                x = random_group(18)
                if mutation == 'padded span':
                    x['step_spans'][0][-1] = (9, 14)
                elif mutation == 'overlap':
                    x['step_spans'][0][1] = (1, 6)
                elif mutation == 'negative':
                    x['step_spans'][0][0] = (-1, 1)
                elif mutation == 'float':
                    x['step_spans'][0][0] = (0.5, 1)
                elif mutation == 'boolean':
                    x['step_spans'][0][0] = (False, 1)
                elif mutation == 'empty':
                    x['step_spans'][0] = []
                    x['step_embeddings'][0] = torch.empty(0, 9)
                elif mutation == 'feature count':
                    x['step_embeddings'][0] = torch.ones(2, 9)
                elif mutation == 'nonfinite feature':
                    x['step_embeddings'][0][0, 0] = float('nan')
                elif mutation == 'empty response':
                    x['response_mask'][0] = False
                with self.assertRaises(ValueError):
                    compute_group_credit(**x)

    def test_original_determinant_marginals_match_inverse_diagonal_identity(self):
        rng = torch.Generator().manual_seed(89)
        for n in (1, 2, 7):
            features = torch.randn(n, 5, generator=rng)
            normalized = F.normalize(features.double(), dim=1)
            matrix = torch.eye(n, dtype=torch.float64) + normalized @ normalized.T
            expected = -torch.linalg.inv(matrix).diagonal().log()
            self.assertTrue(torch.allclose(credit.dqo_marginals(features).double(), expected, atol=8e-7, rtol=0))

    def test_standardize_uses_sample_variance_and_ties_are_zero(self):
        x = torch.tensor([0., 0., 1., 1.], dtype=torch.float64)
        expected = torch.tensor([-.5, -.5, .5, .5], dtype=torch.float64)/(math.sqrt(1/3)+1e-6)
        self.assertTrue(torch.equal(credit.standardize(x), expected))
        self.assertTrue(torch.equal(credit.standardize(torch.ones(4)), torch.zeros(4)))

    def test_pinned_kernel_tampering_is_rejected_before_import(self):
        credit._production.cache_clear()
        try:
            with patch('credit.Path.read_bytes', return_value=b'not the pinned kernel'):
                with self.assertRaisesRegex(ValueError, 'Pinned production STRIDE source changed'):
                    credit._production()
        finally:
            credit._production.cache_clear()
        self.assertEqual(credit._production().STRIDE_EPS, .001)

    def test_large_redundant_group_has_finite_zero_novelty(self):
        k, n, dimension = 8, 32, 16
        features = torch.arange(1, dimension+1, dtype=torch.float32).repeat(n, 1)
        spans = [[(2*j, 2*j) for j in range(n)] for _ in range(k)]
        result = compute_group_credit(
            answer_correct=[1]*k, step_embeddings=[features.clone() for _ in range(k)],
            step_spans=spans, step_eligible=[True]*k,
            whole_embeddings=features[:k], response_mask=torch.ones(k, 2*n, dtype=torch.bool),
            pooled_step_embeddings=features[:k])
        self.assertTrue(all(bool(torch.isfinite(x).all()) for x in result['advantages'].values()))
        self.assertTrue(all(not bool(x.any()) for x in result['advantages'].values()))


class PoolingProperties(unittest.TestCase):
    def test_randomized_union_pool_is_partition_invariant(self):
        for seed in range(15):
            rng = torch.Generator().manual_seed(seed)
            hidden = torch.randn(17, 8, generator=rng)
            spans = [(1, 6), (9, 13)]
            split = [(1, 1), (2, 4), (5, 6), (9, 11), (12, 13)]
            whole, steps, pooled = pool_reference_hidden(hidden, spans, include_pooled=True)
            expected = F.normalize(hidden[[1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13]].double().mean(0), dim=0)
            self.assertTrue(torch.allclose(pooled.double(), expected, atol=2e-7, rtol=0))
            split_pooled = pool_reference_hidden(hidden, split, include_pooled=True)[2]
            self.assertTrue(torch.allclose(pooled, split_pooled, atol=2e-7, rtol=0))
            hidden[0] = 1e5
            hidden[7:9] = -1e5
            hidden[14:] = 1e5
            changed = pool_reference_hidden(hidden, spans, include_pooled=True)
            self.assertTrue(torch.equal(changed[2], pooled))
            self.assertTrue(torch.equal(changed[1], steps))
            self.assertFalse(torch.equal(changed[0], whole))

    def test_empty_zero_and_cancelling_hidden_features(self):
        hidden = torch.zeros(4, 3)
        for spans in ([], [(0, 3)]):
            result = pool_reference_hidden(hidden, spans, include_pooled=True)
            self.assertEqual(result[1].shape, (len(spans), 3))
            self.assertTrue(all(not bool(x.any()) for x in result))
        hidden = torch.tensor([[1., 0], [-1., 0]])
        self.assertFalse(bool(pool_reference_hidden(hidden, [(0, 0), (1, 1)], include_pooled=True)[2].any()))

    def test_span_iterable_is_not_consumed_before_pooling(self):
        hidden = torch.arange(18, dtype=torch.float32).reshape(6, 3)
        spans = [(0, 1), (4, 5)]
        expected = pool_reference_hidden(hidden, spans, include_pooled=True)
        actual = pool_reference_hidden(hidden, iter(spans), include_pooled=True)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(actual, expected)))

    def test_invalid_hidden_and_span_inputs_raise_value_error(self):
        for hidden in (torch.empty(0, 3), torch.empty(3, 0), torch.zeros(3),
                       torch.ones(3, 3, dtype=torch.long), [[1., 2.]],
                       torch.full((3, 3), float('nan')), torch.full((3, 3), float('inf'))):
            with self.subTest(hidden=hidden):
                with self.assertRaises(ValueError):
                    pool_reference_hidden(hidden, [])
        for spans in (None, [(0,)], [(0, 1, 2)], [2], [(0., 1)], [(False, 1)],
                      [(-1, 1)], [(1, 0)], [(0, 3)], [(0, 1), (1, 2)], [(2, 2), (0, 0)]):
            with self.subTest(spans=spans):
                with self.assertRaises(ValueError):
                    pool_reference_hidden(torch.ones(3, 3), spans, include_pooled=True)


class ClippedLossProperties(unittest.TestCase):
    def test_piecewise_clipped_loss_and_gradient_against_scalar_oracle(self):
        ratios = [.3, .7, .9, 1., 1.1, 1.3, 2.]
        for advantage in (-2., 0., 3.):
            old = torch.full((len(ratios),), -2., dtype=torch.float64)
            new = (old + torch.tensor(ratios, dtype=torch.float64).log()).requires_grad_()
            loss, difference, flags = loss_terms(new, old, torch.full_like(new, advantage), clip=.2)
            expected_loss = 0.
            expected_gradient = []
            for ratio in ratios:
                bound = min(max(ratio, .8), 1.2)
                expected_loss -= min(ratio*advantage, bound*advantage)
                clipped = (advantage > 0 and ratio > 1.2) or (advantage < 0 and ratio < .8)
                expected_gradient.append(0. if clipped else -advantage*ratio)
            self.assertAlmostEqual(float(loss.detach()), expected_loss, places=11)
            loss.backward()
            self.assertTrue(torch.allclose(new.grad, torch.tensor(expected_gradient, dtype=torch.float64), atol=1e-12, rtol=0))
            self.assertFalse(difference.requires_grad)
            self.assertEqual(flags.tolist(), [abs(ratio-1) > .2 for ratio in ratios])

    def test_randomized_loss_gradient_matches_finite_differences(self):
        rng = torch.Generator().manual_seed(188)
        old = torch.randn(24, generator=rng, dtype=torch.float64)
        offsets = torch.randn(24, generator=rng, dtype=torch.float64) * .4
        new = (old + offsets).requires_grad_()
        advantages = torch.randn(24, generator=rng, dtype=torch.float64)
        loss_terms(new, old, advantages)[0].backward()
        expected = []
        for i in range(24):
            plus, minus = new.detach().clone(), new.detach().clone()
            plus[i] += 1e-6
            minus[i] -= 1e-6
            expected.append((float(loss_terms(plus, old, advantages)[0])-float(loss_terms(minus, old, advantages)[0]))/(2e-6))
        self.assertTrue(torch.allclose(new.grad, torch.tensor(expected, dtype=torch.float64), atol=2e-9, rtol=0))

    def test_extreme_log_ratios_are_finite_and_stably_clamped(self):
        new = torch.tensor([-1000., 1000., -1000., 1000.], dtype=torch.float64, requires_grad=True)
        advantages = torch.tensor([1., 1., -1., -1.], dtype=torch.float64)
        loss = loss_terms(new, torch.zeros(4, dtype=torch.float64), advantages)[0]
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        self.assertTrue(torch.equal(new.grad, torch.zeros_like(new)))

    def test_nonfinite_log_probabilities_are_rejected(self):
        for new, old in ((float('nan'), 0.), (float('inf'), 0.), (0., -float('inf')), (float('inf'), float('inf'))):
            with self.assertRaisesRegex(ValueError, 'Nonfinite'):
                loss_terms(torch.tensor([new]), torch.tensor([old]), torch.ones(1))


if __name__ == '__main__':
    unittest.main()
