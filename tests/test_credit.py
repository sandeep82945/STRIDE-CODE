"""Math checks on supplied features; these do not validate semantic diversity."""
import hashlib
from pathlib import Path
import unittest
import torch
from credit import compute_group_credit


def fixture():
    eye = torch.eye(12)
    features = [eye[[0, 1]], eye[[0, 1]]]
    features += [eye[[2 + 2*i, 3 + 2*i]] for i in range(5)]
    features += [eye[[0, 1]]]
    return dict(answer_correct=[1]*7 + [0], step_embeddings=features,
                step_spans=[[(0, 1), (3, 5)]]*8, step_eligible=[True]*8,
                whole_embeddings=torch.stack([x.mean(0) for x in features]),
                response_mask=torch.ones(8, 7, dtype=torch.bool))


class CreditTests(unittest.TestCase):
    def test_kernel_and_credit_sources_preserved(self):
        root = Path(__file__).resolve().parents[1] / 'training'
        for name, expected in {
            'production_stride.py': 'a547f1b13423011c68b061d3ea93dffbbe97a6fe896ce709c2310a478a22ab0b',
            'credit.py': '8c89f0419c142171dc5a76ebdcb7f9444aacfe91706f5c09e6cade594482df93',
        }.items():
            self.assertEqual(hashlib.sha256((root/name).read_bytes()).hexdigest(), expected)

    def test_centering_can_reverse_correctness_at_alpha_one(self):
        result = compute_group_credit(**fixture(), alpha=1.)
        self.assertGreater(float(result['main'][0, 0]), 0)
        self.assertLess(float(result['advantages']['centered'][0, 0]), 0)
        self.assertGreater(float(result['advantages']['local_positive'][0, 0]), 0)
        self.assertGreater(result['diagnostics']['arms']['centered']['main_reversed_tokens'], 0)

    def test_positive_uniform_mass_support_and_nonnegativity(self):
        for alpha in (0., .5, 1., 2.):
            result = compute_group_credit(**fixture(), alpha=alpha)
            mapped = result['mapped_mask']
            local = result['auxiliary']['local_positive']
            uniform = result['auxiliary']['token_uniform']
            self.assertTrue(bool((local >= 0).all() and (uniform >= 0).all()))
            self.assertFalse(bool(local[~mapped].any() or uniform[~mapped].any()))
            self.assertTrue(torch.allclose(local.sum(1), uniform.sum(1), atol=1e-6, rtol=1e-6))
            self.assertEqual(result['diagnostics']['arms']['local_positive']['main_opposed_tokens'], 0)
            self.assertEqual(result['diagnostics']['arms']['token_uniform']['main_opposed_tokens'], 0)

    def test_inactive_gate_is_accuracy_only(self):
        x = fixture()
        x['answer_correct'] = [1] + [0]*7
        result = compute_group_credit(**x)
        self.assertFalse(result['diagnostics']['gate_active'])
        for arm in ('local_positive', 'token_uniform', 'centered'):
            self.assertFalse(bool(result['auxiliary'][arm].any()))
            self.assertTrue(torch.equal(result['advantages'][arm], result['main']))

    def test_all_rollouts_are_references_but_only_correct_receive_bonus(self):
        eye = torch.eye(4)
        x = dict(answer_correct=[1, 0, 1],
                 step_embeddings=[eye[[0]], eye[[0]], eye[[1]]],
                 step_spans=[[(0, 1)]] * 3,
                 step_eligible=[True] * 3,
                 whole_embeddings=torch.stack([eye[0], eye[0], eye[1]]),
                 response_mask=torch.ones(3, 2, dtype=torch.bool))
        correct_only = compute_group_credit(**x, novelty_reference_scope='correct_only')
        all_rollouts = compute_group_credit(**x, novelty_reference_scope='all_rollouts')
        for arm in ('local_positive', 'token_uniform'):
            self.assertFalse(bool(all_rollouts['auxiliary'][arm][1].any()))
        self.assertFalse(torch.equal(correct_only['auxiliary']['local_positive'][0],
                                    all_rollouts['auxiliary']['local_positive'][0]))
        self.assertEqual(all_rollouts['diagnostics']['novelty_reference_responses'], 3)
        self.assertEqual(all_rollouts['diagnostics']['incorrect_reference_responses'], 1)

    def test_all_rollouts_support_one_correct_response(self):
        eye = torch.eye(3)
        x = dict(answer_correct=[1, 0], step_embeddings=[eye[[0]], eye[[1]]],
                 step_spans=[[(0, 1)]] * 2, step_eligible=[True, True],
                 whole_embeddings=torch.stack([eye[0], eye[1]]),
                 response_mask=torch.ones(2, 2, dtype=torch.bool))
        old = compute_group_credit(**x, novelty_reference_scope='correct_only')
        new = compute_group_credit(**x, novelty_reference_scope='all_rollouts')
        self.assertFalse(old['diagnostics']['gate_active'])
        self.assertTrue(new['diagnostics']['gate_active'])
        self.assertTrue(bool(new['auxiliary']['local_positive'][0].any()))
        self.assertFalse(bool(new['auxiliary']['local_positive'][1].any()))


if __name__ == '__main__':
    unittest.main()
