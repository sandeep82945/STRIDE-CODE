#!/usr/bin/env python3
"""Allocated-GPU invariant check for all-reference, correct-only STRIDE."""
import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training'))
from credit import compute_group_credit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    device = torch.device('cuda:0')
    eye = torch.eye(4, device=device)
    values = dict(answer_correct=[1, 0, 1],
                  step_embeddings=[eye[[0]], eye[[0]], eye[[1]]],
                  step_spans=[[(0, 1)]] * 3,
                  step_eligible=[True] * 3,
                  whole_embeddings=torch.stack([eye[0], eye[0], eye[1]]),
                  response_mask=torch.ones(3, 2, dtype=torch.bool, device=device))
    old = compute_group_credit(**values, novelty_reference_scope='correct_only')
    new = compute_group_credit(**values, novelty_reference_scope='all_rollouts')
    for arm in ('local_positive', 'token_uniform'):
        assert not bool(new['auxiliary'][arm][1].any())
        assert torch.allclose(new['auxiliary'][arm].sum(1)[[0, 2]],
                              new['auxiliary']['local_positive'].sum(1)[[0, 2]])
    assert not torch.equal(old['auxiliary']['local_positive'][0],
                           new['auxiliary']['local_positive'][0])
    diagnostics = new['diagnostics']
    assert diagnostics['novelty_reference_scope'] == 'all_rollouts'
    assert diagnostics['novelty_reference_responses'] == 3
    assert diagnostics['incorrect_reference_responses'] == 1
    result = dict(passed=True, device=str(device), incorrect_auxiliary_mass=0.,
                  reference_responses=3, correct_bonus_responses=2,
                  token_uniform_mass_preserved=True,
                  incorrect_reference_changes_correct_novelty=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
