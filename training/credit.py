"""Five credit rules on identical supplied features; no text/GT feature extraction.

Original token spans are inclusive. ``step_eligible`` describes the caller's
syntax/mapping gate, separately from binary answer correctness. DQO uses all
responses. This is a shared-reference DQO-style control, not a replication of
the paper's learned quality reward or external embedding model.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import importlib.util
import math
from pathlib import Path

import torch
import torch.nn.functional as F

ARMS = ('grpo', 'local_positive', 'token_uniform', 'centered', 'dqo')
PRODUCTION_SHA256 = 'a547f1b13423011c68b061d3ea93dffbbe97a6fe896ce709c2310a478a22ab0b'


@lru_cache(maxsize=1)
def _production():
    path = Path(__file__).with_name('production_stride.py')
    if hashlib.sha256(path.read_bytes()).hexdigest() != PRODUCTION_SHA256:
        raise ValueError('Pinned production STRIDE source changed')
    spec = importlib.util.spec_from_file_location('qwen4b_pilot_pinned_stride', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def standardize(reward):
    """Prior pilot GRPO convention: sample standard deviation and 1e-6 ridge."""
    return (reward - reward.mean()) / (reward.std(unbiased=True) + 1e-6)


def dqo_marginals(whole_embeddings):
    x = F.normalize(whole_embeddings.float(), p=2, dim=1)
    matrix = x @ x.T + torch.eye(len(x), device=x.device)
    sign, full = torch.linalg.slogdet(matrix)
    if not bool(sign > 0):
        raise ValueError('Nonpositive DQO determinant')
    out = []
    for i in range(len(x)):
        keep = torch.arange(len(x), device=x.device) != i
        sign, minor = torch.linalg.slogdet(matrix[keep][:, keep])
        if not bool(sign > 0):
            raise ValueError('Nonpositive DQO minor')
        out.append(full - minor)
    return torch.stack(out)


@torch.no_grad()
def compute_group_credit(answer_correct, step_embeddings, step_spans,
                         step_eligible, whole_embeddings, response_mask, *, alpha=1.,
                         novelty_reference_scope='correct_only'):
    """Return main, auxiliary/advantages dictionaries, mask and diagnostics.

    All inputs refer to ONE question's complete rollout group, in common order.
    The returned tensors are detached float32 on response_mask.device.
    local_positive is alpha*U, broadcasting raw bounded novelty per step token.
    token_uniform preserves that mass within each response's same mapped tokens.
    centered is the exact pinned production routine, including its token-weighted
    baseline. DQO standardizes accuracy+alpha*whole-response marginal jointly.
    There is no RMS matching, length division for local U, or other rescaling.
    """
    if isinstance(alpha, bool) or not math.isfinite(float(alpha)) or alpha < 0:
        raise ValueError('alpha must be finite and nonnegative')
    alpha = float(alpha)
    if novelty_reference_scope not in ('correct_only', 'all_rollouts'):
        raise ValueError('Unknown novelty reference scope')
    if response_mask.ndim != 2 or response_mask.dtype != torch.bool:
        raise ValueError('response_mask must be a rank-two boolean tensor')
    k, width = response_mask.shape
    if k < 2 or width < 1 or not bool(response_mask.any(1).all()):
        raise ValueError('Need at least two nonempty responses')
    if not (len(answer_correct) == len(step_embeddings) == len(step_spans) == len(step_eligible) == k):
        raise ValueError('Group dimensions disagree')
    device = response_mask.device
    reward = torch.as_tensor(answer_correct, dtype=torch.float32, device=device)
    if reward.shape != (k,) or not bool(((reward == 0) | (reward == 1)).all()):
        raise ValueError('answer_correct must contain binary answer scores')
    whole = torch.as_tensor(whole_embeddings, device=device).float()
    if whole.ndim != 2 or whole.shape[0] != k or whole.shape[1] < 1 or not bool(torch.isfinite(whole).all()):
        raise ValueError('Whole-response features must be finite K by D')
    mapped = torch.zeros_like(response_mask)
    eligible = []
    references = []
    for i, (features, spans) in enumerate(zip(step_embeddings, step_spans)):
        if features.ndim != 2 or features.shape[0] != len(spans) or features.shape[1] != whole.shape[1]:
            raise ValueError('Require one same-dimensional feature per declared step span')
        if not bool(torch.isfinite(features).all()):
            raise ValueError('Nonfinite step features')
        if not bool(step_eligible[i]):
            continue
        if not spans:
            raise ValueError('Eligible response has no mapped step spans')
        last = -1
        for pair in spans:
            if len(pair) != 2:
                raise ValueError('Each inclusive span needs two endpoints')
            a, b = pair
            if isinstance(a, bool) or isinstance(b, bool) or int(a) != a or int(b) != b:
                raise ValueError('Span endpoints must be integer token positions')
            a, b = int(a), int(b)
            if not (last < a <= b < width) or a < 0 or not bool(response_mask[i, a:b+1].all()):
                raise ValueError('Invalid, overlapping or padded eligible span')
            if reward[i] > 0:
                mapped[i, a:b+1] = True
            last = b
        references.append(i)
        if reward[i] > 0:
            eligible.append(i)

    if novelty_reference_scope == 'correct_only':
        references = list(eligible)

    production = _production()
    eps = production.STRIDE_EPS
    centered = production.stride_group_diversity_advantage(
        step_embeddings, step_spans, eligible, response_mask,
        alpha=alpha, eps=eps, strict=True)
    raw_u = torch.zeros_like(centered)
    step_u = {}
    active = len(references) >= 2 and bool(eligible)
    if active:
        sizes = [len(step_spans[i]) for i in references]
        phi = production._normalized_phi(step_embeddings, references, sizes, device)
        gram = phi @ phi.T
        offsets = [0]
        for n in sizes:
            offsets.append(offsets[-1] + n)
        for local_i, i in enumerate(references):
            if i not in eligible:
                continue
            v = production._conditional_variances(gram, offsets, len(references), local_i, eps)
            u = (v - eps).clamp(min=0., max=1.)
            u = torch.where(u <= eps, torch.zeros_like(u), u)
            step_u[str(i)] = u.tolist()
            for j, (a, b) in enumerate(step_spans[i]):
                raw_u[i, a:b+1] = u[j]
        if novelty_reference_scope == 'correct_only':
            reconstruction = torch.zeros_like(raw_u)
            reconstruction[mapped] = alpha * (raw_u[mapped] - raw_u[mapped].mean())
            if not torch.equal(centered, reconstruction):
                raise AssertionError('Positive-U reconstruction differs from pinned centered rule')

    local = alpha * raw_u
    budgets = local.double().sum(1)
    uniform = (budgets / mapped.sum(1).clamp_min(1)).float()[:, None].expand_as(local) * mapped
    mass_error = (uniform.double().sum(1) - budgets).abs()
    if bool((mass_error > 2 * torch.finfo(torch.float32).eps * budgets.clamp_min(1)).any()):
        raise AssertionError('Uniform credit changed per-response coefficient mass')
    main = standardize(reward)[:, None] * response_mask
    div = dqo_marginals(whole)
    dqo = standardize(reward + alpha * div)[:, None] * response_mask
    auxiliary = {'grpo': torch.zeros_like(main), 'local_positive': local,
                 'token_uniform': uniform, 'centered': centered, 'dqo': dqo - main}
    advantages = {arm: main + auxiliary[arm] for arm in ARMS}
    # Preserve the combined-reward result exactly, avoiding subtract/add rounding.
    advantages['dqo'] = dqo
    denominator = int(response_mask.sum())
    arm_stats = {}
    for arm in ARMS:
        aux, total = auxiliary[arm], advantages[arm]
        if not bool(torch.isfinite(total).all()) or bool(total[~response_mask].any()):
            raise AssertionError('Nonfinite advantage or credit on padding')
        conflict = response_mask & (main * aux < 0)
        reversal = response_mask & (main * total < 0)
        arm_stats[arm] = {
            'aux_rms': math.sqrt(float(aux.double().square().sum()) / denominator),
            'total_rms': math.sqrt(float(total.double().square().sum()) / denominator),
            'aux_mass': float(aux.double().sum()),
            'response_aux_mass': aux.double().sum(1).tolist(),
            'aux_positive_tokens': int((aux > 0).sum()),
            'aux_negative_tokens': int((aux < 0).sum()),
            'main_opposed_tokens': int(conflict.sum()),
            'main_reversed_tokens': int(reversal.sum()),
            'correct_response_negative_aux_tokens': int(((reward[:, None] > 0) & (aux < 0)).sum()),
        }
    return {'main': main, 'auxiliary': auxiliary, 'advantages': advantages,
            'mapped_mask': mapped, 'diagnostics': {
                'source_sha256': PRODUCTION_SHA256, 'alpha': alpha, 'eps': eps,
                'correct_responses': int(reward.sum()), 'eligible_responses': len(eligible),
                'novelty_reference_scope': novelty_reference_scope,
                'novelty_reference_responses': len(references),
                'incorrect_reference_responses': len(set(references) - set(eligible)),
                'gate_active': active, 'positive_bonus_nonzero': bool(local.any()),
                'mapped_tokens': int(mapped.sum()), 'step_u': step_u,
                'local_uniform_maxdiff': float((local - uniform).abs().max()),
                'local_uniform_differing_tokens': int(((local - uniform).abs() > 1e-7).sum()),
                'uniform_response_mass_max_error': float(mass_error.max()),
                'main_rms': math.sqrt(float(main.double().square().sum()) / denominator),
                'dqo_marginals': div.tolist(), 'arms': arm_stats}}
