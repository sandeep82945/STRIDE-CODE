"""Four-arm comparison, preserving the original local/uniform/GRPO rules.

Single-step STRIDE collapses each eligible response's mapped raw hidden states
into one embedding, then uses the same conditional-variance kernel and positive
transform. It does not average precomputed step novelty or match its mass to
local STRIDE: changing representation can change the novelty magnitude.
"""
import math
import torch

from credit import compute_group_credit as original_credit, _production
from runtime_config import ARMS


@torch.no_grad()
def compute_group_credit(answer_correct, step_embeddings, step_spans,
                         step_eligible, whole_embeddings, response_mask, *,
                         pooled_step_embeddings, alpha=1., novelty_reference_scope='correct_only'):
    original = original_credit(answer_correct, step_embeddings, step_spans,
                               step_eligible, whole_embeddings, response_mask, alpha=alpha,
                               novelty_reference_scope=novelty_reference_scope)
    main, mapped = original['main'], original['mapped_mask']
    pooled = torch.as_tensor(pooled_step_embeddings, device=main.device).float()
    if pooled.shape != torch.as_tensor(whole_embeddings).shape or not bool(torch.isfinite(pooled).all()):
        raise ValueError('Require finite K by D pooled mapped-token embeddings')
    eligible = mapped.any(1).nonzero(as_tuple=True)[0].tolist()
    single = torch.zeros_like(main)
    novelty = {}
    if len(eligible) >= 2:
        production = _production()
        eps = production.STRIDE_EPS
        phi = production._normalized_phi([x[None, :] for x in pooled], eligible,
                                         [1] * len(eligible), main.device)
        gram = phi @ phi.T
        offsets = list(range(len(eligible) + 1))
        for local_i, i in enumerate(eligible):
            v = production._conditional_variances(gram, offsets, len(eligible), local_i, eps)
            u = (v - eps).clamp(min=0., max=1.)
            u = torch.where(u <= eps, torch.zeros_like(u), u)
            novelty[str(i)] = float(u[0])
            single[i, mapped[i]] = float(alpha) * u[0]
    if not bool(torch.isfinite(single).all()) or bool((single < 0).any()):
        raise ValueError('Invalid single-step positive credit')
    auxiliary = {arm: original['auxiliary'][arm] for arm in ARMS if arm != 'single_step'}
    auxiliary['single_step'] = single
    advantages = {arm: main + auxiliary[arm] for arm in ARMS}
    diagnostics = dict(original['diagnostics'])
    diagnostics.pop('dqo_marginals', None)
    stats = {arm: original['diagnostics']['arms'][arm] for arm in ARMS if arm != 'single_step'}
    denominator = int(response_mask.sum())
    total = advantages['single_step']
    stats['single_step'] = {
        'aux_rms': math.sqrt(float(single.double().square().sum()) / denominator),
        'total_rms': math.sqrt(float(total.double().square().sum()) / denominator),
        'aux_mass': float(single.double().sum()),
        'response_aux_mass': single.double().sum(1).tolist(),
        'aux_positive_tokens': int((single > 0).sum()),
        'aux_negative_tokens': int((single < 0).sum()),
        'main_opposed_tokens': int((response_mask & (main * single < 0)).sum()),
        'main_reversed_tokens': int((response_mask & (main * total < 0)).sum()),
        'correct_response_negative_aux_tokens': 0,
    }
    diagnostics.update(arms=stats, single_step_u=novelty,
                       single_step_positive_bonus_nonzero=bool(single.any()),
                       local_single_maxdiff=float((auxiliary['local_positive'] - single).abs().max()),
                       single_step_pooling='raw_hidden_mean_over_same_mapped_tokens',
                       single_step_mass_matched_to_local=False)
    return dict(main=main, auxiliary=auxiliary, advantages=advantages,
                mapped_mask=mapped, diagnostics=diagnostics)
