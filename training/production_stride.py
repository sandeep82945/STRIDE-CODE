"""STRIDE final algorithm: step-level token diversity advantage.

Implements the specification exactly:

    phi_ij = e_ij / ||e_ij||_2                      (L2-normalized step embedding)
    G_x    = Phi_x Phi_x^T,  K_x = G_x + eps I      (eps = 1e-3, no adaptive sigma)

For target rollout i, condition on ALL steps of the other correct rollouts
(B_i) plus the causal past of rollout i itself:

    order  = (B_i, s_i1, ..., s_im_i)
    L L^T  = K_x[order, order]
    v_ij   = L[p_ij, p_ij]^2       with p_ij = |B_i| + j

    delta_exact_ij = log(v_ij / eps)                (diagnostic; exact chain rule)
    u_tilde_ij     = clip(v_ij - eps, 0, 1)
    u_ij           = u_tilde_ij * 1[u_tilde_ij > tau],   tau = eps

Token-level advantage, prompt-local, with a TOKEN-WEIGHTED baseline so the
term is exactly mean-zero over the mapped tokens:

    U[i, t] = u_ij for t in T_ij ;  M_x = mapped tokens of correct rollouts
    b_x     = mean(U[M_x])
    A_div[i, t] = alpha * (u_ij - b_x)  on M_x, else 0

Properties guaranteed (and covered by the test suite):
  * |A_div| <= alpha                       since u_ij, b_x in [0, 1]
  * sum over M_x of A_div == 0             (no prompt-local correctness offset)
  * exact duplicates receive u_ij == 0     (thresholded below tau)
  * sum_j delta_exact_ij == F(B_i u S_i) - F(B_i)   (Cholesky chain rule)

There is NO per-group z-score, NO RMS normalization, NO division by the step
count, and NO division by the span length. Those divisions were what made the
diversity term ~1/300 the scale of the GRPO advantage in the previous
implementation; the bounded u_ij in [0, 1] makes alpha directly interpretable
and, unlike a ratio-to-outcome normalization, cannot blow up when the outcome
advantage collapses to zero in all-correct groups.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

__all__ = [
    "STRIDE_EPS",
    "stride_group_diversity_advantage",
    "exact_chain_credits",
    "log_det_F",
    "effective_rank",
]

STRIDE_EPS = 1e-3


def log_det_F(gram: torch.Tensor, eps: float = STRIDE_EPS) -> float:
    """F_eps(A) = log det(I + eps^-1 G_A). Used to verify the chain rule."""
    n = gram.shape[0]
    if n == 0:
        return 0.0
    mat = torch.eye(n, dtype=torch.float64, device=gram.device) + gram.to(torch.float64) / eps
    return float(torch.linalg.slogdet(mat)[1].item())


def effective_rank(gram: torch.Tensor) -> float:
    """Entropy-based effective rank of the Gram matrix (spec logging item)."""
    n = gram.shape[0]
    if n == 0:
        return 0.0
    ev = torch.linalg.eigvalsh(gram.to(torch.float64)).clamp(min=0.0)
    total = float(ev.sum().item())
    if total <= 0.0:
        return 0.0
    p = ev / total
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()).item())


def _normalized_phi(
    step_embeddings: Sequence[torch.Tensor],
    correct: Sequence[int],
    sizes: Sequence[int],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    return torch.cat(
        [
            F.normalize(
                step_embeddings[i][:n].to(device=device, dtype=dtype),
                p=2,
                dim=-1,
            )
            for i, n in zip(correct, sizes)
        ],
        dim=0,
    )


def _conditional_variances(
    gram: torch.Tensor,
    offsets: Sequence[int],
    n_rollouts: int,
    local_i: int,
    eps: float,
) -> torch.Tensor:
    """v_ij for every own step of rollout local_i, conditioned on (B_i, causal past)."""
    device = gram.device
    own = torch.arange(offsets[local_i], offsets[local_i + 1], device=device)
    other_blocks = [
        torch.arange(offsets[r], offsets[r + 1], device=device)
        for r in range(n_rollouts)
        if r != local_i
    ]
    others = (
        torch.cat(other_blocks)
        if other_blocks
        else torch.empty(0, dtype=torch.long, device=device)
    )
    order = torch.cat([others, own])

    g_ordered = gram.index_select(0, order).index_select(1, order)
    k_ordered = g_ordered + eps * torch.eye(
        order.numel(), device=device, dtype=g_ordered.dtype
    )
    chol = torch.linalg.cholesky(k_ordered)
    n_own = own.numel()
    return chol.diagonal()[-n_own:].square()


def exact_chain_credits(
    step_embeddings: Sequence[torch.Tensor],
    correct_indices: Sequence[int],
    eps: float = STRIDE_EPS,
) -> Dict[int, torch.Tensor]:
    """delta_exact_ij = log(v_ij / eps) per rollout. Diagnostic / chain-rule check."""
    correct = [i for i in correct_indices if step_embeddings[i].shape[0] > 0]
    if len(correct) < 2:
        return {}
    device = step_embeddings[correct[0]].device
    sizes = [step_embeddings[i].shape[0] for i in correct]
    # float64: this is the diagnostic path used to verify the exact chain-rule
    # identity, where float32 leaves ~1e-6 residual (kernel condition number is
    # ~1/eps). The production u_ij path stays float32 -- it is clipped to [0,1]
    # and thresholded at tau, so 1e-6 is far below anything that matters there.
    phi = _normalized_phi(step_embeddings, correct, sizes, device, dtype=torch.float64)
    gram = phi @ phi.T
    offsets = [0]
    for n in sizes:
        offsets.append(offsets[-1] + n)

    out: Dict[int, torch.Tensor] = {}
    for local_i, global_i in enumerate(correct):
        v = _conditional_variances(gram, offsets, len(correct), local_i, eps)
        out[global_i] = torch.log(v / eps)
    return out


@torch.no_grad()
def stride_group_diversity_advantage(
    step_embeddings: Sequence[torch.Tensor],
    step_spans: Sequence[Sequence[Tuple[int, int]]],
    correct_indices: Sequence[int],
    response_mask: torch.Tensor,
    alpha: float = 1.0,
    eps: float = STRIDE_EPS,
    tau: Optional[float] = None,
    stats: Optional[dict] = None,
    strict: bool = True,
) -> torch.Tensor:
    """Step-wise token diversity advantage for ONE prompt group.

    Args:
        step_embeddings: per-rollout Tensor[n_i, d] of step embeddings.
        step_spans: per-rollout list of inclusive, disjoint (start, end) token spans.
        correct_indices: verified-correct rollout indices for this prompt.
        response_mask: bool Tensor[batch, response_length] of valid response tokens.
        alpha: scale. |A_div| <= alpha by construction.
        eps: kernel ridge. tau defaults to eps.
        stats: optional dict, populated with the spec's required logging fields.
        strict: raise on malformed spans (tests). When False, a malformed group
            returns zeros and records `invalid_span` in stats -- training must
            not die on one bad parse.

    Returns:
        Tensor[batch, response_length] float32 diversity advantage.
    """
    if tau is None:
        tau = eps
    device = response_mask.device
    batch_size, response_length = response_mask.shape
    a_div = torch.zeros(batch_size, response_length, device=device, dtype=torch.float32)

    if stats is not None:
        stats.setdefault("group_active", 0.0)
        stats.setdefault("invalid_span", 0.0)

    # A rollout contributes only if it has BOTH parsed steps and embeddings.
    correct = [
        i
        for i in correct_indices
        if len(step_spans[i]) > 0 and step_embeddings[i].shape[0] > 0
    ]
    if len(correct) < 2:
        return a_div

    sizes = [min(step_embeddings[i].shape[0], len(step_spans[i])) for i in correct]
    if any(n <= 0 for n in sizes):
        correct = [i for i, n in zip(correct, sizes) if n > 0]
        sizes = [n for n in sizes if n > 0]
        if len(correct) < 2:
            return a_div

    phi = _normalized_phi(step_embeddings, correct, sizes, device)
    gram = phi @ phi.T
    offsets = [0]
    for n in sizes:
        offsets.append(offsets[-1] + n)

    novelty: Dict[int, torch.Tensor] = {}
    for local_i, global_i in enumerate(correct):
        v = _conditional_variances(gram, offsets, len(correct), local_i, eps)
        u = (v - eps).clamp(min=0.0, max=1.0)
        u = torch.where(u <= tau, torch.zeros_like(u), u)
        novelty[global_i] = u

    # Broadcast each step's novelty across its disjoint span. U is temporary;
    # only the tokens marked in M participate in the baseline and the output.
    u_tok = torch.zeros_like(a_div)
    mask = torch.zeros_like(response_mask, dtype=torch.bool)
    for local_i, global_i in enumerate(correct):
        previous_end = -1
        for j, (start, end) in enumerate(step_spans[global_i][: sizes[local_i]]):
            start, end = int(start), int(end)
            bad = (
                not (0 <= start <= end < response_length)
                or start <= previous_end
                or not bool(response_mask[global_i, start : end + 1].all())
            )
            if bad:
                if strict:
                    raise ValueError(
                        f"invalid STRIDE step-token span rollout={global_i} step={j} "
                        f"span=({start},{end}) response_length={response_length}"
                    )
                if stats is not None:
                    stats["invalid_span"] += 1.0
                return torch.zeros_like(a_div)
            u_tok[global_i, start : end + 1] = novelty[global_i][j]
            mask[global_i, start : end + 1] = True
            previous_end = end

    if not bool(mask.any()):
        return a_div

    baseline = u_tok[mask].mean()
    a_div[mask] = alpha * (u_tok[mask] - baseline)

    if stats is not None:
        all_u = torch.cat([novelty[i] for i in correct])
        off = torch.triu_indices(gram.shape[0], gram.shape[0], offset=1, device=device)
        pair_cos = gram[off[0], off[1]] if off.numel() else torch.zeros(1, device=device)
        n_cond = float(gram.shape[0]) - float(sizes[0])
        er = effective_rank(gram)
        stats["group_active"] += 1.0
        stats["u_mean"] = float(all_u.mean().item())
        stats["u_std"] = float(all_u.std().item()) if all_u.numel() > 1 else 0.0
        stats["u_zero_frac"] = float((all_u == 0).float().mean().item())
        stats["a_div_rms"] = float(a_div[mask].square().mean().sqrt().item())
        stats["token_coverage"] = float(
            mask.sum().item() / max(float(response_mask.sum().item()), 1.0)
        )
        stats["pair_cos_mean"] = float(pair_cos.mean().item())
        stats["pair_cos_neg_frac"] = float((pair_cos < 0).float().mean().item())
        stats["gram_effective_rank"] = er
        stats["cond_to_rank_ratio"] = float(n_cond / er) if er > 0 else 0.0
        stats["n_steps_total"] = float(gram.shape[0])

    return a_div
