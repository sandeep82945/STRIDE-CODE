"""Reference embeddings over identical token coverage for STRIDE ablations."""


def pool_reference_hidden(hidden, spans, *, include_pooled=False):
    """Pool raw response hidden states; normalize only after pooling.

    ``spans`` are the parser's inclusive token spans. Gaps, the final answer and
    other unmapped tokens are not introduced into the single-step embedding.
    The two original outputs preserve the existing feature computations.
    """
    from numbers import Integral
    import torch
    import torch.nn.functional as F
    if (not isinstance(hidden, torch.Tensor) or hidden.ndim != 2
            or hidden.shape[0] < 1 or hidden.shape[1] < 1
            or not hidden.is_floating_point()):
        raise ValueError('Expected nonempty response-token hidden states')
    if not bool(torch.isfinite(hidden).all()):
        raise ValueError('Reference hidden states must be finite')
    # Materialize once: validation must not consume an iterable before pooling.
    try:
        spans = list(spans)
    except TypeError as error:
        raise ValueError('Expected an iterable of mapped token spans') from error
    last = -1
    for pair in spans:
        try:
            a, b = pair
        except (TypeError, ValueError) as error:
            raise ValueError('Each mapped token span requires two endpoints') from error
        if (isinstance(a, bool) or isinstance(b, bool)
                or not isinstance(a, Integral) or not isinstance(b, Integral)):
            raise ValueError('Mapped token span endpoints must be integers')
        if not (last < a <= b < hidden.shape[0]):
            raise ValueError('Invalid or overlapping mapped token spans')
        last = b
    whole = F.normalize(hidden.mean(0), dim=0).cpu()
    step = torch.stack([F.normalize(hidden[a:b+1].mean(0), dim=0)
                        for a, b in spans]).cpu() if spans else torch.empty((0, hidden.shape[-1]))
    if not include_pooled:
        return whole, step
    # Use raw token sums, not normalized step vectors or equally weighted means.
    if spans:
        count = sum(b - a + 1 for a, b in spans)
        total = torch.stack([hidden[a:b+1].sum(0) for a, b in spans]).sum(0)
        pooled = F.normalize(total / count, dim=0).cpu()
    else:
        pooled = torch.zeros(hidden.shape[-1], dtype=hidden.dtype)
    return whole, step, pooled
