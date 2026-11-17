"""Fixed-base LoRA KL regularization and resumable update-indexed warmup.

The original GRPO sampled k3 penalty is added directly to the loss, without
importance-ratio bias correction. It is not an exact vocabulary-summed KL.
Reference forwards happen before policy forwards, with no graph retained.
"""
from contextlib import contextmanager
import math


def learning_rate_at_update(base_lr, warmup_steps, completed_updates):
    """LR for the NEXT update: update 1 uses base_lr / warmup_steps."""
    if not math.isfinite(base_lr) or base_lr <= 0:
        raise ValueError('Base learning rate must be finite and positive')
    if type(warmup_steps) is not int or warmup_steps < 0:
        raise ValueError('Warmup steps must be a nonnegative integer')
    if type(completed_updates) is not int or completed_updates < 0:
        raise ValueError('Completed updates must be a nonnegative integer')
    return base_lr * (min((completed_updates + 1) / warmup_steps, 1.) if warmup_steps else 1.)


def set_update_learning_rate(optimizer, base_lr, warmup_steps, completed_updates):
    """No hidden scheduler state: strict contract + committed step define LR."""
    effective = learning_rate_at_update(base_lr, warmup_steps, completed_updates)
    for group in optimizer.param_groups:
        group['lr'] = effective
    return effective


def validate_kl_coefficient(coefficient):
    if not math.isfinite(coefficient) or coefficient < 0:
        raise ValueError('KL coefficient must be finite and nonnegative')


@contextmanager
def frozen_lora_reference(actor):
    """Borrow the immutable base LM; restore every module's training flag."""
    import torch
    from policy_runtime import policy_mode
    module = getattr(actor, 'module', actor)
    if policy_mode(actor) != 'lora' or not hasattr(module, 'disable_adapter'):
        raise ValueError('Fixed-base KL currently requires a LoRA policy')
    if any(p.requires_grad and 'lora_' not in name for name, p in module.named_parameters()):
        raise ValueError('KL reference requires every non-LoRA base parameter to be frozen')
    training_flags = [(child, child.training) for child in module.modules()]
    try:
        module.eval()
        with torch.no_grad(), module.disable_adapter():
            yield module
    finally:
        for child, was_training in training_flags:
            child.training = was_training


def reference_logps(actor, sequences):
    """Completion-only selected logps, detached and with no padded positions."""
    from batched_learner import selected_logps
    with frozen_lora_reference(actor) as reference:
        values = selected_logps(reference, sequences)
        return [value.detach() for value in values]


def sampled_k3_kl(new_logps, reference_logps):
    """Original GRPO k3: exp(log ref - log policy) - (log ref - log policy) - 1.

    The caller supplies only real completion positions (including EOS and any
    truncated completion tokens). This applies even when reward advantages are
    all zero. Reference tensors are detached defensively. No clamping changes
    the estimator; nonfinite values fail rather than silently changing it.
    """
    import torch
    if new_logps.shape != reference_logps.shape or new_logps.numel() == 0:
        raise ValueError('Policy and reference require equal nonempty token shapes')
    delta = reference_logps.detach().float() - new_logps.float()
    penalty = torch.expm1(delta) - delta
    if not bool(torch.isfinite(penalty).all()):
        raise ValueError('Nonfinite sampled KL penalty')
    return penalty
