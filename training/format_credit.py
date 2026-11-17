"""Binary format reward for the separately identified warmup cohort only.

The caller supplies target-blind native ``step_eligible`` flags, including its
truncation and original-token-mapping gates. This module does not inspect text,
count steps, assess reasoning quality, or create a diversity reward.
"""
from __future__ import annotations

import math
import torch


def _binary(values, name, k, device):
    result=torch.as_tensor(values, dtype=torch.float32, device=device)
    if result.shape!=(k,) or not bool(((result==0)|(result==1)).all()):
        raise ValueError(name+' must contain one binary value per response')
    return result


def _standardize(rewards):
    # Match the existing pilot's sample-std + 1e-6 convention. Explicit tied
    # handling avoids spurious signal from floating-point means of e.g. 1.1.
    if bool((rewards==rewards[0]).all()):
        return torch.zeros_like(rewards)
    return (rewards-rewards.mean())/(rewards.std(unbiased=True)+1e-6)


def _std(values):
    return 0. if bool((values==values[0]).all()) else float(values.std(unbiased=True))


@torch.no_grad()
def compute_format_credit(answer_correct, format_flags, response_mask, *, format_coefficient=0.1):
    """Joint GRPO advantages for one question's complete rollout group.

    ``r_i = accuracy_i + format_coefficient * format_i``. The coefficient is
    restricted to [0,1), so every correct response outranks every incorrect
    response before and after joint standardization. This does not guarantee
    positive advantages for all correct responses: a correct response can be
    below its group's mean, especially in an all-correct group.

    Return detached float32 tensors on the mask device. ``format_effect`` is
    joint-normalized advantage minus accuracy-only advantage; it includes the
    changed group normalization and is not a separately normalized bonus.
    The coefficient-zero result is accuracy-only GRPO. Apply this function
    during warmup only; downstream branch training uses its ordinary rewards.
    """
    if isinstance(format_coefficient,bool):
        raise ValueError('format_coefficient must be numeric in [0,1)')
    try: coefficient=float(format_coefficient)
    except (TypeError,ValueError): raise ValueError('format_coefficient must be numeric in [0,1)') from None
    if not math.isfinite(coefficient) or not 0.<=coefficient<1.:
        raise ValueError('format_coefficient must be finite and in [0,1)')
    if not isinstance(response_mask,torch.Tensor) or response_mask.ndim!=2 or response_mask.dtype!=torch.bool:
        raise ValueError('response_mask must be a rank-two boolean tensor')
    k,width=response_mask.shape
    if k<2 or width<1 or not bool(response_mask.any(1).all()):
        raise ValueError('At least two nonempty responses are required')
    accuracy=_binary(answer_correct,'answer_correct',k,response_mask.device)
    flags=_binary(format_flags,'format_flags',k,response_mask.device)
    rewards=accuracy+coefficient*flags
    main_vector=_standardize(accuracy)
    joint_vector=_standardize(rewards)
    main=main_vector[:,None]*response_mask
    advantages=joint_vector[:,None]*response_mask
    effect=advantages-main
    correct=accuracy.bool()
    wrong=~correct
    accuracy_tied=bool((accuracy==accuracy[0]).all())
    format_tied=bool((flags==flags[0]).all())
    joint_tied=bool((rewards==rewards[0]).all())
    def rms(tensor):return float(tensor[response_mask].square().mean().sqrt())
    correct_tokens=correct[:,None]&response_mask
    diagnostics=dict(
        group_size=k,format_coefficient=coefficient,
        valid_tokens=int(response_mask.sum()),response_tokens=response_mask.sum(1).tolist(),
        correct_responses=int(correct.sum()),format_eligible_responses=int(flags.sum()),
        correct_format_eligible_responses=int((correct&flags.bool()).sum()),
        accuracy_tied=accuracy_tied,format_tied=format_tied,joint_reward_tied=joint_tied,
        format_signal_active=bool(coefficient>0. and not format_tied),
        format_rescues_accuracy_tie=bool(coefficient>0. and accuracy_tied and not format_tied),
        accuracy_reward_mean=float(accuracy.mean()),accuracy_reward_std=_std(accuracy),
        format_flag_mean=float(flags.mean()),format_flag_std=_std(flags),
        joint_reward_mean=float(rewards.mean()),joint_reward_std=_std(rewards),
        main_rms=rms(main),advantage_rms=rms(advantages),format_effect_rms=rms(effect),
        negative_correct_responses=int(((joint_vector<0)&correct).sum()),
        negative_correct_tokens=int(((advantages<0)&correct_tokens).sum()),
        negative_main_correct_responses=int(((main_vector<0)&correct).sum()),
        negative_format_effect_correct_tokens=int(((effect<0)&correct_tokens).sum()),
        format_effect_opposes_accuracy_tokens=int(((effect*main<0)&response_mask).sum()),
        correctness_main_sign_flip_tokens=int(((advantages*main<0)&response_mask).sum()),
        nonzero_advantage_responses=int((joint_vector!=0).sum()),
        min_correct_minus_max_wrong_reward=(float(rewards[correct].min()-rewards[wrong].max())
                                            if bool(correct.any()) and bool(wrong.any()) else None),
    )
    return dict(rewards=rewards,main=main,advantages=advantages,format_effect=effect,
                diagnostics=diagnostics)
