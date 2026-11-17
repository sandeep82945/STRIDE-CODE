"""Optional paired LoRA learner forwards; unchanged summed response-token loss.

Each row is an independent left-padded causal sequence. No examples are joined
in the attention dimension. Position IDs restart at zero on each real sequence;
padding contributes neither loss nor token normalization. Dropout must be zero
for comparable single/pair training (the configured Qwen3 LoRA uses zero).
"""
from contextlib import nullcontext
import math
import time


def packed_records(records, size):
    if size != 2:
        raise ValueError('Paired learner requires microbatch size two')
    ordered = sorted(records, key=lambda item: len(item[0]['prompt_token_ids']) + len(item[2]['token_ids']))
    return [ordered[i:i + size] for i in range(0, len(ordered), size)]


def selected_logps(actor, sequences):
    """One model forward; return a differentiable vector per real response."""
    import torch
    import torch.nn.functional as F
    from policy_runtime import forward_context
    if not sequences or any(not p or not r for p, r in sequences):
        raise ValueError('Nonempty prompts and responses required')
    device = next(actor.parameters()).device
    width = max(len(p) + len(r) - 1 for p, r in sequences)
    response_width = max(len(r) for _, r in sequences)
    inputs = torch.zeros((len(sequences), width), dtype=torch.long, device=device)
    mask = torch.zeros_like(inputs)
    for row, (prompt, response) in enumerate(sequences):
        values = prompt + response[:-1]
        inputs[row, -len(values):] = torch.tensor(values, dtype=torch.long, device=device)
        mask[row, -len(values):] = 1
    positions = (mask.cumsum(-1) - 1).clamp_min(0)
    with forward_context(actor):
        logits = actor(input_ids=inputs, attention_mask=mask, position_ids=positions,
                       use_cache=False, logits_to_keep=response_width).logits
    if logits.shape[:2] != (len(sequences), response_width):
        raise ValueError('Model did not retain requested response logits')
    # Normalize only actual response positions in FP32, as the original path.
    return [-F.cross_entropy(logits[row, -len(response):].float(),
                             torch.tensor(response, dtype=torch.long, device=device), reduction='none')
            for row, (_, response) in enumerate(sequences)]


def update_distributed(actor, optimizer, groups, *, dist, force_signal=False, microbatch_size=2, kl_coefficient=0.):
    import torch
    import trainer as pilot
    from stabilization import reference_logps, sampled_k3_kl, validate_kl_coefficient
    validate_kl_coefficient(kl_coefficient)
    from policy_runtime import policy_mode
    if policy_mode(actor) != 'lora':
        raise ValueError('Paired learner supports LoRA DDP only')
    actor.train()
    optimizer.zero_grad(set_to_none=True)
    parameters = [p for p in actor.parameters() if p.requires_grad]
    device = parameters[0].device
    records = [(g, i, r) for g in groups for i, r in enumerate(g['records'])]
    if not records:
        raise ValueError('Every learner rank needs at least one rollout')
    packs = packed_records(records, microbatch_size)
    tokens = torch.tensor(sum(len(r['token_ids']) for _, _, r in records), device=device, dtype=torch.int64)
    dist.all_reduce(tokens, op=dist.ReduceOp.SUM)
    denominator = int(tokens)
    if denominator <= 0:
        raise ValueError('Positive real token count required')
    scale = dist.get_world_size() / denominator
    forwards = torch.tensor(len(packs), device=device, dtype=torch.int64)
    dist.all_reduce(forwards, op=dist.ReduceOp.MAX)
    padded_count = int(forwards)
    counts = torch.tensor([len(records), len(packs)], device=device, dtype=torch.int64)
    dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    began = time.time()
    local_loss = local_abs_sum = local_abs_max = 0.
    local_clipped = 0
    local_kl = local_policy_loss = 0.
    for j in range(padded_count):
        real = j < len(packs)
        pack = packs[j] if real else [records[0]]
        context = actor.no_sync() if j + 1 < padded_count else nullcontext()
        sequences = [(g['prompt_token_ids'], r['token_ids']) for g, _, r in pack]
        references = reference_logps(actor, sequences) if real and kl_coefficient else [None] * len(pack)
        with context:
            new = selected_logps(actor, [(g['prompt_token_ids'], r['token_ids']) for g, _, r in pack])
            if not real:
                (sum(x.sum() for x in new) * 0.).backward()
                del new
                continue
            losses = []
            for (group, index, record), logps, reference in zip(pack, new, references):
                old = torch.tensor(record['logprobs'], device=device, dtype=torch.float32)
                if len(logps) != len(old) or not len(old):
                    raise ValueError('Behavior and learner token lengths disagree')
                advantage = group['advantages'][index, :len(old)].to(device)
                if force_signal:
                    advantage = torch.ones_like(advantage)
                loss, diff, clipped = pilot.loss_terms(logps, old, advantage)
                local_policy_loss += float(loss.detach())
                if reference is not None:
                    kl = sampled_k3_kl(logps, reference).sum()
                    local_kl += float(kl.detach())
                    loss = loss + kl_coefficient * kl
                    del kl
                losses.append(loss)
                local_loss += float(loss.detach())
                local_abs_sum += float(diff.abs().sum())
                local_abs_max = max(local_abs_max, float(diff.abs().max()))
                local_clipped += int(clipped.sum())
            (sum(losses) * scale).backward()
        del new, losses, loss, diff, clipped, old, advantage, logps, reference, references
    norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.))
    if not math.isfinite(norm):
        raise ValueError('Nonfinite synchronized gradient')
    optimizer.step()
    actor.eval()
    sums = torch.tensor([local_loss, local_abs_sum, local_clipped, local_kl, local_policy_loss], device=device, dtype=torch.float64)
    dist.all_reduce(sums, op=dist.ReduceOp.SUM)
    maxima = torch.tensor([local_abs_max, time.time() - began,
                           torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0],
                          device=device, dtype=torch.float64)
    dist.all_reduce(maxima, op=dist.ReduceOp.MAX)
    return dict(loss=float(sums[0]) / denominator, gradient_norm=norm,
                policy_loss=float(sums[4]) / denominator,
                sampled_kl_mean=float(sums[3]) / denominator,
                kl_loss=kl_coefficient * float(sums[3]) / denominator,
                kl_coefficient=kl_coefficient,
                response_tokens=denominator, update_seconds=float(maxima[1]),
                actor_behavior_abs_logp_mean=float(sums[1]) / denominator,
                actor_behavior_abs_logp_max=float(maxima[0]),
                ratio_clip_fraction=float(sums[2]) / denominator,
                peak_gpu_bytes=int(maxima[2]), rollout_count=int(counts[0]),
                collective_padding_forwards=padded_count * dist.get_world_size() - int(counts[1]),
                learner_microbatch_size=microbatch_size,
                learner_forwards_per_rank=padded_count)
