"""Resumable branch driver derived from the frozen eight-GPU trainer.

Each completed update commits AdamW and all four learner RNG states. Resume
requires a verified state_NNN checkpoint and a fresh output attempt directory.
Fresh training updates the full base model by default; LoRA is optional.
Optional linear learning-rate warmup and fixed-base sampled KL are recorded in
the strict resume contract; there is no format reward or diversity-coefficient ramp.

Eight allocated GPUs: four FSDP (full) or DDP (LoRA) learners and four isolated vLLM samplers.

Run with torchrun --standalone --nnodes=1 --nproc_per_node=4. CUDA_VISIBLE_DEVICES
must retain all eight Slurm-allocated IDs. Each rank uses the corresponding
first-half learner device and launches its sampler on the corresponding second
half ID. One fresh global rollout batch produces exactly one optimizer update.

The copied pilot credit, parsing, actor, log-probability and reference routines
are reused. DDP's gradient average is compensated by world_size / global real
response-token count; this is the pilot's exact summed token loss. Whole prompt
groups remain local, so group standardization and STRIDE credit are unchanged.
"""
from __future__ import annotations

import argparse
import re
from contextlib import nullcontext
from datetime import timedelta
from importlib.metadata import version
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

import trainer as pilot
import runtime_config as config
from tracking import setup_observers, submit_validation
from policy_runtime import wrap_actor, policy_mode, save_policy
from stabilization import set_update_learning_rate
import resumable_checkpoint as checkpoints


ROOT = Path(__file__).resolve().parent
ARMS = config.ARMS


def allocation(visible, local_rank, world_size):
    ids = visible.split(',')
    if world_size != 4 or len(ids) != 8 or any(not x.strip() for x in ids):
        raise ValueError('Require four torchrun learner ranks and eight Slurm-visible GPUs')
    if len(set(ids)) != 8 or not 0 <= local_rank < world_size:
        raise ValueError('Invalid or duplicate allocated GPU IDs')
    return ids[local_rank], ids[world_size + local_rank]


def epoch_batch(rows, step, batch_size, seed=42, epoch_offset=0):
    per_epoch = config.epoch_update_count(len(rows), batch_size)
    if step < 0 or epoch_offset < 0:
        raise ValueError('Negative schedule position')
    epoch, within = divmod(step, per_epoch)
    order = list(range(len(rows)))
    random.Random(seed + epoch + epoch_offset).shuffle(order)
    return [rows[i] for i in order[within * batch_size:(within + 1) * batch_size]]


def rank_offset(batch_size, rank, world_size):
    if world_size < 1 or batch_size < world_size or not 0 <= rank < world_size:
        raise ValueError('Every learner needs a real prompt group')
    width, extra = divmod(batch_size, world_size)
    return rank * width + min(rank, extra)


def rank_batch(batch, rank, world_size):
    start = rank_offset(len(batch), rank, world_size)
    width, extra = divmod(len(batch), world_size)
    return batch[start:start + width + (rank < extra)]


def checkpoint_pins(total_steps, per_epoch, *, keep_all=False):
    """Retention is operational and does not change the scientific resume contract.

    All-checkpoint mode includes the initial policy/state pair at step zero.
    A resumed attempt retains every checkpoint it writes; its immutable source
    attempt is never pruned by this trainer.
    """
    if keep_all:
        return set(range(total_steps + 1))
    return set(range(per_epoch, total_steps + 1, per_epoch)) | {total_steps}


validate_rows = config.validate_rows


def ddp_token_scale(global_tokens, world_size):
    if global_tokens <= 0 or world_size < 1:
        raise ValueError('Positive real token count and world size required')
    return world_size / global_tokens


def trainable_fingerprint(actor):
    """Smoke-only collective fingerprint; full FSDP parameters must be gathered."""
    import torch
    if policy_mode(actor) == 'full':
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        context = FSDP.summon_full_params(actor, recurse=True, writeback=False)
    else:
        context = nullcontext()
    digest = hashlib.sha256()
    with context:
        for name, parameter in actor.named_parameters():
            if parameter.requires_grad:
                digest.update(name.encode())
                digest.update(str(tuple(parameter.shape)).encode())
                digest.update(str(parameter.dtype).encode())
                digest.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def update_distributed(actor, optimizer, groups, *, dist, force_signal=False, kl_coefficient=0.):
    import torch

    from stabilization import reference_logps, sampled_k3_kl, validate_kl_coefficient
    validate_kl_coefficient(kl_coefficient)
    full_training = policy_mode(actor) == 'full'
    if kl_coefficient and full_training:
        raise ValueError('Fixed-base KL currently requires a LoRA policy')
    actor.train()
    optimizer.zero_grad(set_to_none=True)
    parameters = [p for p in actor.parameters() if p.requires_grad]
    device = parameters[0].device
    records = [(g, i, r) for g in groups for i, r in enumerate(g['records'])]
    if not records:
        raise ValueError('Every learner rank needs at least one rollout')
    local_tokens = sum(len(r['token_ids']) for _, _, r in records)
    tokens = torch.tensor(local_tokens, device=device, dtype=torch.int64)
    dist.all_reduce(tokens, op=dist.ReduceOp.SUM)
    denominator = int(tokens)
    scale = ddp_token_scale(denominator, dist.get_world_size())
    # Equal forward/backward counts preserve DDP/FSDP collective order. Only
    # compute is padded: samplers, token normalization and metrics see real data.
    count = torch.tensor(len(records), device=device, dtype=torch.int64)
    low, high = count.clone(), count.clone()
    dist.all_reduce(low, op=dist.ReduceOp.MIN)
    dist.all_reduce(high, op=dist.ReduceOp.MAX)
    if int(low) < 1:
        raise ValueError('Every learner rank needs at least one rollout')
    total_records = count.clone()
    dist.all_reduce(total_records, op=dist.ReduceOp.SUM)
    padded_count = int(high)
    began = time.time()
    local_loss = 0.
    local_abs_sum = 0.
    local_abs_max = 0.
    local_clipped = 0
    local_kl = 0.
    local_policy_loss = 0.
    for j in range(padded_count):
        real = j < len(records)
        group, index, record = records[j] if real else records[0]
        # Full FSDP reduces each microbatch so gradients stay sharded. LoRA
        # DDP accumulates locally and synchronizes on the final rollout.
        context = actor.no_sync() if not full_training and j + 1 < padded_count else nullcontext()
        # Finish the no-grad base forward before creating policy activations.
        reference = (reference_logps(actor, [(group['prompt_token_ids'], record['token_ids'])])[0]
                     if real and kl_coefficient else None)
        with context:
            new = pilot.selected_logps(actor, group['prompt_token_ids'], record['token_ids'])
            if not real:
                # Reuse a real sequence only as a differentiable zero-loss
                # collective placeholder. It is never a generated rollout.
                (new.sum() * 0.).backward()
                del new
                continue
            old = torch.tensor(record['logprobs'], device=device, dtype=torch.float32)
            if len(new) != len(old) or not len(old):
                raise ValueError('Behavior and learner token lengths disagree')
            advantage = group['advantages'][index, :len(old)].to(device)
            if force_signal:
                advantage = torch.ones_like(advantage)
            loss, diff, clipped = pilot.loss_terms(new, old, advantage)
            local_policy_loss += float(loss.detach())
            if reference is not None:
                kl = sampled_k3_kl(new, reference).sum()
                local_kl += float(kl.detach())
                loss = loss + kl_coefficient * kl
                del kl
            (loss * scale).backward()
        local_loss += float(loss.detach())
        local_abs_sum += float(diff.abs().sum())
        local_abs_max = max(local_abs_max, float(diff.abs().max()))
        local_clipped += int(clipped.sum())
        del new, old, advantage, loss, diff, clipped, reference
    norm = float(actor.clip_grad_norm_(1.) if full_training else torch.nn.utils.clip_grad_norm_(parameters, 1.))
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
                peak_gpu_bytes=int(maxima[2]), rollout_count=int(total_records),
                collective_padding_forwards=padded_count * dist.get_world_size() - int(total_records))


def update_learner(actor, optimizer, groups, *, dist, force_signal=False, learner_microbatch_size=1, kl_coefficient=0.):
    """Single or paired response forwards with identical token objectives."""
    if learner_microbatch_size == 2:
        from batched_learner import update_distributed as paired_update
        return paired_update(actor, optimizer, groups, dist=dist, force_signal=force_signal, kl_coefficient=kl_coefficient)
    if learner_microbatch_size != 1:
        raise ValueError('Learner microbatch must be one or two')
    return update_distributed(actor, optimizer, groups, dist=dist, force_signal=force_signal, kl_coefficient=kl_coefficient)


def diagnostics(groups, arm):
    samples = [r for g in groups for r in g['records']]
    result = dict(groups=len(groups), responses=len(samples),
                  correct=sum(bool(r['score']['answer_correct']) for r in samples),
                  eligible_correct=sum(bool(r['score']['answer_correct'] and r['score']['step_eligible']) for r in samples),
                  format_eligible=sum(bool(r['score']['step_eligible']) for r in samples),
                  truncated=sum(r['finish_reason'] == 'length' for r in samples),
                  response_tokens=sum(len(r['token_ids']) for r in samples),
                  gate_active=sum(bool(g['diagnostics'].get('gate_active', g['diagnostics'].get('eligible_responses', 0) >= 2)) for g in groups),
                  local_uniform_different_groups=sum(g['diagnostics'].get('local_uniform_maxdiff', 0) > 1e-7 for g in groups),
                  local_single_different_groups=sum(g['diagnostics'].get('local_single_maxdiff', 0) > 1e-7 for g in groups),
                  bonus_nonzero_groups=sum(g['diagnostics'].get('arms', {}).get(arm, {}).get('aux_positive_tokens', 0) > 0 for g in groups),
                  local_uniform_differing_tokens=sum(g['diagnostics'].get('local_uniform_differing_tokens', 0) for g in groups),
                  mapped_tokens=sum(g['diagnostics'].get('mapped_tokens', 0) for g in groups),
                  format_rescued_tied_groups=sum(bool(g['diagnostics'].get('format', {}).get('format_rescues_accuracy_tie', False)) for g in groups),
                  nonzero_advantage_tokens=sum(int(g['advantages'].ne(0).sum()) for g in groups))
    for field in ('aux_mass', 'aux_positive_tokens', 'aux_negative_tokens',
                  'main_opposed_tokens', 'main_reversed_tokens'):
        result[field] = sum(g['diagnostics'].get('arms', {}).get(arm, {}).get(field, 0) for g in groups)
    return result


def metrics(stats, totals, *, epoch):
    n, k = totals['responses'], totals['groups']
    result = {'train/' + key: value for key, value in stats.items()
              if isinstance(value, (int, float)) and not isinstance(value, bool)}
    result.update({'train/epoch': epoch, 'rollouts/answer_correct_fraction': totals['correct'] / n,
                   'rollouts/eligible_correct_fraction': totals['eligible_correct'] / n,
                   'rollouts/format_eligible_fraction': totals['format_eligible'] / n,
                   'rollouts/truncated_fraction': totals['truncated'] / n,
                   'rollouts/mean_response_tokens': totals['response_tokens'] / n,
                   'credit/activation_fraction': totals['gate_active'] / k,
                   'credit/local_uniform_different_group_fraction': totals['local_uniform_different_groups'] / k,
                   'credit/local_single_different_group_fraction': totals['local_single_different_groups'] / k,
                   'credit/bonus_nonzero_group_fraction': totals['bonus_nonzero_groups'] / k,
                   'credit/nonzero_advantage_token_fraction': totals['nonzero_advantage_tokens'] / totals['response_tokens'],
                   'credit/format_rescued_tied_group_fraction': totals['format_rescued_tied_groups'] / k})
    for name in ('aux_mass', 'aux_positive_tokens', 'aux_negative_tokens', 'main_opposed_tokens',
                 'main_reversed_tokens', 'local_uniform_differing_tokens', 'mapped_tokens'):
        result['credit/' + name] = totals[name]
    return result


def save_synchronized(actor, out, step, dist, *, tokenizer=None, finetune_mode='full'):
    prefix = 'model' if finetune_mode == 'full' else 'adapter'
    target = Path(out) / f'{prefix}_{step:03d}'
    # FSDP state collection is collective: every learner must enter the saver.
    save_policy(actor, target, tokenizer, finetune_mode=finetune_mode)
    dist.barrier()
    required = 'config.json' if finetune_mode == 'full' else 'adapter_model.safetensors'
    if not (target / required).is_file():
        raise FileNotFoundError('Rank-zero policy checkpoint is not visible after synchronization')
    return str(target)


def prune_synchronization_adapters(out, current_step, pinned_steps, *, finetune_mode='full'):
    """Keep current/previous and all epoch snapshots consumed by async evaluators."""
    out = Path(out).resolve()
    prefix = 'model' if finetune_mode == 'full' else 'adapter'
    for old in out.glob(prefix + '_*'):
        if re.fullmatch(prefix + r'_[0-9]{3,}', old.name) is None:
            continue
        step = int(old.name.split('_')[1])
        if step >= current_step - 1 or step in pinned_steps:
            continue
        if old.is_symlink() or old.parent.resolve() != out or not old.is_dir():
            raise ValueError('Unexpected synchronization adapter path')
        # Only immutable directories created by this trainer with an external
        # manifest qualify. Keep manifests so synchronization provenance remains.
        if not old.with_name(old.name + '_manifest.json').is_file():
            raise ValueError('Refusing to prune an adapter without its manifest')
        shutil.rmtree(old)


def training_contract(args, world_size, training_questions, identity):
    """Scientific/runtime invariants; attempt paths and logging may change."""
    names = ('trainer_resumable.py', 'resumable_checkpoint.py', 'trainer_distributed.py',
             'trainer.py', 'policy_runtime.py', 'runtime_config.py', 'stabilization.py', 'batched_learner.py', 'sampler_worker.py', 'credit.py', 'production_stride.py',
             'experiment_credit.py', 'feature_pooling.py', 'tracking.py',
             'validation_jobs.py', 'evaluate_async.py', 'evaluate.py',
             'scoring.py', 'native_contract.py', 'native_declared_steps.py',
             'native_step_parser.py', 'format_credit.py')
    files = [ROOT / name for name in names] + sorted((ROOT / 'pilot_grading').rglob('*.py'))
    contract = config.scientific_contract(args, world_size=world_size,
                   training_questions=training_questions, identity=identity)
    contract.update(versions={name: version(name) for name in ('torch', 'transformers', 'vllm', 'peft', 'numpy')},
                    cuda_launch_blocking=os.environ.get('CUDA_LAUNCH_BLOCKING', '0'),
                    code_sha256={str(path.relative_to(ROOT)): pilot.digest(path) for path in files})
    return contract


def initial_training_state(args, world_size, training_questions, identity):
    """Fresh base initialization or a verified continuation of this exact experiment.

    The contract always records the original base initialization. The attempt's
    started.json separately records whether it resumed and from which state.
    No adapter-only checkpoint can replace the complete optimizer/RNG state.
    """
    contract = training_contract(args, world_size, training_questions, identity)
    extend = getattr(args, 'allow_epoch_extension', False)
    if extend and not args.resume_from:
        raise ValueError('Epoch extension requires a verified resume checkpoint')
    resume = (checkpoints.inspect_checkpoint(args.resume_from, None if extend else contract, world_size)
              if args.resume_from else None)
    if extend:
        previous = resume['contract']
        old_epochs, new_epochs = previous.get('epochs'), contract.get('epochs')
        if type(old_epochs) is not int or type(new_epochs) is not int or new_epochs <= old_epochs:
            raise ValueError('Epoch extension requires a strictly larger epoch target')
        if {k: v for k, v in previous.items() if k != 'epochs'} != {k: v for k, v in contract.items() if k != 'epochs'}:
            raise ValueError('Epoch extension may change only epochs; all other training contract fields must match')
        if not 0 <= resume['completed_step'] <= old_epochs * config.epoch_update_count(training_questions, args.batch_size):
            raise ValueError('Resume checkpoint exceeds its original training schedule')
    start_step = resume['completed_step'] if resume else 0
    steps = config.schedule_steps(args, training_questions)
    if not 0 <= start_step <= steps:
        raise ValueError('Checkpoint step outside the configured training schedule')
    return contract, resume, start_step, steps


def main():
    parser = argparse.ArgumentParser()
    config.add_training_arguments(parser, resumable=True)
    args = parser.parse_args()
    config.validate_options(args, resumable=True)
    model = str(config.resolve_model_path(args.model_path, args.model))
    args.model_path = model
    data, rows = config.load_training_data(args)
    local_rank, world_size = int(os.environ['LOCAL_RANK']), int(os.environ['WORLD_SIZE'])
    learner_id, sampler_id = allocation(os.environ.get('CUDA_VISIBLE_DEVICES', ''), local_rank, world_size)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    import torch
    import torch.distributed as dist
    from transformers import AutoTokenizer
    from sampler_worker import SamplerClient

    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl', timeout=timedelta(minutes=90))
    rank = dist.get_rank()
    if rank != local_rank:
        raise ValueError('This trainer supports a single eight-GPU node only')
    out = Path(args.out).resolve()
    client = None
    run = None
    training_complete = False
    try:
        # Hash model payload once on rank zero, then share the same identity.
        identity_packet = [config.model_identity(model) if rank == 0 else None]
        dist.broadcast_object_list(identity_packet, src=0)
        identity = identity_packet[0]
        epoch_batch(rows, 0, args.batch_size, args.seed)
        contract, resume, start_step, steps = initial_training_state(args, world_size, len(rows), identity)
        retention = 'all' if args.keep_all_checkpoints else 'latest_two_and_epochs'
        initial_policy = resume['policy_path'] if resume else None
        initial_adapter = initial_policy if args.finetune_mode == 'lora' else None
        if rank == 0:
            if out.exists() and (out.is_symlink() or any(out.iterdir())):
                raise FileExistsError('Use an empty, fresh attempt directory')
            out.mkdir(parents=True, exist_ok=True)
            if (out / 'started.json').exists():
                raise FileExistsError('Use a fresh attempt directory')
            pilot.write(out / 'started.json', dict(args=vars(args), started=time.time(),
                model_identity=identity, data_sha256=pilot.digest(data),
                code_sha256={str(p.relative_to(ROOT)): pilot.digest(p) for p in ROOT.rglob('*.py')},
                slurm_job_id=os.environ.get('SLURM_JOB_ID'), world_size=world_size,
                start_step=start_step, resume_from=resume['state_path'] if resume else None,
                attempt_initialization='resume' if resume else contract['initialization'],
                training_contract=contract,
                context_limit=8192, group_size=args.rollouts,
                global_prompt_batch=world_size if args.mode == 'smoke' else args.batch_size,
                checkpoint_retention=retention,
                resume_state_retention=args.resume_state_retention,
                epoch_extension_from=resume['contract']['epochs'] if args.allow_epoch_extension else None,
                actual_format_coefficient=0.,
                note='Direct training from the base model; constant diversity coefficient, no format reward.'))
        dist.barrier()
        run, validation = setup_observers(args, rows, out, dist,
                               updates=config.schedule_steps(args, len(rows)))
        rank_out = out / f'rank_{rank}'
        rank_out.mkdir(exist_ok=False)
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        import numpy as np
        np.random.seed(args.seed)
        tok = AutoTokenizer.from_pretrained(model, local_files_only=True, use_fast=True)
        # Validate context length before allocating either model process.
        for row in rows:
            pilot.render(tok, row)
        pilot.write(rank_out / 'runtime.json', dict(versions={name: version(name) for name in
                     ('torch', 'transformers', 'vllm', 'peft')}, rank=rank,
                     gpu=torch.cuda.get_device_name(local_rank), learner_device=learner_id,
                     sampler_device=sampler_id, allocated_devices=os.environ['CUDA_VISIBLE_DEVICES'],
                     input_adapter_sha256=pilot.digest(Path(initial_adapter) / 'adapter_model.safetensors') if initial_adapter else None))
        model_actor = pilot.load_actor(model, initial_adapter, finetune_mode=args.finetune_mode,
                         checkpoint=initial_policy if args.finetune_mode == 'full' else None,
                         device='cpu' if args.finetune_mode == 'full' else 'cuda')
        model_actor.eval()
        actor = wrap_actor(model_actor, local_rank, args.finetune_mode)
        need_features = args.mode == 'smoke' or args.arm != 'grpo'
        reference = (pilot.load_reference(model) if args.finetune_mode == 'full'
                     else actor.module) if need_features else None
        optimizer = torch.optim.AdamW([p for p in actor.parameters() if p.requires_grad], lr=args.learning_rate, weight_decay=0., foreach=False)
        resume_rng = checkpoints.load_training_state(resume, optimizer, rank) if resume else None
        policy_checkpoint = save_synchronized(actor, out, start_step, dist, tokenizer=tok, finetune_mode=args.finetune_mode)
        # Start full sampling from the actual initial/resumed policy export.
        # This avoids loading the base engine only to restart it before step 1.
        client = SamplerClient.launch(
            policy_checkpoint if args.finetune_mode == 'full' else model,
            rank_out / 'ipc', sampler_device=sampler_id,
            eager=args.sampler_eager, startup_timeout=1200,
            max_num_seqs=args.sampler_max_num_seqs, engine_seed=args.seed,
            finetune_mode=args.finetune_mode, tokenizer_path=model,
            model_version=(start_step + 1) if args.finetune_mode == 'full' else 0)
        per_epoch = config.epoch_update_count(len(rows), args.batch_size)
        pinned_steps = checkpoint_pins(steps, per_epoch, keep_all=args.keep_all_checkpoints)
        # Startup, DDP, sampler, and W&B initialization can consume RNG. Restore only
        # after all of them, then commit the initial snapshot in this new attempt.
        if resume_rng is not None:
            checkpoints.restore_rng(resume_rng, torch.device('cuda', local_rank))
        state_checkpoint = checkpoints.save_distributed_state(out, start_step, actor, optimizer, dist,
                             contract, device=torch.device('cuda', local_rank), finetune_mode=args.finetune_mode)
        if rank == 0 and args.resume_state_retention == 'latest-epoch':
            checkpoints.retain_latest_epoch_states(out, start_step, per_epoch)
        dist.barrier()
        for step in range(start_step, steps):
            began = time.time()
            global_batch = epoch_batch(rows, step, args.batch_size, args.seed, epoch_offset=0)
            batch = rank_batch(global_batch, rank, world_size)
            if args.mode == 'smoke':
                batch = batch[:1]
            offset = rank_offset(len(global_batch), rank, world_size)
            seed = 20260980 + args.seed * 1000000 + step * args.batch_size + offset
            generated = client.generate(pilot.request(tok, batch, f'update_{step}', policy_checkpoint, step + 1, seed,
                                       cap=args.smoke_cap if args.mode == 'smoke' else 8192,
                                       n=args.rollouts, finetune_mode=args.finetune_mode), timeout=args.generation_timeout)
            annotation_started = time.time()
            # GRPO has identically zero diversity credit: skipping its unused
            # reference forwards is an exact objective-preserving optimization.
            need_features = args.arm != 'grpo'
            groups = pilot.annotate_groups(tok, batch, generated, reference if need_features else None,
                                           args.arm, alpha=args.alpha, format_coefficient=0.,
                                           novelty_reference_scope=args.novelty_reference_scope)
            annotation_seconds = time.time() - annotation_started
            pilot.write(rank_out / f'rollouts_{step:03d}.json', pilot.serialize_groups(groups))
            before = frozen_before = None
            if args.mode == 'smoke':
                first = groups[0]['records'][0]
                with torch.no_grad():
                    before = pilot.selected_logps(actor, groups[0]['prompt_token_ids'], first['token_ids']).cpu()
                frozen_before = pilot.features(reference, groups[0]['prompt_token_ids'], first['token_ids'], [])[0]
            effective_lr = set_update_learning_rate(optimizer, args.learning_rate, args.warmup_steps, step)
            stats = update_learner(actor, optimizer, groups, dist=dist, force_signal=args.mode == 'smoke',
                                   learner_microbatch_size=args.learner_microbatch_size,
                                   kl_coefficient=args.kl_coefficient)
            stats.update(learning_rate=effective_lr, target_learning_rate=args.learning_rate,
                         warmup_steps=args.warmup_steps)
            policy_checkpoint = save_synchronized(actor, out, step + 1, dist, tokenizer=tok, finetune_mode=args.finetune_mode)
            if args.mode == 'smoke':
                with torch.no_grad():
                    after = pilot.selected_logps(actor, groups[0]['prompt_token_ids'], first['token_ids']).cpu()
                frozen_after = pilot.features(reference, groups[0]['prompt_token_ids'], first['token_ids'], [])[0]
                changed = float((after - before).abs().max())
                fixed = float((frozen_after - frozen_before).abs().max())
                fresh = client.generate(pilot.request(tok, batch, 'fresh_policy', policy_checkpoint, step + 2, seed,
                                        cap=args.smoke_cap, n=args.rollouts, finetune_mode=args.finetune_mode), timeout=args.generation_timeout)
                differences = []
                for gr in fresh['rows']:
                    for sample in gr['samples']:
                        with torch.no_grad():
                            hf = pilot.selected_logps(actor, gr['prompt_token_ids'], sample['token_ids']).cpu()
                        differences.append((hf - torch.tensor(sample['logprobs'])).abs())
                err = torch.cat(differences)
                fingerprint = trainable_fingerprint(actor)
                fingerprints = [None] * world_size
                dist.all_gather_object(fingerprints, fingerprint)
                smoke = dict(adapter_changed_logps=changed, fixed_reference_maxdiff=fixed,
                             fresh_adapter_abs_logp_mean=float(err.mean()),
                             fresh_adapter_abs_logp_p99=float(torch.quantile(err, .99)),
                             trainable_sha256=fingerprint,
                             all_rank_trainable_weights_equal=len(set(fingerprints)) == 1)
                pilot.write(rank_out / 'smoke_checks.json', smoke)
                pilot.write(rank_out / 'fresh_sampler.json', fresh)
                if not (changed > 1e-5 and fixed == 0 and stats['actor_behavior_abs_logp_mean'] < .05 and float(err.mean()) < .05
                        and len(set(fingerprints)) == 1):
                    raise AssertionError(smoke)
            local = diagnostics(groups, args.arm)
            local.update(generation_seconds=generated['generation_seconds'], annotation_seconds=annotation_seconds,
                         full_model_refresh_seconds=generated.get('full_model_refresh_seconds', 0.))
            all_local = [None] * world_size
            dist.all_gather_object(all_local, local)
            if rank == 0:
                totals = {name: sum(d[name] for d in all_local) for name in local
                          if name not in ('generation_seconds', 'annotation_seconds', 'full_model_refresh_seconds')}
                stats.update(step=step + 1, arm=args.arm, mode=args.mode, finetune_mode=args.finetune_mode,
                             policy_checkpoint=policy_checkpoint, adapter=policy_checkpoint if args.finetune_mode == 'lora' else None,
                             format_coefficient=0.,
                             wall_seconds=time.time() - began,
                             generation_seconds=max(d['generation_seconds'] for d in all_local),
                             annotation_seconds=max(d['annotation_seconds'] for d in all_local),
                             full_model_refresh_seconds=max(d['full_model_refresh_seconds'] for d in all_local))
                progress = ((step + 1) * world_size / len(rows) if args.mode == 'smoke'
                            else config.epoch_progress(step + 1, len(rows), args.batch_size))
                logged = metrics(stats, totals, epoch=progress)
                pilot.write(out / f'update_{step + 1:03d}.json', dict(**stats, totals=totals, metrics=logged))
                if run is not None:
                    run.log(logged, step=step + 1)
                print(json.dumps(stats), flush=True)
            dist.barrier()
            checkpoint_began = time.time()
            state_checkpoint = checkpoints.save_distributed_state(out, step + 1, actor, optimizer, dist,
                                 contract, device=torch.device('cuda', local_rank), finetune_mode=args.finetune_mode)
            if rank == 0:
                submit_validation(validation, run, completed_step=step + 1,
                                  checkpoint=policy_checkpoint, per_epoch=per_epoch)
                if args.resume_state_retention == 'latest-epoch':
                    removed_states = checkpoints.retain_latest_epoch_states(out, step + 1, per_epoch)
                    removed = []
                else:
                    removed_states = []
                    removed = checkpoints.prune_complete_pairs(out, step + 1, pinned_steps)
                pilot.write(out / f'checkpoint_{step + 1:03d}.json',
                            dict(completed_step=step + 1, state_checkpoint=state_checkpoint,
                                 checkpoint_seconds=time.time() - checkpoint_began,
                                 checkpoint_retention=retention,
                                 resume_state_retention=args.resume_state_retention,
                                 pruned_state_steps=removed_states,
                                 pruned_steps=removed))
            dist.barrier()
        if rank == 0:
            pilot.write(out / 'done.json', dict(complete=True, steps=steps, finetune_mode=args.finetune_mode,
                       final_policy_checkpoint=policy_checkpoint,
                       final_model_checkpoint=policy_checkpoint if args.finetune_mode == 'full' else None,
                       final_adapter=policy_checkpoint if args.finetune_mode == 'lora' else None,
                       final_state_checkpoint=state_checkpoint, start_step=start_step,
                       checkpoint_retention=retention,
                       resume_state_retention=args.resume_state_retention,
                       epochs=(steps * world_size / len(rows) if args.mode == 'smoke'
                               else config.epoch_progress(steps, len(rows), args.batch_size)), ended=time.time(),
                       validation=validation.summary(),
                       wandb_url=run.url if run is not None else None))
        training_complete = True
    finally:
        if client is not None:
            client.close()
        if run is not None:
            run.finish(exit_code=0 if training_complete else 1)
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
