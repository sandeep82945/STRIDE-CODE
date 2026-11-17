"""Distributed policy wrapping and immutable full-model/LoRA exports.

Full training keeps sharded FP32 parameters and Adam states on the learner GPUs;
FSDP casts the gathered parameters to BF16 for forward/backward computation.
Exports retain FP32 parameters so resuming does not round the master weights.
"""
from __future__ import annotations

from contextlib import nullcontext
from functools import partial
import os
from pathlib import Path


def _check_mode(finetune_mode):
    if finetune_mode not in ('full', 'lora'):
        raise ValueError('finetune_mode must be full or lora')
    return finetune_mode


def policy_mode(actor):
    """Read the mode attached by wrap_actor (legacy DDP actors use LoRA)."""
    return _check_mode(getattr(actor, '_stride_finetune_mode', 'lora'))


def wrap_actor(model_actor, local_rank, finetune_mode='full'):
    """All learner ranks call with identically initialized model parameters."""
    import torch
    finetune_mode = _check_mode(finetune_mode)
    if finetune_mode == 'lora':
        from torch.nn.parallel import DistributedDataParallel
        actor = DistributedDataParallel(
            model_actor, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=False, broadcast_buffers=False)
    else:
        from torch.distributed.fsdp import (
            FullyShardedDataParallel, MixedPrecision, ShardingStrategy)
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

        # Auto-wrap decoder blocks rather than gathering the entire 4B model
        # for each rollout. Use the concrete model's declared block names;
        # fail explicitly for unsupported architectures instead of silently
        # falling back to a single enormous FSDP unit.
        block_names = set(getattr(model_actor, '_no_split_modules', None) or ())
        block_types = {type(module) for module in model_actor.modules()
                       if type(module).__name__ in block_names
                       and ('DecoderLayer' in type(module).__name__
                            or 'Block' in type(module).__name__)}
        if not block_types:
            raise ValueError('Full training requires identifiable transformer decoder blocks')
        for parameter in model_actor.parameters():
            if parameter.is_floating_point() and parameter.dtype != torch.float32:
                raise ValueError('Full training must initialize FP32 master parameters')
            if not parameter.requires_grad:
                raise ValueError('Full training requires all model parameters to be trainable')
        actor = FullyShardedDataParallel(
            model_actor,
            auto_wrap_policy=partial(transformer_auto_wrap_policy,
                                     transformer_layer_cls=block_types),
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,
                                           reduce_dtype=torch.bfloat16,
                                           buffer_dtype=torch.bfloat16),
            device_id=torch.device('cuda', local_rank),
            sync_module_states=True,
            use_orig_params=True,
            limit_all_gathers=True,
        )
    actor._stride_finetune_mode = finetune_mode
    return actor


def forward_context(actor):
    """Autocast full-policy compute while retaining FP32 master parameters."""
    if policy_mode(actor) == 'full':
        import torch
        return torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    return nullcontext()


def save_policy(actor, folder, tokenizer=None, finetune_mode='full'):
    """All ranks export one immutable policy and wait for its publication.

    Full state gathering is collective even though only rank zero writes. A
    full-policy export includes the tokenizer so it can be loaded independently
    by the sampler. The sampler explicitly casts these FP32 weights to BF16.
    """
    import torch.distributed as dist
    import resumable_checkpoint as checkpoints

    finetune_mode = _check_mode(finetune_mode)
    if policy_mode(actor) != finetune_mode:
        raise ValueError('Actor wrapping and checkpoint finetune_mode disagree')
    if finetune_mode == 'full' and tokenizer is None:
        raise ValueError('Full-policy exports require the tokenizer')
    rank = dist.get_rank()
    folder = Path(folder)
    temporary = folder.with_name(folder.name + '.writing')
    manifest = folder.with_name(folder.name + '_manifest.json')
    status = [None]
    if rank == 0:
        try:
            if folder.exists() or temporary.exists() or manifest.exists():
                raise FileExistsError(f'Policy checkpoints are immutable: {folder}')
            temporary.mkdir()
        except Exception as error:
            status[0] = f'{type(error).__name__}: {error}'
    dist.broadcast_object_list(status, src=0)
    if status[0] is not None:
        raise RuntimeError(f'Cannot prepare policy export: {status[0]}')

    state = None
    if finetune_mode == 'full':
        from torch.distributed.fsdp import (
            FullyShardedDataParallel, FullStateDictConfig, StateDictType)
        with FullyShardedDataParallel.state_dict_type(
                actor, StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
            state = actor.state_dict()
    if rank == 0:
        try:
            if finetune_mode == 'full':
                actor.module.save_pretrained(temporary, state_dict=state,
                                             safe_serialization=True,
                                             max_shard_size='5GB')
                tokenizer.save_pretrained(temporary)
            else:
                actor.module.save_pretrained(temporary, safe_serialization=True)
            files = list(temporary.iterdir())
            if not files or any(p.is_symlink() or not p.is_file() for p in files):
                raise ValueError('Policy exports must contain only regular, flat files')
            hashes = {p.name: checkpoints.digest(p) for p in sorted(files)}
            for path in files:
                with path.open('rb') as stream:
                    os.fsync(stream.fileno())
            checkpoints._fsync_directory(temporary)
            temporary.rename(folder)
            checkpoints._fsync_directory(folder.parent)
            checkpoints.write_json(manifest, hashes)
        except Exception as error:
            status[0] = f'{type(error).__name__}: {error}'
    del state
    # Also propagate rank-zero filesystem errors to the other learners instead
    # of leaving them waiting at a post-save barrier indefinitely.
    dist.broadcast_object_list(status, src=0)
    if status[0] is not None:
        raise RuntimeError(f'Cannot publish policy export: {status[0]}')
    dist.barrier()
    return str(folder)
