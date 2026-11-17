"""Immutable policy/AdamW/RNG checkpoint pairs, committed only by COMPLETE.json.

This module performs no model training and never mutates a source checkpoint.
Torch is imported lazily so filesystem validation also works on CPU-only hosts.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil


FORMAT_VERSION = 2
STATE_NAME = re.compile(r'state_(\d{3,})\Z')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def _regular(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'Missing or unsafe checkpoint file: {path}')
    return path


def _directory(path):
    path = Path(path)
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f'Missing or unsafe checkpoint directory: {path}')
    return path


def _fsync_directory(path):
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.writing')
    if path.exists() or temporary.exists():
        raise FileExistsError(f'Checkpoint files are immutable: {path}')
    with temporary.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    # Publish without replacement even if another writer creates the final
    # path after our initial check. A failed write remains as crash evidence.
    os.link(temporary, path)
    temporary.unlink()
    _fsync_directory(path.parent)


def save_torch(path, value):
    import torch
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    with path.open('xb') as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())


def capture_rng(device=None):
    import numpy as np
    import torch
    numpy_state = np.random.get_state()
    # Primitive types plus tensors permit weights_only=True on restore.
    result = dict(python=random.getstate(), torch_cpu=torch.get_rng_state(),
                  numpy=(numpy_state[0], numpy_state[1].tolist(), int(numpy_state[2]),
                         int(numpy_state[3]), float(numpy_state[4])), cuda=None)
    if device is not None:
        # Never initialize RNG contexts on the four sampler GPUs.
        result['cuda'] = torch.cuda.get_rng_state(device).cpu()
    return result


def restore_rng(state, device=None):
    import numpy as np
    import torch
    if (state['cuda'] is None) != (device is None):
        raise ValueError('Saved CUDA RNG contract does not match the target device')
    random.setstate(state['python'])
    a, b, c, d, e = state['numpy']
    np.random.set_state((a, np.asarray(b, dtype=np.uint32), c, d, e))
    torch.set_rng_state(state['torch_cpu'])
    if device is not None:
        torch.cuda.set_rng_state(state['cuda'], device)


def verify_policy(policy, finetune_mode='full'):
    if finetune_mode not in ('full', 'lora'):
        raise ValueError('Invalid checkpoint finetune mode')
    policy = _directory(policy)
    manifest_path = _regular(policy.with_name(policy.name + '_manifest.json'))
    hashes = json.loads(manifest_path.read_text())
    if not isinstance(hashes, dict):
        raise ValueError('Policy manifest must map filenames to hashes')
    if finetune_mode == 'lora':
        if not {'adapter_model.safetensors', 'adapter_config.json'} <= hashes.keys():
            raise ValueError('Adapter manifest misses weights/config')
    else:
        if not {'config.json', 'tokenizer_config.json'} <= hashes.keys():
            raise ValueError('Full-policy manifest misses model/tokenizer configuration')
        if not ('tokenizer.json' in hashes or 'tokenizer.model' in hashes
                or {'vocab.json', 'merges.txt'} <= hashes.keys()):
            raise ValueError('Full-policy manifest misses tokenizer vocabulary')
        if 'model.safetensors' not in hashes and 'model.safetensors.index.json' not in hashes:
            raise ValueError('Full-policy manifest misses weights')
    if set(hashes) != {p.name for p in policy.iterdir()}:
        raise ValueError('Policy directory and manifest disagree')
    for name, expected in hashes.items():
        if Path(name).name != name or name in ('.', '..'):
            raise ValueError('Unsafe policy manifest member')
        if digest(_regular(policy / name)) != expected:
            raise ValueError(f'Policy hash mismatch: {name}')
    if finetune_mode == 'full' and 'model.safetensors.index.json' in hashes:
        index = json.loads((policy / 'model.safetensors.index.json').read_text())
        weight_map = index.get('weight_map', {})
        if (not isinstance(weight_map, dict) or not weight_map
                or not all(isinstance(name, str) and name.endswith('.safetensors')
                           and name in hashes for name in weight_map.values())):
            raise ValueError('Full-policy weights index references missing or unsafe shards')
    return dict(files=hashes, manifest_sha256=digest(manifest_path))


def verify_adapter(adapter):
    """Compatibility entry point for immutable legacy LoRA adapters."""
    return verify_policy(adapter, 'lora')


def _state_names(world_size, finetune_mode):
    if finetune_mode not in ('full', 'lora'):
        raise ValueError('Invalid checkpoint finetune mode')
    optimizer_names = ({f'optimizer_rank_{rank}.pt' for rank in range(world_size)}
                       if finetune_mode == 'full' else {'optimizer.pt'})
    return optimizer_names | {f'rng_rank_{rank}.pt' for rank in range(world_size)}


def _check_contract_mode(contract, finetune_mode):
    if not isinstance(contract, dict):
        raise ValueError('Checkpoint training contract must be a mapping')
    declared = contract.get('finetune_mode')
    # Older LoRA checkpoint contracts predate this field. A full-model
    # checkpoint must always explicitly identify its new training regime.
    if declared is None and finetune_mode == 'lora':
        return
    if declared != finetune_mode:
        raise ValueError('Checkpoint finetune mode disagrees with its training contract')


def prepare_state(out, step):
    out = _directory(out)
    if type(step) is not int or step < 0:
        raise ValueError('Invalid completed step')
    final = out / f'state_{step:03d}'
    stage = out / (final.name + '.writing')
    if final.exists() or stage.exists():
        raise FileExistsError('State checkpoints are immutable')
    stage.mkdir()
    _fsync_directory(out)
    return stage


def commit_state(out, step, world_size, contract, finetune_mode='lora'):
    """Rank zero calls only after every rank has durably saved its RNG file."""
    if type(step) is not int or step < 0:
        raise ValueError('Invalid completed step')
    if type(world_size) is not int or world_size < 1:
        raise ValueError('Invalid checkpoint world size')
    _check_contract_mode(contract, finetune_mode)
    out = _directory(out)
    stage = _directory(out / f'state_{step:03d}.writing')
    final = out / f'state_{step:03d}'
    if final.exists() or world_size < 1:
        raise ValueError('Invalid state commit')
    names = _state_names(world_size, finetune_mode)
    if {p.name for p in stage.iterdir()} != names:
        raise ValueError('Incomplete or unexpected optimizer/RNG state files')
    hashes = {name: digest(_regular(stage / name)) for name in sorted(names)}
    policy_name = f'{"model" if finetune_mode == "full" else "adapter"}_{step:03d}'
    policy = verify_policy(out / policy_name, finetune_mode)
    # The legacy adapter saver uses rename but not fsync. Make the entire pair
    # durable before exposing this stronger completion marker.
    for name in policy['files']:
        with (out / policy_name / name).open('rb') as stream:
            os.fsync(stream.fileno())
    with (out / (policy_name + '_manifest.json')).open('rb') as stream:
        os.fsync(stream.fileno())
    _fsync_directory(out / policy_name)
    metadata = dict(format_version=FORMAT_VERSION, completed_step=step,
                    world_size=world_size, contract=contract, state_files=hashes,
                    finetune_mode=finetune_mode, policy_name=policy_name, policy=policy,
                    optimizer_layout=('fsdp_local_v1' if finetune_mode == 'full'
                                      else 'replicated_v1'))
    if finetune_mode == 'lora':
        metadata.update(adapter_name=policy_name, adapter=policy)
    write_json(stage / 'metadata.json', metadata)
    stage.rename(final)
    _fsync_directory(out)
    # A crash before this last write always leaves an ineligible checkpoint.
    write_json(final / 'COMPLETE.json', dict(format_version=FORMAT_VERSION,
               completed_step=step, metadata_sha256=digest(final / 'metadata.json')))
    return final


def inspect_checkpoint(state, expected_contract=None, expected_world_size=None):
    state = _directory(state)
    match = STATE_NAME.fullmatch(state.name)
    if match is None:
        raise ValueError('Expected an immutable state_NNN checkpoint directory')
    complete = json.loads(_regular(state / 'COMPLETE.json').read_text())
    metadata_path = _regular(state / 'metadata.json')
    if complete.get('metadata_sha256') != digest(metadata_path):
        raise ValueError('Checkpoint metadata hash mismatch')
    meta = json.loads(metadata_path.read_text())
    step = int(match.group(1))
    version = meta.get('format_version')
    if (version not in (1, FORMAT_VERSION) or complete.get('format_version') != version
            or complete.get('completed_step') != step or meta.get('completed_step') != step):
        raise ValueError('Checkpoint completion/version/step mismatch')
    world_size = meta.get('world_size')
    if type(world_size) is not int or world_size < 1:
        raise ValueError('Invalid checkpoint world size')
    if expected_world_size is not None and world_size != expected_world_size:
        raise ValueError('Cannot change learner world size while resuming')
    if expected_contract is not None and meta.get('contract') != expected_contract:
        raise ValueError('Training contract changed while resuming')
    # Version 1 exclusively stored LoRA adapters with replicated optimizer
    # states. Version 2 declares policy mode and optimizer layout explicitly.
    finetune_mode = 'lora' if version == 1 else meta.get('finetune_mode')
    names = _state_names(world_size, finetune_mode)
    _check_contract_mode(meta.get('contract'), finetune_mode)
    if version == FORMAT_VERSION:
        expected_layout = 'fsdp_local_v1' if finetune_mode == 'full' else 'replicated_v1'
        if meta.get('optimizer_layout') != expected_layout:
            raise ValueError('Checkpoint optimizer layout disagrees with finetune mode')
    if set(meta.get('state_files', {})) != names:
        raise ValueError('Missing optimizer or per-rank RNG state')
    if {p.name for p in state.iterdir()} != names | {'metadata.json', 'COMPLETE.json'}:
        raise ValueError('Unexpected or incomplete state directory')
    for name, expected in meta['state_files'].items():
        if digest(_regular(state / name)) != expected:
            raise ValueError(f'Checkpoint state hash mismatch: {name}')
    policy_name = meta.get('adapter_name') if version == 1 else meta.get('policy_name')
    expected_name = f'{"model" if finetune_mode == "full" else "adapter"}_{step:03d}'
    if policy_name != expected_name:
        raise ValueError('Checkpoint references an unexpected policy path')
    policy = state.parent / policy_name
    provenance = meta.get('adapter') if version == 1 else meta.get('policy')
    if verify_policy(policy, finetune_mode) != provenance:
        raise ValueError('Checkpoint policy provenance mismatch')
    if version == FORMAT_VERSION and finetune_mode == 'lora':
        if meta.get('adapter_name') != policy_name or meta.get('adapter') != provenance:
            raise ValueError('Checkpoint adapter and policy metadata disagree')
    result = dict(meta, state_path=str(state.resolve()),
                  policy_path=str(policy.resolve()), finetune_mode=finetune_mode)
    if finetune_mode == 'lora':
        result['adapter_path'] = str(policy.resolve())
    return result


def load_training_state(checkpoint, optimizer, rank, device=None):
    """Call after wrapping/model/optimizer initialization; restore RNG last.

    Full-mode optimizer state is a raw local FSDP shard, restored only with the
    same world size, wrapping code, parameter order, and scientific contract.
    It is deliberately not a portable/repartitionable optimizer checkpoint.
    """
    import torch
    # Revalidate at the moment of deserialization, not just before slow startup.
    meta = inspect_checkpoint(checkpoint['state_path'], checkpoint['contract'], checkpoint['world_size'])
    if type(rank) is not int or not 0 <= rank < meta['world_size']:
        raise ValueError('Rank outside checkpoint topology')
    root = Path(meta['state_path'])
    optimizer_name = (f'optimizer_rank_{rank}.pt'
                      if meta['finetune_mode'] == 'full' else 'optimizer.pt')
    optimizer.load_state_dict(torch.load(root / optimizer_name, map_location='cpu', weights_only=True))
    return torch.load(root / f'rng_rank_{rank}.pt', map_location='cpu', weights_only=True)


def save_distributed_state(out, step, actor, optimizer, dist, contract, device=None,
                           finetune_mode='lora'):
    """All ranks call after update and logging; returns a committed state path."""
    rank, world_size = dist.get_rank(), dist.get_world_size()
    _state_names(world_size, finetune_mode)
    _check_contract_mode(contract, finetune_mode)
    from policy_runtime import policy_mode
    if policy_mode(actor) != finetune_mode:
        raise ValueError('Actor wrapping and optimizer checkpoint finetune mode disagree')
    if rank == 0:
        prepare_state(out, step)
    dist.barrier()
    stage = Path(out) / f'state_{step:03d}.writing'
    if finetune_mode == 'full':
        # FSDP's optimizer owns only this rank's FP32 parameter shards. Persist
        # each raw optimizer state without gathering all Adam states on rank 0.
        save_torch(stage / f'optimizer_rank_{rank}.pt', optimizer.state_dict())
    elif rank == 0:
        save_torch(stage / 'optimizer.pt', optimizer.state_dict())
    save_torch(stage / f'rng_rank_{rank}.pt', capture_rng(device))
    dist.barrier()
    if rank == 0:
        commit_state(out, step, world_size, contract, finetune_mode)
    dist.barrier()
    return str(Path(out) / f'state_{step:03d}')


def prune_complete_pairs(out, current_step, pinned_steps):
    """Only this attempt's verified complete pairs; incomplete evidence stays."""
    out = _directory(out).resolve()
    removed = []
    for state in sorted(out.glob('state_*')):
        match = STATE_NAME.fullmatch(state.name)
        if match is None or not (state / 'COMPLETE.json').is_file():
            continue
        step = int(match.group(1))
        if step >= current_step - 1 or step in pinned_steps:
            continue
        meta = inspect_checkpoint(state)
        policy = Path(meta['policy_path'])
        if state.is_symlink() or state.parent.resolve() != out or policy.parent != out:
            raise ValueError('Refusing to prune a checkpoint outside this attempt')
        # Remove eligibility first. Keep policy manifests for provenance audits.
        (state / 'COMPLETE.json').unlink()
        shutil.rmtree(state)
        shutil.rmtree(policy)
        removed.append(step)
    _fsync_directory(out)
    return removed


def inspect_checkpoint_receipt(out, step, expected_contract=None, expected_world_size=None):
    """Authenticate an archived LoRA commit and its retained adapter, not Adam/RNG.

    A receipt preserves the exact original metadata and completion bytes. It
    proves which policy was committed but cannot substitute for a resume state.
    """
    out = _directory(out).resolve()
    if type(step) is not int or step < 0:
        raise ValueError('Invalid receipt step')
    receipts = _directory(out / 'checkpoint_receipts')
    receipt = _directory(receipts / f'state_{step:03d}')
    if {p.name for p in receipt.iterdir()} != {'metadata.json', 'COMPLETE.json'}:
        raise ValueError('Unexpected checkpoint receipt files')
    marker = json.loads(_regular(receipt / 'COMPLETE.json').read_text())
    metadata_path = _regular(receipt / 'metadata.json')
    if marker.get('metadata_sha256') != digest(metadata_path):
        raise ValueError('Checkpoint receipt metadata hash mismatch')
    meta = json.loads(metadata_path.read_text())
    if (marker.get('format_version') != FORMAT_VERSION or meta.get('format_version') != FORMAT_VERSION
            or marker.get('completed_step') != step or meta.get('completed_step') != step
            or meta.get('finetune_mode') != 'lora' or meta.get('optimizer_layout') != 'replicated_v1'):
        raise ValueError('Checkpoint receipt version/step/mode mismatch')
    world = meta.get('world_size')
    if type(world) is not int or world < 1 or (expected_world_size is not None and world != expected_world_size):
        raise ValueError('Checkpoint receipt world size mismatch')
    _check_contract_mode(meta.get('contract'), 'lora')
    if expected_contract is not None and meta.get('contract') != expected_contract:
        raise ValueError('Checkpoint receipt training contract mismatch')
    hashes = meta.get('state_files', {})
    if set(hashes) != _state_names(world, 'lora') or any(not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value)
                                                       for value in hashes.values()):
        raise ValueError('Checkpoint receipt optimizer/RNG inventory mismatch')
    name = f'adapter_{step:03d}'
    if meta.get('policy_name') != name or meta.get('adapter_name') != name:
        raise ValueError('Checkpoint receipt policy path mismatch')
    policy = verify_policy(out / name, 'lora')
    if meta.get('policy') != policy or meta.get('adapter') != policy:
        raise ValueError('Checkpoint receipt policy provenance mismatch')
    return dict(meta, receipt_path=str(receipt), policy_path=str(out / name), adapter_path=str(out / name),
                optimizer_payload_verified=False, resumable=False)


def archive_checkpoint_receipt(out, step):
    """Publish immutable byte-identical small commit receipts before state pruning."""
    out = _directory(out).resolve()
    state = out / f'state_{step:03d}'
    receipts = out / 'checkpoint_receipts'
    if not receipts.exists():
        receipts.mkdir()
        _fsync_directory(out)
    _directory(receipts)
    destination = receipts / state.name
    if destination.exists():
        inspect_checkpoint_receipt(out, step)
        for name in ('metadata.json', 'COMPLETE.json'):
            if _regular(destination / name).read_bytes() != _regular(state / name).read_bytes():
                raise ValueError('Existing checkpoint receipt differs from its live state')
        return destination
    inspected = inspect_checkpoint(state)
    if inspected['finetune_mode'] != 'lora':
        raise ValueError('State-only retention supports LoRA checkpoints only')
    stage = destination.with_name(destination.name + '.writing')
    stage.mkdir()
    for name in ('metadata.json', 'COMPLETE.json'):
        with _regular(state / name).open('rb') as source, (stage / name).open('xb') as target:
            target.write(source.read())
            target.flush()
            os.fsync(target.fileno())
    _fsync_directory(stage)
    stage.rename(destination)
    _fsync_directory(receipts)
    inspect_checkpoint_receipt(out, step)
    return destination


def retain_latest_epoch_states(out, current_step, per_epoch):
    """Keep all LoRA policies/receipts, only the latest epoch and latest update states.

    The latest completed epoch is zero until epoch one finishes. Incomplete
    state directories remain as crash evidence. Only this attempt is pruned;
    immutable resume-source attempts are never traversed or changed.
    """
    out = _directory(out).resolve()
    if type(current_step) is not int or current_step < 0 or type(per_epoch) is not int or per_epoch < 1:
        raise ValueError('Invalid latest-epoch retention schedule')
    archive_checkpoint_receipt(out, current_step)
    keep = {current_step, current_step // per_epoch * per_epoch}
    removed = []
    for state in sorted(out.glob('state_*')):
        match = STATE_NAME.fullmatch(state.name)
        if match is None or not (state / 'COMPLETE.json').is_file():
            continue
        step = int(match.group(1))
        if step in keep or step > current_step:
            continue
        if state.is_symlink() or state.parent.resolve() != out:
            raise ValueError('Refusing to prune state outside this attempt')
        archive_checkpoint_receipt(out, step)
        # Keep every adapter and external policy manifest. Removing the marker
        # first prevents another reader from treating partial deletion as live.
        (state / 'COMPLETE.json').unlink()
        shutil.rmtree(state)
        removed.append(step)
    _fsync_directory(out)
    return removed
