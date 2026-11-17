"""Bind a one-GPU Slurm step to its permitted ROCm device before Python imports.

Slurm GRES ordinals need not match ROCr ordinals. Use the unique GPU UUID of
the sole accessible render node; leave eight-GPU training steps unchanged.
"""
import json
import os
from pathlib import Path
import re
import sys

from allocation_probe_v2 import device_inventory, topology_inventory, number

ROOT = Path('/path/to/cluster-user/system/stride_full2048_10epoch_20260915/results')
MASKS = ('ROCR_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES',
         'CUDA_VISIBLE_DEVICES', 'GPU_DEVICE_ORDINAL')
METADATA = ('SLURM_JOB_ID', 'SLURM_STEP_ID', 'SLURM_JOB_GPUS',
            'SLURM_STEP_GPUS', 'SLURMD_NODENAME')


def gpu_ids(value):
    if not isinstance(value, str) or not re.fullmatch(r'[0-9]+(?:,[0-9]+)*', value):
        raise ValueError('Requires explicit numeric Slurm GPU IDs')
    values = [int(item) for item in value.split(',')]
    if len(values) != len(set(values)):
        raise ValueError('Duplicate Slurm GPU IDs')
    return values


def step_size(env):
    for name in ('SLURM_JOB_ID', 'SLURM_STEP_ID'):
        if not re.fullmatch(r'[0-9]+', env.get(name, '')):
            raise ValueError('GPU bootstrap requires a numeric Slurm job and srun step')
    job = gpu_ids(env.get('SLURM_JOB_GPUS'))
    step = gpu_ids(env.get('SLURM_STEP_GPUS'))
    if len(step) not in (1, 8) or not set(step) <= set(job):
        raise ValueError('Expected one or eight GPUs contained in the Slurm job allocation')
    return len(step)


def permitted_gpu(devices, topology):
    kfd = [item for item in devices if item.get('path') == '/dev/kfd']
    if len(kfd) != 1 or kfd[0].get('open_read_write') is not True:
        raise ValueError('Allocated process cannot open /dev/kfd')
    permitted = [item for item in devices
                 if re.fullmatch(r'/dev/dri/renderD[0-9]+', item.get('path', ''))
                 and item.get('character_device') is True
                 and item.get('open_read_write') is True]
    if len(permitted) != 1:
        raise ValueError('One-GPU binding requires exactly one permitted render device')
    render = permitted[0]
    matches = [node for node in topology
               if number(node.get('properties', {}).get('drm_render_minor')) == render['minor']
               and number(node.get('gpu_id')) not in (None, 0)]
    if len(matches) != 1:
        raise ValueError('Permitted render device must map to exactly one KFD GPU')
    node = matches[0]
    identity = number(node.get('properties', {}).get('unique_id'))
    if identity is None or not 0 < identity < 2**64:
        raise ValueError('Permitted GPU has no valid 64-bit unique_id')
    return {'uuid': 'GPU-' + format(identity, '016x'),
            'render_device': render, 'kfd_device': kfd[0], 'topology': node}


def normalize(env, devices=None, topology=None):
    """Pure decision helper. Returns a copied environment and mapping evidence."""
    count = step_size(env)
    before = {name: env.get(name) for name in (*METADATA, *MASKS)}
    updated = dict(env)
    evidence = {'original_environment': before, 'allocated_gpus': count,
                'gpu_execution_tested': False, 'mapping_verified': False}
    if count == 8:
        evidence['action'] = 'eight_gpu_passthrough'
        return updated, evidence
    if devices is None or topology is None:
        raise ValueError('One-GPU binding requires device access and topology evidence')
    mapping = permitted_gpu(devices, topology)
    updated.update(ROCR_VISIBLE_DEVICES=mapping['uuid'],
                   HIP_VISIBLE_DEVICES='0', CUDA_VISIBLE_DEVICES='0')
    updated.pop('GPU_DEVICE_ORDINAL', None)
    evidence.update(action='one_gpu_uuid_binding', mapping_verified=True,
                    mapping=mapping,
                    normalized_environment={name: updated.get(name) for name in MASKS})
    return updated, evidence


def save_evidence(env, evidence):
    directory = ROOT / ('job-' + env['SLURM_JOB_ID'])
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ('allocation-step-' + env['SLURM_STEP_ID'] + '.json')
    # A Slurm step is used once. Never overwrite evidence from an earlier launch.
    with path.open('x') as stream:
        json.dump(evidence, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return path


def main():
    arguments = sys.argv[1:]
    if len(arguments) < 2 or arguments[0] != '--':
        raise ValueError('Usage: allocated_gpu.py -- SCRIPT [ARGUMENTS...]')
    env = dict(os.environ)
    count = step_size(env)
    devices = device_inventory() if count == 1 else None
    topology = topology_inventory() if count == 1 else None
    updated, evidence = normalize(env, devices, topology)
    if count == 1:
        evidence['uid'] = os.getuid()
        path = save_evidence(env, evidence)
        print('ALLOCATED_GPU_UUID', json.dumps({'uuid': evidence['mapping']['uuid'],
              'render_device': evidence['mapping']['render_device']['path'],
              'evidence': str(path)}), flush=True)
    os.environ.clear()
    os.environ.update(updated)
    os.execv(sys.executable, [sys.executable, '-u', *arguments[1:]])


if __name__ == '__main__':
    main()
