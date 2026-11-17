#!/usr/bin/env python3
"""Run three evaluations on distinct real GPUs in a single four-GRES allocation.

The verified node counts /dev/kfd as GRES slot zero. Require that slot plus
three configured render slots; never use the extra render device permitted by
the cgroup unless its own configured slot is included in the allocation.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from allocated_gpu import gpu_ids
from allocation_probe_v2 import device_inventory, topology_inventory, number
from run_evaluation_arm import evaluation_command

ROOT = Path('/path/to/cluster-user/system/stride_sample100_20260915')
NODE = 'auh7-1b-gpu-194'
ARMS = ('single_step', 'token_uniform', 'grpo')
MASKS = ('ROCR_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES', 'GPU_DEVICE_ORDINAL')


def configured_devices(text):
    devices = []
    for line in text.splitlines():
        tokens = shlex.split(line, comments=True)
        if not tokens:
            continue
        fields = {}
        for token in tokens:
            key, separator, value = token.partition('=')
            if not separator or key in fields:
                raise ValueError('Unsupported or ambiguous GRES configuration')
            fields[key] = value
        if set(fields) != {'Name', 'Type', 'File'} or fields['Name'] != 'gpu' or fields['Type'] != 'mi210':
            raise ValueError('GRES configuration changed from the verified MI210 layout')
        devices.append(fields['File'])
    expected = ['/dev/kfd', *(f'/dev/dri/renderD{minor}' for minor in range(128, 136))]
    if devices != expected:
        raise ValueError('GRES slot-to-device order differs from the verified node')
    return devices


def bindings(env, snapshot, devices, topology):
    job, step = env.get('SLURM_JOB_ID', ''), env.get('SLURM_STEP_ID', '')
    if any(not value.isascii() or not value.isdigit() for value in (job, step)):
        raise ValueError('Expected numeric job and srun step identities')
    if env.get('SLURMD_NODENAME') != NODE or snapshot.get('node') != NODE or snapshot.get('job_id') != job:
        raise ValueError('Host GRES snapshot does not identify this allocation on the verified node')
    content = snapshot['content']
    if snapshot.get('source') != '/etc/slurm/gres.conf' or hashlib.sha256(content.encode()).hexdigest() != snapshot.get('sha256'):
        raise ValueError('Host GRES snapshot integrity check failed')
    gres = configured_devices(content)
    job_slots, step_slots = gpu_ids(env.get('SLURM_JOB_GPUS')), gpu_ids(env.get('SLURM_STEP_GPUS'))
    if len(job_slots) != 4 or set(step_slots) != set(job_slots) or 0 not in step_slots:
        raise ValueError('Require one four-slot step containing kfd slot zero and three render slots')
    if any(slot >= len(gres) for slot in step_slots):
        raise ValueError('Allocated slot is not in the recorded GRES configuration')
    by_path = {item['path']: item for item in devices}
    if len(by_path) != len(devices):
        raise ValueError('Ambiguous device inventory')
    kfd = by_path.get('/dev/kfd', {})
    if kfd.get('character_device') is not True or kfd.get('open_read_write') is not True:
        raise ValueError('Allocated process cannot open /dev/kfd')
    selected = []
    for arm, slot in zip(ARMS, sorted(set(step_slots) - {0})):
        render = by_path.get(gres[slot], {})
        if render.get('character_device') is not True or render.get('open_read_write') is not True:
            raise ValueError('Selected allocated render device is inaccessible')
        expected_minor = int(Path(gres[slot]).name.removeprefix('renderD'))
        if render.get('minor') != expected_minor:
            raise ValueError('Selected device minor disagrees with its configured render identity')
        matches = [node for node in topology
                   if number(node.get('properties', {}).get('drm_render_minor')) == expected_minor
                   and number(node.get('gpu_id')) not in (None, 0)]
        if len(matches) != 1:
            raise ValueError('Allocated render device lacks exactly one KFD GPU mapping')
        identity = number(matches[0].get('properties', {}).get('unique_id'))
        if identity is None or not 0 < identity < 2**64:
            raise ValueError('Allocated GPU lacks a valid unique hardware identity')
        selected.append({'arm': arm, 'gres_slot': slot, 'render_device': render,
                         'topology': matches[0], 'uuid': 'GPU-' + format(identity, '016x')})
    if len(selected) != 3 or len({item['uuid'] for item in selected}) != 3:
        raise ValueError('Parallel workers require three distinct allocated GPU identities')
    return selected


def launch_children(plans, root=ROOT):
    """Start every worker before waiting; retain healthy arms if another fails."""
    children = {}
    with ExitStack() as streams:
        try:
            for plan in plans:
                stdout = streams.enter_context(Path(plan['stdout']).open('x'))
                stderr = streams.enter_context(Path(plan['stderr']).open('x'))
                children[plan['arm']] = subprocess.Popen(
                    plan['command'], env=plan['environment'], cwd=root / 'code/stride_handoff',
                    stdout=stdout, stderr=stderr)
            return {arm: child.wait() for arm, child in children.items()}
        except BaseException:
            for child in children.values():
                if child.poll() is None:
                    child.terminate()
            for child in children.values():
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            raise


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gres-evidence', type=Path, required=True)
    cli = parser.parse_args()
    env = dict(os.environ)
    job = env['SLURM_JOB_ID']
    directory = ROOT / 'results' / f'job-{job}' / 'parallel_evaluations'
    if cli.gres_evidence != directory / 'gres_configuration.json' or cli.gres_evidence.is_symlink():
        raise ValueError('GRES evidence must be this job\'s host-created snapshot')
    snapshot = json.loads(cli.gres_evidence.read_text())
    devices, topology = device_inventory(), topology_inventory()
    selected = bindings(env, snapshot, devices, topology)
    submission = json.loads((ROOT / 'results/submission.json').read_text())
    report = {'job_id': job, 'step_id': env['SLURM_STEP_ID'], 'node': NODE, 'passed': False,
              'allocated_gres_slots': gpu_ids(env['SLURM_STEP_GPUS']), 'selected': selected,
              'devices': devices, 'topology': topology, 'children': {}}
    target = directory / 'workers.json'
    if target.exists():
        raise ValueError('Refusing to overwrite an earlier parallel worker report')
    plans = []
    for item in selected:
        arm = item['arm']
        child_env = dict(env)
        for mask in MASKS:
            child_env.pop(mask, None)
        child_env.update(ROCR_VISIBLE_DEVICES=item['uuid'], HIP_VISIBLE_DEVICES='0', CUDA_VISIBLE_DEVICES='0')
        temporary = Path(env['TMPDIR']) / arm
        temporary.mkdir(parents=True, exist_ok=False)
        child_env['TMPDIR'] = str(temporary)
        command = evaluation_command(arm, submission, ROOT)
        stage = str(ROOT / 'code/stride_handoff/camd/stage.py')
        command = [sys.executable, '-u', *command[command.index(stage):]]
        output = str(ROOT / 'evaluations' / f'{arm}-epoch4-{job}')
        plan = {'arm': arm, 'command': command, 'environment': child_env,
                'stdout': str(ROOT / 'logs' / f'job-{job}-{arm}.out'),
                'stderr': str(ROOT / 'logs' / f'job-{job}-{arm}.err')}
        plans.append(plan)
        report['children'][arm] = {key: value for key, value in plan.items() if key != 'environment'}
        report['children'][arm].update(output=output,
                                      gpu_environment={mask: child_env.get(mask) for mask in MASKS})
    write(target, report)
    print('PARALLEL_GPU_BINDINGS', json.dumps([
        {'arm': item['arm'], 'gres_slot': item['gres_slot'], 'uuid': item['uuid'],
         'render_device': item['render_device']['path']} for item in selected]), flush=True)
    try:
        returns = launch_children(plans)
        for arm, returncode in returns.items():
            child = report['children'][arm]
            child['returncode_informational_only'] = returncode
            stage_report = ROOT / 'results' / f'job-{job}' / f'eval-{arm}' / 'run.json'
            try:
                actual = json.loads(stage_report.read_text())
                child['explicit_completion_verified'] = (
                    actual.get('status') == 'VERIFIED' and actual.get('job_id') == job
                    and actual.get('stage') == 'eval' and actual.get('arm') == arm
                    and actual.get('output') == child['output'])
            except (OSError, ValueError):
                child['explicit_completion_verified'] = False
        report['passed'] = all(child['returncode_informational_only'] == 0
                               and child['explicit_completion_verified']
                               for child in report['children'].values())
    finally:
        write(target, report)
    print('PARALLEL_WORKERS_RESULT', json.dumps({'job_id': job, 'passed': report['passed']}), flush=True)
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
