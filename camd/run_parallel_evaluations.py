#!/usr/bin/env python3
"""Snapshot host GRES evidence, run three GPU evaluations, and gate on the host."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path('/path/to/cluster-user/system/stride_sample100_20260915')
ARMS = ('single_step', 'token_uniform', 'grpo')
NODE = 'auh7-1b-gpu-194'


def write_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    job = os.environ['SLURM_JOB_ID']
    if not job.isascii() or not job.isdigit() or os.environ.get('SLURMD_NODENAME') != NODE:
        raise ValueError('Parallel evaluation requires a numeric job on the verified node')
    directory = ROOT / 'results' / f'job-{job}' / 'parallel_evaluations'
    directory.mkdir(parents=True, exist_ok=False)
    gres = Path('/etc/slurm/gres.conf').read_bytes()
    if len(gres) > 65536:
        raise ValueError('Unexpectedly large GRES configuration')
    snapshot = directory / 'gres_configuration.json'
    write_new(snapshot, {'job_id': job, 'node': NODE, 'source': '/etc/slurm/gres.conf',
                         'content': gres.decode('utf-8'), 'sha256': hashlib.sha256(gres).hexdigest()})
    code = ROOT / 'code/stride_handoff/camd'
    process = subprocess.run([
        'srun', '--exact', '--ntasks=1', '--cpus-per-task=24', '--gres=gpu:4', '--mem=192G',
        '/bin/bash', str(code / 'enter_runtime.sh'), 'gpu-parallel',
        str(code / 'parallel_eval_workers.py'), '--gres-evidence', str(snapshot)], check=False)
    report = {'job_id': job, 'arms': list(ARMS), 'passed': False,
              'srun_returncode': process.returncode, 'host_gate_returncodes': {}}
    # Torch can return zero after a native failure. Every arm needs an explicit
    # independent host gate even if its GPU worker claimed successful completion.
    for arm in ARMS:
        gate = subprocess.run([sys.executable, '-I', '-S', str(code / 'host_gate.py'),
                               job, 'eval', arm], check=False)
        report['host_gate_returncodes'][arm] = gate.returncode
    try:
        workers = json.loads((directory / 'workers.json').read_text())
        report['worker_artifacts_verified'] = (
            workers.get('job_id') == job and workers.get('passed') is True
            and set(workers.get('children', {})) == set(ARMS))
    except (OSError, ValueError):
        report['worker_artifacts_verified'] = False
    report['passed'] = (process.returncode == 0 and report['worker_artifacts_verified']
                        and all(value == 0 for value in report['host_gate_returncodes'].values()))
    write_new(directory / 'host_verification.json', report)
    print('PARALLEL_EVALUATIONS_RESULT', json.dumps(report), flush=True)
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
