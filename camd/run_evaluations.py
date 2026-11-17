#!/usr/bin/env python3
"""Evaluate the base and four final adapters in one allocated one-GPU job."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path('/path/to/cluster-user/system/stride_sample100_20260915')
CODE = ROOT / 'code/stride_handoff'
ARMS = ['local_positive', 'single_step', 'token_uniform', 'grpo']


def main():
    job = os.environ['SLURM_JOB_ID']
    submission = json.loads((ROOT / 'results/submission.json').read_text())
    runs = submission['training_runs']
    if set(runs) != set(ARMS) or submission.get('complete') is not True:
        raise ValueError('Missing complete four-method submission manifest')
    manifest = {'version': 1, 'arms': {}}
    for arm in ['base', *ARMS]:
        arguments = ['eval', '--arm', arm]
        if arm == 'base':
            output = ROOT / 'evaluations' / f'base-{job}'
            manifest['baseline'] = str(output)
        else:
            run = Path(runs[arm])
            if run.parent != ROOT / 'checkpoints' or not run.name.startswith(arm + '-train-'):
                raise ValueError('Training run is outside the submitted pilot')
            arguments += ['--run-dir', str(run), '--epoch', '4']
            output = ROOT / 'evaluations' / f'{arm}-epoch4-{job}'
            manifest['arms'][arm] = {'4': str(output)}
        command = ['srun', '--exact', '--ntasks=1', '--cpus-per-task=8', '--gres=gpu:1', '--mem=64G',
                   '/bin/bash', str(CODE / 'camd/enter_runtime.sh'), 'gpu', str(CODE / 'camd/stage.py'), *arguments]
        subprocess.run(command, check=True)
        # This process is host Python without Torch. Require real completion artifacts.
        subprocess.run([sys.executable, '-I', '-S', str(CODE / 'camd/host_gate.py'), job, 'eval', arm], check=True)
    path = ROOT / 'results/evaluation_manifest.json'
    with path.open('x') as stream:
        json.dump(manifest, stream, indent=2)
        stream.write('\n')
    subprocess.run([sys.executable, '-I', '-S', str(CODE / 'camd/compare_evaluations.py'),
                    '--manifest', str(path), '--output', str(ROOT / 'results/comparison')], check=True)
    print('ALL_FOUR_METHODS_COMPARED', flush=True)


if __name__ == '__main__':
    main()
