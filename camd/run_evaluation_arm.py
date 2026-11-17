#!/usr/bin/env python3
"""Evaluate exactly one pilot arm and require its independent host artifact gate."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path('/path/to/cluster-user/system/stride_sample100_20260915')
ARMS = ('local_positive', 'single_step', 'token_uniform', 'grpo')


def evaluation_command(arm, submission, root=ROOT):
    if arm not in ('base', *ARMS):
        raise ValueError('Unknown evaluation method')
    runs = submission.get('training_runs', {})
    if not isinstance(runs, dict) or set(runs) != set(ARMS) or submission.get('complete') is not True:
        raise ValueError('Missing complete four-method submission manifest')
    arguments = ['eval', '--arm', arm]
    if arm != 'base':
        run = Path(runs[arm])
        checkpoints = root / 'checkpoints'
        if (run.parent != checkpoints or not run.name.startswith(arm + '-train-')
                or not run.name.removeprefix(arm + '-train-').isascii()
                or not run.name.removeprefix(arm + '-train-').isdigit()
                or run.is_symlink() or run.resolve().parent != checkpoints.resolve()):
            raise ValueError('Training run is outside the submitted pilot or has an invalid identity')
        arguments += ['--run-dir', str(run), '--epoch', '4']
    code = root / 'code/stride_handoff'
    return ['srun', '--exact', '--ntasks=1', '--cpus-per-task=8', '--gres=gpu:1', '--mem=64G',
            '/bin/bash', str(code / 'camd/enter_runtime.sh'), 'gpu',
            str(code / 'camd/stage.py'), *arguments]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('arm', choices=('base', *ARMS))
    cli = parser.parse_args(argv)
    job = os.environ['SLURM_JOB_ID']
    if not job.isascii() or not job.isdigit():
        raise ValueError('Expected a numeric Slurm job identity')
    submission = json.loads((ROOT / 'results/submission.json').read_text())
    subprocess.run(evaluation_command(cli.arm, submission, ROOT), check=True)
    subprocess.run([sys.executable, '-I', '-S', str(ROOT / 'code/stride_handoff/camd/host_gate.py'),
                    job, 'eval', cli.arm], check=True)
    output_name = f'base-{job}' if cli.arm == 'base' else f'{cli.arm}-epoch4-{job}'
    print('EVALUATION_ARM_VERIFIED', json.dumps({
        'job_id': job, 'arm': cli.arm, 'output': str(ROOT / 'evaluations' / output_name),
    }), flush=True)


if __name__ == '__main__':
    main()
