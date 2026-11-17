#!/usr/bin/env python3
"""Submit the four LoRA arms and their fixed validation comparisons to Slurm."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path('/path/to/cluster-user/system/stride_sample100_20260915')
CODE = ROOT / 'code/stride_handoff'
ARMS = ['local_positive', 'single_step', 'token_uniform', 'grpo']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--submit', action='store_true', help='Actually submit; otherwise print the plan')
    cli = parser.parse_args()
    evidence = json.loads((ROOT / 'results/cpu_verified.json').read_text())
    if not evidence['passed'] or not evidence['source_stable']:
        raise ValueError('CPU suite has not passed')
    for name, expected in evidence['source_sha256'].items():
        if hashlib.sha256((CODE / name).read_bytes()).hexdigest() != expected:
            raise ValueError('CPU-tested source changed: ' + name)
    record_path = ROOT / 'results/submission.json'
    if cli.submit and record_path.exists():
        raise FileExistsError('Submission exists; inspect it before recovering partial submission')
    record = {'version': 1, 'lora_only': True, 'time_limit': '5-00:00:00', 'jobs': [], 'complete': False}

    def save():
        if cli.submit:
            temporary = record_path.with_suffix('.tmp')
            temporary.write_text(json.dumps(record, indent=2) + '\n')
            temporary.replace(record_path)

    def submit(stage, arm, options=(), dependencies=()):
        small = stage in ('eval', 'upload', 'compare')
        command = ['sbatch', '--parsable', '--kill-on-invalid-dep=yes',
                   '--job-name=system-100-' + arm + '-' + stage]
        if small:
            command += ['--gres=gpu:1', '--cpus-per-task=8', '--mem=64G']
        if dependencies:
            command += ['--dependency=afterok:' + ':'.join(dependencies)]
        if stage == 'compare':
            command += [str(CODE / 'camd/compare.sbatch')]
        else:
            command += [str(CODE / 'camd/job.sbatch'), arm]
        if cli.submit:
            response = subprocess.check_output(command, text=True).strip()
            job = response.split(';', 1)[0]
            if not job.isdecimal():
                raise ValueError('Unexpected sbatch response')
        else:
            job = str(900000 + len(record['jobs']))
        item = dict(job_id=job, stage=stage, arm=arm, dependencies=list(dependencies), command=command)
        record['jobs'].append(item)
        save()
        print(json.dumps(item), flush=True)
        return job

    save()
    training_jobs = []
    for arm in ARMS:
        training = submit('train', arm)
        record.setdefault('training_runs', {})[arm] = str(ROOT / 'checkpoints' / f'{arm}-train-{training}')
        training_jobs.append(training)
    submit('compare', 'all', dependencies=training_jobs)
    record['complete'] = True
    save()
    print(json.dumps({'submitted': cli.submit, 'jobs': len(record['jobs']), 'complete': True}))


if __name__ == '__main__':
    main()
