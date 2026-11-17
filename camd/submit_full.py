#!/usr/bin/env python3
"""Submit this exact stabilized LoRA family once, retaining durable partial receipts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_credit_reference import verify_credit_reference

ROOT = Path('/path/to/cluster-user/system/stride_allref_correctonly_20260925')
CODE = ROOT / 'code/stride_handoff'
ARMS = ['local_positive', 'token_uniform']
MODEL_KEY = 'qwen17b'
ALPHA = 1.0
QOS = 'gtqos'
TIME_LIMIT = '3-00:00:00'
JOB_PREFIX = 'system-s17-ar'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--submit', action='store_true')
    cli = parser.parse_args()
    evidence = json.loads((ROOT / 'results/preflight_verified.json').read_text())
    if not evidence['passed'] or not evidence['source_stable'] or evidence['skipped'] or evidence['checks_run'] < 4:
        raise ValueError('Experiment preflight has not passed completely')
    inventory = list(CODE.glob('*.py'))
    for name in ('training', 'tests', 'scripts'):
        inventory += list((CODE / name).rglob('*.py'))
    if {str(p.relative_to(CODE)) for p in inventory} != set(evidence['source_sha256']):
        raise ValueError('Tested source inventory changed')
    for name, expected in evidence['source_sha256'].items():
        if hashlib.sha256((CODE / name).read_bytes()).hexdigest() != expected:
            raise ValueError('CPU-tested source changed: ' + name)
    frozen = json.loads((ROOT / 'results/production_manifest.json').read_text())
    for name, expected in frozen['files'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen production file changed: ' + name)
    verify_credit_reference(CODE)
    repositories = json.loads((ROOT / 'results/hf_repositories.json').read_text())
    if set(repositories['repos']) != set(ARMS) or not repositories['verified_public']:
        raise ValueError('The new public repository must be prepared')
    record_path = ROOT / 'results/submission.json'
    if cli.submit and record_path.exists():
        raise FileExistsError('Submission receipt exists; inspect it before recovering partial submission')
    record = dict(version=2, model=MODEL_KEY, thinking_mode=False, qos=QOS, time_limit=TIME_LIMIT, lora_only=True, training_questions=2048,
        epochs=4, updates=128, global_prompt_batch=64, rollouts_per_prompt=8, gpus_per_job=8,
        learners_per_job=4, samplers_per_job=4, base_job=False, independent_parallel_jobs=True,
        learning_rate=2e-5, warmup_steps=10, kl_coefficient=0.0, alpha=ALPHA,
        novelty_reference_scope='all_rollouts', diversity_credit_scope='correct_only',
        initialization='original_base_fresh_lora',
        pending_submission=None, jobs=[], training_runs={}, public_repositories=repositories['repos'], complete=False)

    def save():
        if cli.submit:
            tmp = record_path.with_suffix('.writing')
            with tmp.open('x') as stream:
                stream.write(json.dumps(record, indent=2) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
            tmp.replace(record_path)

    # Exclusively reserve before sbatch; even a simultaneous launcher cannot duplicate jobs.
    if cli.submit:
        with record_path.open('x') as stream:
            stream.write(json.dumps(record, indent=2) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
    for arm in ARMS:
        short_arm = {'local_positive': 'lp', 'token_uniform': 'tu'}[arm]
        command = ['sbatch', '--parsable', f'--job-name={JOB_PREFIX}-{short_arm}', str(CODE / 'camd/full_job.sbatch'), arm]
        record['pending_submission'] = dict(arm=arm, command=command)
        save()
        job = subprocess.check_output(command, text=True).strip().split(';', 1)[0] if cli.submit else str(900000 + len(record['jobs']))
        if not job.isdecimal():
            raise ValueError('Unexpected sbatch response; inspect queue before retry')
        item = dict(job_id=job, stage='train', arm=arm, dependencies=[], command=command)
        record['jobs'].append(item)
        record['pending_submission'] = None
        record['training_runs'][arm] = str(ROOT / 'checkpoints' / f'{arm}-train-{job}')
        save()
        print(json.dumps(item), flush=True)
    record['complete'] = True
    save()
    print(json.dumps({'submitted': cli.submit, 'jobs': len(record['jobs']), 'complete': True}))


if __name__ == '__main__':
    main()
