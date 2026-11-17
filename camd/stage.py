#!/usr/bin/env python3
"""Execute one pilot stage; explicit artifacts determine success, not exit alone."""
import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
import time

CODE = Path(__file__).resolve().parents[1]
ROOT = Path('/path/to/cluster-user/system/stride_sample100_20260915')
MODEL = Path('/path/to/cluster-user/system/stride/models/qwen4b')
ARMS = ['local_positive', 'single_step', 'token_uniform', 'grpo']
PROJECT = 'stride-sample100-20260915'
GROUP = 'qwen4b-lora-four-methods-seed42'
sys.path.insert(0, str(CODE))
import run as launcher
from env_config import load_env_file
from verify_outputs import verify


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def invoke(command, env):
    print('COMMAND', json.dumps([str(x) for x in command]), flush=True)
    subprocess.run([str(x) for x in command], env=env, cwd=CODE, check=True)


def cpu_evidence():
    evidence = read(ROOT / 'results/cpu_verified.json')
    assert evidence['passed'] and evidence['source_stable']
    assert not evidence['failures'] and not evidence['errors']
    assert evidence['tests_run'] == 449 and not evidence['skipped']
    inventory = list(CODE.glob('*.py'))
    for directory in ('training', 'tests', 'scripts'):
        inventory.extend((CODE / directory).rglob('*.py'))
    assert {str(p.relative_to(CODE)) for p in inventory} == set(evidence['source_sha256'])
    for name, expected in evidence['source_sha256'].items():
        assert hashlib.sha256((CODE / name).read_bytes()).hexdigest() == expected, name
    return {'tests_run': evidence['tests_run'], 'versions': evidence['versions']}


def dependencies(report_dir):
    result = subprocess.run([sys.executable, '-m', 'pip', 'check'], text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (report_dir / 'pip_check.txt').write_text(result.stdout)
    known = ['pygobject 3.42.1 requires pycairo, which is not installed.']
    assert result.returncode == 0 or (
        result.returncode == 1 and result.stdout.strip().splitlines() == known), 'Unexpected package conflict'


def tracking(stage):
    env_file = ROOT / 'config/wandb.env'
    # Always specify a file; never fall back to another account's ~/.netrc.
    values = {}
    load_env_file(env_file, environ=values)
    online = bool(values.get('WANDB_API_KEY', '').strip())
    mode = 'disabled' if stage == 'smoke' else 'online' if online else 'offline'
    return env_file, mode, values.get('WANDB_ENTITY') or 'YOUR_ACCOUNT'


def preflight(kind, env, report_dir):
    result = subprocess.run([sys.executable, str(CODE / 'scripts/check_runtime.py'),
                             '--runtime', 'rocm', '--gpus', '1' if kind == 'eval' else '8'],
                            env=env, cwd=CODE, text=True, stdout=subprocess.PIPE)
    (report_dir / 'runtime_preflight.json').write_text(result.stdout)
    actual = json.loads(result.stdout)
    assert actual['passed'] is True and actual['checks']['gpu_primitives'] is True
    if kind == 'smoke':
        path = report_dir / 'credit_device_parity.json'
        invoke([sys.executable, CODE / 'camd/credit_device_parity.py', 'check',
                '--reference', CODE / 'camd/credit_cpu_reference.json', '--output', path,
                '--device', 'cuda:0'], env)
        assert read(path)['passed'] is True, 'GPU credit parity failed'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['smoke', 'train', 'eval'])
    p.add_argument('--arm', choices=ARMS + ['base'], required=True)
    p.add_argument('--cap', type=int, choices=[128, 8192], default=8192)
    p.add_argument('--rollouts', type=int, choices=[2, 8], default=8)
    p.add_argument('--eager', action='store_true')
    p.add_argument('--smoke-output', type=Path)
    p.add_argument('--run-dir', type=Path)
    p.add_argument('--epoch', type=int, choices=[1, 2, 3, 4])
    cli = p.parse_args()
    if cli.arm == 'base' and cli.stage != 'eval':
        p.error('Only evaluation uses the base model arm')
    job = os.environ['SLURM_JOB_ID']
    report_dir = ROOT / 'results' / f'job-{job}' / f'{cli.stage}-{cli.arm}'
    report_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {'stage': cli.stage, 'arm': cli.arm, 'job_id': job, 'status': 'STARTED',
              'node': os.environ.get('SLURMD_NODENAME'),
              'versions': {name: version(name) for name in ['torch', 'vllm', 'transformers', 'peft', 'numpy']}}
    write(report_dir / 'run.json', report)
    try:
        report['cpu_validation'] = cpu_evidence()
        dependencies(report_dir)
        env_file, mode, entity = tracking(cli.stage)
        report['wandb'] = {'mode': mode, 'project': PROJECT, 'entity': entity}
        env = launcher.execution_environment('eval' if cli.stage == 'eval' else 'train')
        preflight(cli.stage, env, report_dir)
        common = [sys.executable, CODE / 'run.py', cli.stage, '--no-settings',
                  '--env-file', env_file, '--hf-upload', 'off',
                  '--model', 'qwen4b', '--model-path', MODEL, '--seed', '42',
                  '--wandb-project', PROJECT, '--wandb-group', GROUP,
                  '--wandb-entity', entity, '--wandb-mode', mode,
                  '--sampler-max-num-seqs', '32', '--generation-timeout', '3600']
        if cli.stage == 'eval':
            if cli.arm == 'base':
                output = ROOT / 'evaluations' / f'base-{job}'
            else:
                if cli.run_dir is None or cli.epoch is None:
                    raise ValueError('Checkpoint evaluation requires run directory and epoch')
                training = verify('train', cli.run_dir)
                assert training['passed'] is True and training['arm'] == cli.arm
                output = ROOT / 'evaluations' / f'{cli.arm}-epoch{cli.epoch}-{job}'
                common += ['--adapter', cli.run_dir / f'adapter_{cli.epoch * 2:03d}',
                           '--training-step', str(cli.epoch * 2), '--epoch', str(cli.epoch)]
            command = common + ['--data', ROOT / 'data/val.json',
                      '--train-data', ROOT / 'data/train.json', '--out', output,
                      '--rollouts', '8', '--batch-size', '16',
                      '--wandb-name', f'{cli.arm}-eval-epoch{cli.epoch or 0}-{job}']
        else:
            output = ROOT / 'checkpoints' / f'{cli.arm}-{cli.stage}-{job}'
            if cli.stage == 'train':
                if cli.smoke_output is None:
                    raise ValueError('Training requires its verified integrated smoke output')
                assert verify('smoke', cli.smoke_output)['passed'] is True
                smoke = read(cli.smoke_output / 'started.json')['args']
                assert smoke['arm'] == cli.arm and smoke['smoke_cap'] == 128
                assert smoke['rollouts'] == 2 and smoke['sampler_eager'] is True
            command = common + ['--data', ROOT / 'data/train.json', '--out', output,
                      '--arm', cli.arm, '--finetune-mode', 'lora', '--epochs', '4',
                      '--batch-size', '64', '--rollouts', str(cli.rollouts if cli.stage == 'smoke' else 8),
                      '--alpha', '1', '--learning-rate', '0.0001', '--validation-mode', 'off']
            if cli.stage == 'smoke':
                command += ['--smoke-cap', str(cli.cap)]
            if cli.eager:
                command += ['--sampler-eager']
        report['output'] = str(output)
        report['verification_kind'] = cli.stage
        report['command'] = [str(x) for x in command]
        write(report_dir / 'run.json', report)
        invoke(command, env)
        verification = verify(cli.stage, output)
        assert verification['passed'] is True
        write(report_dir / 'verification.json', verification)
        report['status'] = 'VERIFIED'
    except BaseException as exc:
        report['status'] = 'FAILED'
        report['error_type'] = type(exc).__name__
        # Do not mirror API exception bodies that might include credentials.
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - started
        write(report_dir / 'run.json', report)
        print('PILOT_STAGE_RESULT', json.dumps({k: report[k] for k in ['stage', 'arm', 'job_id', 'status', 'elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
