#!/usr/bin/env python3
"""Run an allocated full-training stage with recorded correctness gates."""
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
ROOT = Path('/path/to/cluster-user/system/stride_allref_correctonly_20260925')
MODEL = Path('/path/to/cluster-user/system/stride/models/qwen17b')
ARMS = ['local_positive', 'token_uniform']
MODEL_KEY = 'qwen17b'
ALPHA = 1.0
PROJECT = 'stride-qwen17b-allref-correctonly-20260925'
GROUP = 'qwen17b-allref-correctonly-lr2e5-warmup10-kl0-alpha1-seed42'
sys.path.insert(0, str(CODE))
import run as launcher
from env_config import load_env_file
from verify_full import verify
from verify_credit_reference import verify_credit_reference


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix('.writing')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def invoke(command, env):
    print('COMMAND', json.dumps([str(x) for x in command]), flush=True)
    subprocess.run([str(x) for x in command], env=env, cwd=CODE, check=True)


def cpu_evidence(path):
    evidence = read(path)
    assert evidence['passed'] and evidence['source_stable']
    assert not evidence['failures'] and not evidence['errors']
    assert evidence['checks_run'] >= 4 and not evidence['skipped']
    inventory = list(CODE.glob('*.py'))
    for directory in ('training', 'tests', 'scripts'):
        inventory.extend((CODE / directory).rglob('*.py'))
    assert {str(p.relative_to(CODE)) for p in inventory} == set(evidence['source_sha256'])
    for name, expected in evidence['source_sha256'].items():
        assert hashlib.sha256((CODE / name).read_bytes()).hexdigest() == expected, name
    return {'checks_run': evidence['checks_run'], 'checks': evidence['checks'], 'path': str(path)}


def dependencies(report_dir):
    result = subprocess.run([sys.executable, '-m', 'pip', 'check'], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (report_dir / 'pip_check.txt').write_text(result.stdout)
    known = ['pygobject 3.42.1 requires pycairo, which is not installed.']
    assert result.returncode == 0 or (result.returncode == 1 and result.stdout.strip().splitlines() == known), 'Unexpected package conflict'


def production_evidence():
    manifest = read(ROOT / 'results/production_manifest.json')
    for name, expected in manifest['files'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, 'Frozen production file changed: ' + name
    return {'files_checked': len(manifest['files']),
            'credit_reference': verify_credit_reference(CODE)}


def tracking(stage):
    env_file = ROOT / 'config/wandb.env'
    values = {}
    load_env_file(env_file, environ=values)
    online = bool(values.get('WANDB_API_KEY', '').strip())
    mode = 'disabled' if stage == 'smoke' else 'online' if online else 'offline'
    return env_file, mode, values.get('WANDB_ENTITY') or 'YOUR_ACCOUNT'


def preflight(env, report_dir):
    result = subprocess.run([sys.executable, str(CODE / 'scripts/check_runtime.py'), '--runtime', 'rocm', '--gpus', '8'], env=env, cwd=CODE, text=True, stdout=subprocess.PIPE)
    (report_dir / 'runtime_preflight.json').write_text(result.stdout)
    actual = json.loads(result.stdout)
    assert result.returncode == 0 and actual['passed'] is True and actual['checks']['gpu_primitives'] is True
    path = report_dir / 'credit_device_parity.json'
    invoke([sys.executable, CODE / 'camd/credit_device_parity.py', 'check', '--reference', CODE / 'camd/credit_cpu_reference.json', '--output', path, '--device', 'cuda:0'], env)
    assert read(path)['passed'] is True, 'GPU credit parity failed'
    path = report_dir / 'allref_credit_gate.json'
    invoke([sys.executable, CODE / 'camd/allref_credit_gate.py', '--output', path], env)
    assert read(path)['passed'] is True, 'All-reference credit gate failed'


def select_microbatch(env, report_dir):
    path = report_dir / 'microbatch_probe.json'
    result = subprocess.run([sys.executable, str(CODE / 'camd/probe_microbatch.py'), '--model-path', str(MODEL), '--output', str(path)], env=env, cwd=CODE)
    actual = read(path) if path.exists() else {'passed': False, 'complete': False}
    chosen = 2 if result.returncode == 0 and actual.get('passed') is True and actual.get('recommended_learner_microbatch_size') == 2 else 1
    write(report_dir / 'throughput_selection.json', dict(learner_microbatch_size=chosen, probe_returncode=result.returncode, probe_path=str(path), global_prompt_batch=64, rollouts_per_prompt=8, sampler_max_num_seqs=32, reason='Validated GPU parity, memory and measured speed gain' if chosen == 2 else 'Conservative fallback; paired GPU gate did not pass'))
    print('LEARNER_MICROBATCH_SELECTED', chosen, flush=True)
    return chosen


def start_publisher(arm, output, report_dir):
    """Explicit credential, CPU only, asynchronous to learner updates."""
    env = os.environ.copy()
    for key in ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'OPENAI_API_KEY', 'WANDB_API_KEY'):
        env.pop(key, None)
    env.update(HF_HUB_OFFLINE='0', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_IMPLICIT_TOKEN='1', ROCR_VISIBLE_DEVICES='-1', HIP_VISIBLE_DEVICES='-1', CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', HF_XET_HIGH_PERFORMANCE='0')
    repo = read(ROOT / 'results/hf_repositories.json')['repos'][arm]
    stop = report_dir / 'publisher_stop.json'
    state_dir = ROOT / 'results/hf' / f'{arm}-{os.environ["SLURM_JOB_ID"]}'
    command = [sys.executable, ROOT / 'hf/publish_checkpoints.py', 'watch', '--run-dir', output, '--repo-id', repo, '--arm', arm, '--model-path', MODEL, '--code-dir', CODE, '--token-file', ROOT / 'config/huggingface.env', '--stop-file', stop, '--state-dir', state_dir]
    log = (report_dir / 'publisher.log').open('w')
    worker = subprocess.Popen([str(x) for x in command], env=env, cwd=CODE, stdout=log, stderr=subprocess.STDOUT)
    log.close()
    # Confirm the CPU worker authenticated and verified its public repository
    # before spending days training. No policy output exists at this point.
    try:
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if worker.poll() is not None:
                raise RuntimeError('Checkpoint publisher exited during startup; see publisher.log')
            status_path = state_dir / 'status.json'
            if status_path.is_file():
                status = read(status_path)
                if status.get('repository') == repo and not status.get('errors') and not status.get('stop_requested'):
                    return worker, stop, repo
            time.sleep(1)
        raise RuntimeError('Checkpoint publisher did not become ready; see its status.json')
    except BaseException:
        worker.terminate()
        try:
            worker.wait(timeout=15)
        except subprocess.TimeoutExpired:
            worker.kill()
            worker.wait()
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['smoke', 'train'])
    p.add_argument('--arm', choices=ARMS, required=True)
    p.add_argument('--smoke-output', type=Path)
    cli = p.parse_args()
    job = os.environ['SLURM_JOB_ID']
    job_dir = ROOT / 'results' / f'job-{job}'
    report_dir = job_dir / f'{cli.stage}-{cli.arm}'
    report_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {'stage': cli.stage, 'arm': cli.arm, 'job_id': job, 'status': 'STARTED', 'node': os.environ.get('SLURMD_NODENAME'), 'versions': {name: version(name) for name in ['torch', 'vllm', 'transformers', 'peft', 'numpy']}}
    worker = stop = None
    write(report_dir / 'run.json', report)
    try:
        report['production_validation'] = production_evidence()
        report['source_validation'] = cpu_evidence(ROOT / 'results/preflight_verified.json')
        # This ROCm build lacks CPU LAPACK. Validate exact externally CPU-tested
        # source here, then exercise its credit math and LoRA update on real GPUs.
        report['cpu_validation'] = report['source_validation']
        dependencies(report_dir)
        env_file, mode, entity = tracking(cli.stage)
        report['wandb'] = {'mode': mode, 'project': PROJECT, 'entity': entity}
        env = launcher.execution_environment('train')
        if cli.stage == 'smoke':
            preflight(env, report_dir)
        output = ROOT / 'checkpoints' / f'{cli.arm}-{cli.stage}-{job}'
        command = [sys.executable, CODE / 'run.py', cli.stage, '--no-settings', '--env-file', env_file, '--hf-upload', 'off', '--model', MODEL_KEY, '--model-path', MODEL, '--seed', '42', '--wandb-project', PROJECT, '--wandb-group', GROUP, '--wandb-entity', entity, '--wandb-mode', mode, '--sampler-max-num-seqs', '32', '--generation-timeout', '3600', '--data', ROOT / 'data/train.json', '--out', output, '--arm', cli.arm, '--finetune-mode', 'lora', '--epochs', '4', '--batch-size', '64', '--rollouts', '2' if cli.stage == 'smoke' else '8', '--alpha', str(ALPHA), '--novelty-reference-scope', 'all_rollouts', '--learning-rate', '0.00002', '--warmup-steps', '10', '--kl-coefficient', '0', '--validation-mode', 'off']
        if cli.stage == 'smoke':
            command += ['--smoke-cap', '128', '--sampler-eager']
        else:
            if cli.smoke_output is None:
                raise ValueError('Training requires its verified integrated smoke output')
            smoke = verify('smoke', cli.smoke_output)
            assert smoke['passed'] is True and smoke['arm'] == cli.arm
            assert read(job_dir / f'smoke-{cli.arm}' / 'host_verification.json')['passed'] is True
            microbatch = select_microbatch(env, report_dir)
            command += ['--keep-all-checkpoints', '--resume-state-retention', 'latest-epoch', '--learner-microbatch-size', str(microbatch)]
            report['learner_microbatch_size'] = microbatch
            worker, stop, repo = start_publisher(cli.arm, output, report_dir)
            report['public_hf_repo'] = repo
        report.update(output=str(output), verification_kind=cli.stage, command=[str(x) for x in command])
        write(report_dir / 'run.json', report)
        invoke(command, env)
        # Host Python performs the complete artifact/hash gate once after this stage.
        assert read(output / 'done.json')['complete'] is True
        report['status'] = 'FINISHED'
    except BaseException as exc:
        report['status'] = 'FAILED'
        report['error_type'] = type(exc).__name__
        raise
    finally:
        if worker is not None:
            write(stop, {'training_status': report['status']})
            try:
                report['publisher_exit_code'] = worker.wait(timeout=300)
            except subprocess.TimeoutExpired:
                worker.terminate()
                try:
                    worker.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    worker.kill()
                    worker.wait()
                report['publisher_exit_code'] = worker.returncode
                report['publisher_needs_retry'] = True
            report['publisher_needs_retry'] = report.get('publisher_needs_retry', False) or report['publisher_exit_code'] != 0
        report['elapsed_seconds'] = time.monotonic() - started
        write(report_dir / 'run.json', report)
        print('FULL_STAGE_RESULT', json.dumps({k: report[k] for k in ['stage', 'arm', 'job_id', 'status', 'elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
