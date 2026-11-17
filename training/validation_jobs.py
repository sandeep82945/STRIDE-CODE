"""Queue immutable epoch checkpoints for evaluation on independent Slurm GPUs.

Only the rank-zero trainer calls this module. Submission never waits for the
evaluation job, and a scheduler failure is recorded without stopping training.
No training rank, CUDA allocation, or credential is written into job scripts.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time

from evaluate_async import check_overlap, load_data


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _write_once(path, payload):
    path = Path(path)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def prepare_validation(args, training_rows):
    """Validate all held-out rows before model allocation; return JSON metadata."""
    mode = getattr(args, 'validation_mode', 'off')
    if mode == 'off' or getattr(args, 'mode', 'train') != 'train':
        return {'mode': 'off'}
    if mode != 'slurm':
        raise ValueError('Validation mode must be slurm or off')
    if shutil.which('sbatch') is None:
        raise ValueError('Slurm validation requires sbatch on PATH; use --validation-mode off outside Slurm')
    if not getattr(args, 'validation_data', None):
        raise ValueError('Slurm validation requires --validation-data')
    every = getattr(args, 'validation_every_epochs', 1)
    rollouts = getattr(args, 'validation_rollouts', 8)
    if isinstance(every, bool) or not isinstance(every, int) or every < 1:
        raise ValueError('Validation epoch frequency must be positive')
    if isinstance(rollouts, bool) or not isinstance(rollouts, int) or not 1 <= rollouts <= 64:
        raise ValueError('Validation rollouts must be between 1 and 64')
    val_path, train_path = Path(args.validation_data).resolve(strict=True), Path(args.data).resolve(strict=True)
    val_raw, train_raw = val_path.read_bytes(), train_path.read_bytes()
    val_sha, train_sha = _sha(val_raw), _sha(train_raw)
    expected = getattr(args, 'validation_data_sha256', None)
    if expected is not None and expected.lower() != val_sha:
        raise ValueError('Validation dataset SHA256 mismatch')
    if args.data_sha256.lower() != train_sha:
        raise ValueError('Training dataset SHA256 mismatch before validation preparation')
    held_out = load_data(val_path, val_sha)
    stored_training = load_data(train_path, train_sha)
    if stored_training != training_rows:
        raise ValueError('Validation must use the same complete training rows as the trainer')
    overlap = check_overlap(held_out, training_rows)
    # Copy exact bytes: queued jobs do not depend on editable input datasets.
    inputs = Path(args.out).resolve() / 'validation_jobs' / 'input'
    inputs.mkdir(parents=True, exist_ok=False)
    for filename, payload in [('val.json', val_raw), ('train.json', train_raw)]:
        with (inputs / filename).open('xb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    metadata = dict(mode='slurm', every_epochs=every, rollouts=rollouts,
                    data=str(inputs / 'val.json'), data_sha256=val_sha,
                    training_data=str(inputs / 'train.json'), training_data_sha256=train_sha,
                    source_data=str(val_path), source_training_data=str(train_path),
                    questions=len(held_out), overlap=overlap, seed=args.seed,
                    context_limit=8192, gpu_count_per_job=1, waits_for_jobs=False)
    _write_once(inputs / 'manifest.json', metadata)
    return metadata


_DISTRIBUTED_ENV = {'RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR',
                    'MASTER_PORT', 'GROUP_RANK', 'ROLE_RANK', 'ROLE_WORLD_SIZE',
                    'GROUP_WORLD_SIZE', 'CUDA_VISIBLE_DEVICES', 'NVIDIA_VISIBLE_DEVICES'}


def submission_environment(environ=None):
    """Preserve W&B auth in memory, but let Slurm assign a fresh GPU allocation."""
    environ = os.environ if environ is None else environ
    return {key: value for key, value in environ.items()
            if key not in _DISTRIBUTED_ENV and not key.startswith(('TORCHELASTIC_', 'SLURM_', 'SBATCH_'))}


def _checkpoint_manifest(checkpoint, out, step, mode):
    prefix = 'model' if mode == 'full' else 'adapter' if mode == 'lora' else None
    if prefix is None:
        raise ValueError('Validation checkpoint mode must be full or lora')
    path = Path(checkpoint)
    if path.is_symlink() or path.resolve(strict=True) != out / f'{prefix}_{step:03d}' or not path.is_dir():
        raise ValueError('Validation needs the immutable checkpoint for this completed epoch in this attempt')
    manifest_path = path.with_name(path.name + '_manifest.json')
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError('Validation requires the completed checkpoint manifest')
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    required = {'config.json'} if mode == 'full' else {'adapter_config.json', 'adapter_model.safetensors'}
    if not isinstance(manifest, dict) or not required <= set(manifest):
        raise ValueError('Validation checkpoint manifest is incomplete')
    for name, value in manifest.items():
        if (not isinstance(name, str) or Path(name).name != name or name in ('.', '..')
                or not isinstance(value, str) or re.fullmatch('[a-f0-9]{64}', value) is None):
            raise ValueError('Invalid checkpoint manifest entry')
        if (path / name).is_symlink() or not (path / name).is_file():
            raise ValueError('Validation checkpoint manifest contains a missing or symlinked file')
    if set(manifest) != {item.name for item in path.iterdir()}:
        raise ValueError('Validation checkpoint inventory differs from its manifest')
    # Weight hashing is deliberately performed in the independent evaluator,
    # not repeatedly by the training process for a many-gigabyte full model.
    return str(path.resolve()), _sha(raw)


class ValidationScheduler:
    def __init__(self, args, metadata, *, source_root=None, python=None, submit=None):
        self.args, self.metadata = args, metadata
        self.out = Path(args.out).resolve()
        self.source_root = Path(source_root or Path(__file__).parent).resolve()
        self.python = str(python or sys.executable)
        self.submit = submit or subprocess.run
        self.records = []

    def summary(self):
        return dict(mode=self.metadata['mode'], submissions=list(self.records),
                    submitted=sum(r['status'] == 'submitted' for r in self.records),
                    submission_failures=sum(r['status'] == 'submission_failed' for r in self.records),
                    completion_checked=False, waits_for_jobs=False)

    def _eval_command(self, checkpoint, step, epoch, output):
        a, m = self.args, self.metadata
        cmd = [self.python, '-u', str(self.source_root / 'evaluate_async.py'),
               '--model', a.model, '--model-path', str(Path(a.model_path).resolve()),
               '--checkpoint' if a.finetune_mode == 'full' else '--adapter', checkpoint,
               '--data', m['data'], '--data-sha256', m['data_sha256'],
               '--train-data', m['training_data'], '--train-data-sha256', m['training_data_sha256'],
               '--out', str(output), '--seed', str(m['seed']), '--rollouts', str(m['rollouts']),
               '--sampler-max-num-seqs', '16', '--generation-timeout', str(a.generation_timeout),
               '--epoch', str(epoch), '--training-step', str(step),
               '--wandb-mode', a.wandb_mode, '--wandb-group', a.wandb_group,
               '--wandb-name', f'{a.model}-{a.finetune_mode}-{a.arm}-seed{a.seed}-validation-epoch{epoch}']
        for flag, value in [('--wandb-project', a.wandb_project), ('--wandb-entity', a.wandb_entity)]:
            if value:
                cmd.extend([flag, value])
        if a.sampler_eager:
            cmd.append('--sampler-eager')
        return cmd

    def submit_checkpoint(self, completed_step, checkpoint_path, per_epoch):
        """Called only after a complete pinned epoch checkpoint; never poll jobs."""
        if self.metadata['mode'] == 'off':
            return None
        if per_epoch < 1 or completed_step < 1 or completed_step % per_epoch:
            return None
        epoch = completed_step // per_epoch
        if epoch % self.metadata['every_epochs']:
            return None
        record = dict(epoch=epoch, training_step=completed_step, status='submission_failed',
                      submitted_at=time.time(), inference_gpu_count=1)
        job_dir = self.out / 'validation_jobs' / f'epoch_{epoch:03d}'
        created = False
        try:
            # Exclusive creation prevents an accidental duplicate submission.
            job_dir.mkdir(parents=True, exist_ok=False)
            created = True
            checkpoint, manifest_sha = _checkpoint_manifest(checkpoint_path, self.out, completed_step, self.args.finetune_mode)
            for key, digest_key in [('data', 'data_sha256'), ('training_data', 'training_data_sha256')]:
                if _sha(Path(self.metadata[key]).read_bytes()) != self.metadata[digest_key]:
                    raise ValueError('Pinned validation input changed after preparation')
            target = self.source_root / 'evaluate_async.py'
            if target.is_symlink() or not target.is_file():
                raise ValueError('Frozen evaluation source is missing or symlinked')
            output = self.out / 'validation' / f'epoch_{epoch:03d}'
            if output.exists():
                raise FileExistsError('Validation output must be fresh')
            cmd = self._eval_command(checkpoint, completed_step, epoch, output)
            # Retain Slurm's newly assigned CUDA_VISIBLE_DEVICES. Inherited
            # training allocation fields were removed from sbatch's environment.
            script = '#!/bin/bash\nset -euo pipefail\n'
            script += 'unset ' + ' '.join(sorted(_DISTRIBUTED_ENV - {'CUDA_VISIBLE_DEVICES', 'NVIDIA_VISIBLE_DEVICES'})) + '\n'
            script += f'export PYTHONHASHSEED={self.metadata["seed"] % 2**32}\n'
            script += 'export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1\n'
            script += 'export PYTHONDONTWRITEBYTECODE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn\n'
            script += 'export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 WANDB_DISABLE_CODE=true\n'
            script += 'exec ' + shlex.join(cmd) + '\n'
            script_path = job_dir / 'job.sh'
            with script_path.open('x', encoding='utf-8') as stream:
                stream.write(script)
            submit_cmd = ['sbatch', '--parsable', '--export=ALL', '--nodes=1', '--ntasks=1',
                          '--gres=gpu:1', '--cpus-per-task=4', '--mem=32G', '--time=24:00:00',
                          '--job-name', f'stride-val-e{epoch}', '--chdir', str(self.out),
                          '--output', str(job_dir / 'slurm-%j.out'), '--error', str(job_dir / 'slurm-%j.err')]
            for key in ('partition', 'account', 'qos'):
                value = getattr(self.args, 'validation_' + key, None)
                if value:
                    submit_cmd.extend(['--' + key, value])
            _write_once(job_dir / 'request.json', dict(command=cmd, sbatch_command=submit_cmd,
                        epoch=epoch, training_step=completed_step, checkpoint=checkpoint,
                        checkpoint_manifest_sha256=manifest_sha, source_root=str(self.source_root),
                        evaluation_source_sha256=_sha(target.read_bytes()), output=str(output)))
            result = self.submit(submit_cmd, input=script, text=True, capture_output=True,
                                 env=submission_environment(), timeout=30, check=False)
            if result.returncode != 0:
                # Scheduler output may echo environment or site diagnostics;
                # do not store raw output that could contain credentials.
                raise RuntimeError(f'sbatch rejected validation submission (exit {result.returncode}); inspect scheduler access/settings')
            match = re.fullmatch(r'([0-9]+)(?:;[A-Za-z0-9_.-]+)?', result.stdout.strip())
            if match is None:
                raise RuntimeError('sbatch returned an ambiguous job ID; inspect the scheduler before retrying')
            record.update(status='submitted', job_id=match.group(1), output=str(output),
                          checkpoint=checkpoint, manifest_sha256=manifest_sha)
            _write_once(job_dir / 'submitted.json', record)
        except Exception as exc:
            # Never include arbitrary subprocess output or environment values.
            record['error_type'] = type(exc).__name__
            record['error'] = str(exc) if isinstance(exc, (ValueError, RuntimeError, FileExistsError)) else 'Validation submission failed; inspect job files and scheduler availability'
            if created:
                try:
                    _write_once(job_dir / 'submission_failed.json', record)
                except OSError:
                    pass
            print(f'Validation epoch {epoch}: {record["status"]} ({record["error_type"]}); training continues.', flush=True)
        self.records.append(record)
        return record
