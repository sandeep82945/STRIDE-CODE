"""Portable launcher; preserve scheduler GPU allocation and use this Python env."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from env_config import load_env_file, runtime_arguments
from run_settings import load_settings, settings_arguments
from hf_config import resolve_hf_config

ROOT = Path(__file__).resolve().parent
STAGES = {"smoke": "trainer_distributed.py",
          "train": "trainer_resumable.py", "eval": "evaluate_async.py"}


def option(arguments, key):
    values = []
    for i, item in enumerate(arguments):
        if item == key:
            if i + 1 == len(arguments) or arguments[i + 1].startswith("--"):
                raise ValueError(f"Missing value for {key}")
            values.append(arguments[i + 1])
        elif item.startswith(key + "="):
            values.append(item.split("=", 1)[1])
    if len(values) > 1:
        raise ValueError(f"Duplicate option {key}")
    return values[0] if values else None


def build_command(stage, arguments, python=sys.executable, training_root=None):
    args = list(arguments)
    seed = experiment_seed(stage, args)
    if option(args, "--seed") is None:
        args.extend(["--seed", str(seed)])
    data = option(args, "--data")
    if not data:
        raise ValueError("Pass --data with a prepared JSON split")
    data = Path(data).resolve(strict=True)
    actual = hashlib.sha256(data.read_bytes()).hexdigest()
    supplied = option(args, "--data-sha256")
    if supplied and supplied != actual:
        raise ValueError("--data-sha256 does not match this dataset")
    if supplied is None:
        args.extend(["--data-sha256", actual])
    if not option(args, "--model-path"):
        raise ValueError("Pass --model-path with a downloaded local model")
    target = str((Path(training_root) if training_root else ROOT / "training") / STAGES[stage])
    if stage == "eval":
        return [python, "-u", target, *args]
    return [python, "-u", "-m", "torch.distributed.run", "--standalone",
            "--nnodes=1", "--nproc_per_node=4", target, stage, *args]


def freeze_source(source=None, cache=None):
    """Launch from a content-addressed snapshot so later source edits cannot mix into a job."""
    source = Path(source) if source else ROOT / "training"
    cache = Path(cache) if cache else ROOT / ".cache" / "source"
    if source.is_symlink() or not source.is_dir():
        raise ValueError("Training source must be a regular directory")
    entries = sorted(source.rglob('*'))
    if any(p.is_symlink() and (p.is_dir() or p.suffix == '.py') for p in entries):
        raise ValueError("Training source must not contain symlinked Python files or directories")
    payloads = {p.relative_to(source).as_posix(): p.read_bytes()
                for p in entries if p.is_file() and p.suffix == '.py'}
    if not payloads:
        raise ValueError("Training source contains no Python files")
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()}
    identifier = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / identifier
    if not destination.exists():
        temporary = Path(tempfile.mkdtemp(prefix=".writing-", dir=cache))
        try:
            for name, data in payloads.items():
                path = temporary / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            (temporary / "source_manifest.json").write_text(json.dumps(hashes, indent=2) + "\n")
            try:
                temporary.rename(destination)
            except OSError:
                if not destination.is_dir():
                    raise
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    if destination.is_symlink():
        raise ValueError("Source snapshot must be a regular directory")
    manifest = destination / 'source_manifest.json'
    if (manifest.is_symlink() or not manifest.is_file()
            or json.loads(manifest.read_text()) != hashes):
        raise ValueError("Source snapshot manifest was changed; use a clean cache directory")
    for name, expected in hashes.items():
        path = destination / name
        if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Source snapshot was changed; use a clean cache directory")
    actual_names = {p.relative_to(destination).as_posix() for p in destination.rglob("*.py")}
    if actual_names != set(hashes):
        raise ValueError("Unexpected Python files in source snapshot")
    return destination


def experiment_seed(stage, arguments):
    value = option(arguments, "--seed")
    seed = 42 if value is None else int(value)
    maximum = 2**63 - 1 if stage == "eval" else 2**32 - 1
    if not 0 <= seed <= maximum:
        raise ValueError(f"--seed must be in [0, {maximum}]")
    return seed


def execution_environment(stage, seed=42, *, environ=None):
    env = dict(os.environ if environ is None else environ)
    # Set before starting Python workers. Evaluation sampling supports 63-bit
    # seeds; Python hashing and engine initialization use the low 32 bits.
    env["PYTHONHASHSEED"] = str(seed % 2**32)
    expected = 1 if stage == "eval" else 8
    runtime = env.get("STRIDE_RUNTIME", "nvidia")
    if runtime not in ("nvidia", "rocm"):
        raise ValueError("STRIDE_RUNTIME must be nvidia or rocm")
    if runtime == "rocm":
        # ROCR first restricts physical devices and HIP indexes that restricted
        # list. Keep the scheduler's ROCR mask intact, including for 1-GPU eval.
        physical = env.get("ROCR_VISIBLE_DEVICES", "")
        allocated = [x.strip() for x in physical.split(",")]
        if (len(allocated) != expected or len(set(allocated)) != expected
                or any(not x for x in allocated)):
            raise ValueError(f"ROCm requires exactly {expected} scheduler ROCR_VISIBLE_DEVICES")
        logical = ",".join(str(i) for i in range(expected))
        for name in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
            original = env.get(name)
            if original is not None and original not in (physical, logical):
                raise ValueError(f"Conflicting ROCm allocation mask {name}={original!r}")
            if original is not None:
                env.setdefault("STRIDE_SCHEDULER_" + name, original)
            env[name] = logical
    visible = env.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        # Bare allocated machine; never run this in a Slurm login shell.
        if env.get("SLURM_JOB_ID"):
            raise ValueError("Slurm did not provide CUDA_VISIBLE_DEVICES; fix the GPU allocation")
        import torch
        if torch.version.hip:
            raise ValueError("This handoff targets NVIDIA CUDA; ROCm has not been validated")
        count = torch.cuda.device_count()
        if count != expected:
            raise ValueError(f"Need exactly {expected} allocated GPUs, found {count}; set visibility explicitly")
        visible = ",".join(str(i) for i in range(count))
    ids = [x.strip() for x in visible.split(",")]
    if len(ids) != expected or len(set(ids)) != expected or any(not x for x in ids):
        raise ValueError(f"Expected {expected} distinct scheduler-visible GPUs; got {visible!r}")
    env["CUDA_VISIBLE_DEVICES"] = visible
    for name, value in {"OMP_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false",
                        "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1",
                        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                        "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "VLLM_NO_USAGE_STATS": "1",
                        "WANDB_MODE": "disabled" if stage == "smoke" else "online"}.items():
        env.setdefault(name, value)
    env["WANDB_DISABLE_CODE"] = "true"
    # Hub publication is a separate CPU process; learners need no upload token.
    env.pop("HF_TOKEN", None)
    return env


def hf_environment(environ):
    """Enable Hub networking only for the explicit CPU publisher process."""
    env = dict(environ)
    env.pop('WANDB_API_KEY', None)
    for name in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE',
                 'MASTER_ADDR', 'MASTER_PORT'):
        env.pop(name, None)
    env.update(HF_HUB_OFFLINE='0', TRANSFORMERS_OFFLINE='1',
               HF_ENDPOINT='https://huggingface.co', HF_HUB_DISABLE_TELEMETRY='1',
               CUDA_VISIBLE_DEVICES='', WANDB_MODE='disabled',
               PYTHONDONTWRITEBYTECODE='1')
    return env


def hf_upload_command(config, worker, *, run_dir=None, preflight=False):
    command = [sys.executable, '-u', str(worker), '--namespace', config['namespace']]
    if preflight:
        return [*command, '--preflight']
    if run_dir is None:
        raise ValueError('HF upload requires a completed run directory')
    return [*command, '--run-dir', str(Path(run_dir).resolve()),
            '--epochs', ','.join(map(str, config['epochs'])),
            '--repo-prefix', config['repo_prefix'], '--description=' + config['description'],
            '--retries', str(config['retries'])]


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        "Remaining options are passed to the selected training/evaluation driver. "
        "Use python training/<driver>.py --help for its full options."))
    parser.add_argument("stage", choices=STAGES)
    settings_options = parser.add_mutually_exclusive_group()
    settings_options.add_argument("--settings", help="Run settings TOML; defaults to run_settings.toml beside run.py")
    settings_options.add_argument("--no-settings", action="store_true", help="Ignore TOML defaults and supply driver options explicitly")
    parser.add_argument("--env-file", help="Literal configuration file; defaults to optional .env beside run.py")
    parser.add_argument('--hf-upload', choices=('on', 'off'), help='Automatic upload after successful training')
    parser.add_argument('--hf-epochs', help='Comma-separated completed epochs to upload, e.g. 2,3,4')
    parser.add_argument('--hf-repo-prefix')
    parser.add_argument('--hf-description')
    parser.add_argument('--hf-retries')
    parser.add_argument("--dry-run", action="store_true", help="Print exact command, without allocating/loading GPUs")
    args, forwarded = parser.parse_known_args()
    try:
        configured = os.environ.copy()
        load_env_file(args.env_file, environ=configured)
        settings = None if args.no_settings else load_settings(args.settings)
        if not args.no_settings:
            forwarded = settings_arguments(args.stage, forwarded, settings, environ=configured)
        forwarded = runtime_arguments(args.stage, forwarded, environ=configured,
                                      require_key=not args.dry_run)
        hf = resolve_hf_config(args.stage, settings, vars(args), environ=configured,
                               require_credentials=not args.dry_run)
        if hf['enabled']:
            epochs = int(option(forwarded, '--epochs') or '4')
            if not any(epoch <= epochs for epoch in hf['epochs']):
                raise ValueError('No selected Hugging Face epoch is reached by this training schedule; adjust --hf-epochs or use --hf-upload off')
        command = build_command(args.stage, forwarded)
        if args.dry_run:
            print(json.dumps({"command": command, "required_gpus": 1 if args.stage == "eval" else 8,
                              "PYTHONHASHSEED": str(experiment_seed(args.stage, forwarded) % 2**32),
                              'huggingface': dict(enabled=hf['enabled'], epochs=hf['epochs'],
                                                 namespace=hf['namespace'], repo_prefix=hf['repo_prefix'],
                                                 timing='after successful training',
                                                 stride_private=True, grpo_private=hf.get('private_all', False)),
                              "dry_run": True}, indent=2))
            return 0
        env = execution_environment(args.stage, experiment_seed(args.stage, forwarded),
                                    environ=configured)
        frozen = freeze_source()
        command = build_command(args.stage, forwarded, training_root=frozen)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if hf['enabled']:
        preflight = hf_upload_command(hf, frozen / 'hf_publish.py', preflight=True)
        code = subprocess.call(preflight, env=hf_environment(configured))
        if code:
            print('Hugging Face authentication check failed; training was not started.', file=sys.stderr)
            return code
    code = subprocess.call(command, env=env)
    if code or not hf['enabled']:
        return code
    # torchrun has exited and all learner/sampler processes have been released.
    # Local training completion remains recorded even if publication fails.
    upload = hf_upload_command(hf, frozen / 'hf_publish.py', run_dir=option(forwarded, '--out'))
    code = subprocess.call(upload, env=hf_environment(configured))
    if code:
        print('Training completed; Hugging Face upload did not finish. Use upload_checkpoints.py to retry without retraining.', file=sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
