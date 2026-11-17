"""Retry publication of a completed run without starting training or allocating GPUs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

from env_config import load_env_file
from hf_config import resolve_hf_config
from run_settings import load_settings
from run import freeze_source, hf_environment, hf_upload_command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--settings', help='Defaults to root run_settings.toml')
    parser.add_argument('--env-file', help='Defaults to root .env')
    parser.add_argument('--hf-epochs')
    parser.add_argument('--hf-repo-prefix')
    parser.add_argument('--hf-description')
    parser.add_argument('--hf-retries')
    parser.add_argument('--dry-run', action='store_true', help='Show configuration only; no authentication or upload')
    args = parser.parse_args()
    try:
        env = os.environ.copy()
        load_env_file(args.env_file, environ=env)
        settings = load_settings(args.settings)
        hf = resolve_hf_config('train', settings, {**vars(args), 'hf_upload': 'on'},
                               environ=env, require_credentials=not args.dry_run)
        if any(env.get(key) and env[key] in args.run_dir for key in ('HF_TOKEN', 'WANDB_API_KEY')):
            raise ValueError('Credentials must not appear in a run-directory argument')
        run_dir = Path(args.run_dir).resolve(strict=True)
        if not run_dir.is_dir() or not (run_dir / 'done.json').is_file():
            raise ValueError('Pass a completed training run containing done.json')
        if args.dry_run:
            print(json.dumps(dict(run_dir=str(run_dir), configuration=hf,
                                  configuration_only=True, network_calls=0), indent=2))
            return 0
        command = hf_upload_command(hf, freeze_source() / 'hf_publish.py', run_dir=run_dir)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    return subprocess.call(command, env=hf_environment(env))


if __name__ == '__main__':
    raise SystemExit(main())
