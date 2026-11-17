"""Build and verify a source-only ZIP; never include runtime artifacts."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TOP = {'README.md', 'VERIFICATION.md', 'HF_UPLOAD.md', 'RUN_SETTINGS.md', 'run_settings.toml', 'run_settings.py', 'run.sbatch', 'run.py', 'hf_config.py', 'upload_checkpoints.py', 'env_config.py', '.env.example', '.gitignore', 'requirements-nvidia.txt',
       'requirements-nvidia-lock.txt', 'requirements-cpu.txt', 'THIRD_PARTY_NOTICES.md'}
SUB = {'training', 'scripts', 'tests', 'docs', 'licenses'}
EXT = {'.py', '.md', '.json', '.sbatch', '.txt'}


def source_files(root=None):
    root = Path(root) if root is not None else ROOT
    files = []
    for path in sorted(root.rglob('*')):
        rel = path.relative_to(root)
        if path.is_symlink() or not path.is_file():
            continue
        if rel.as_posix() in TOP:
            files.append(path)
        elif rel.parts[0] in SUB and path.suffix in EXT and not any(part.startswith('.') or part == '__pycache__' for part in rel.parts):
            files.append(path)
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    files = source_files()
    required = TOP | {'scripts/prepare_data.py', 'scripts/download_model.py', 'docs/VALIDATION.md'}
    present = {p.relative_to(ROOT).as_posix() for p in files}
    if not required <= present:
        raise ValueError(f'Missing release inputs: {sorted(required-present)}')
    # Read each input once so a concurrent repository edit cannot produce an
    # archive containing different bytes from those recorded in its manifest.
    payloads = {p.relative_to(ROOT).as_posix(): p.read_bytes() for p in files}
    hashes = {name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()}
    manifest = {'format': 1, 'source_files': hashes, 'contains_models_or_data': False,
                'gpu_training_of_this_portable_release_verified': False}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists() or args.out.is_symlink():
        raise FileExistsError(f'Release output already exists: {args.out}')
    fd, staging_name = tempfile.mkstemp(prefix='.stride-release-', suffix='.zip', dir=args.out.parent)
    os.close(fd)
    staging = Path(staging_name)
    try:
        with zipfile.ZipFile(staging, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for name, payload in payloads.items():
                archive.writestr('stride_handoff/' + name, payload)
            archive.writestr('stride_handoff/RELEASE_MANIFEST.json', json.dumps(manifest, indent=2) + '\n')
        with zipfile.ZipFile(staging) as archive:
            if archive.testzip() is not None:
                raise ValueError('Archive CRC validation failed')
            for name, expected in hashes.items():
                if hashlib.sha256(archive.read('stride_handoff/' + name)).hexdigest() != expected:
                    raise ValueError(f'Archive member hash mismatch: {name}')
        # A hard link publishes atomically and refuses an existing destination,
        # including a concurrent publisher or a dangling destination symlink.
        os.link(staging, args.out)
    finally:
        staging.unlink(missing_ok=True)
    print(json.dumps({'archive': str(args.out.resolve()), 'bytes': args.out.stat().st_size,
                      'source_files': len(hashes), 'sha256': hashlib.sha256(args.out.read_bytes()).hexdigest(),
                      'verified': True}, indent=2))


if __name__ == '__main__':
    main()
