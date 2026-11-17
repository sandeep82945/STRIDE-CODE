#!/usr/bin/env python3
"""Run the unchanged test suite and record its actual outcome independently of exit status."""
import argparse
import hashlib
from importlib.metadata import version, PackageNotFoundError
import json
from pathlib import Path
import sys
import time
import unittest

CODE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(CODE), str(CODE / 'training')]


def hashes():
    roots = ['training', 'tests', 'scripts']
    files = list(CODE.glob('*.py'))
    for root in roots:
        files.extend((CODE / root).rglob('*.py'))
    return {str(p.relative_to(CODE)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(files)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, required=True)
    cli = parser.parse_args()
    if cli.result.exists():
        raise FileExistsError('Use a fresh test-result path')
    before = hashes()
    started = time.monotonic()
    suite = unittest.defaultTestLoader.discover(str(CODE / 'tests'), top_level_dir=str(CODE))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    after = hashes()
    versions = {}
    for name in ['torch', 'transformers', 'peft', 'numpy', 'pyarrow', 'huggingface-hub', 'vllm']:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    report = {'passed': result.wasSuccessful() and before == after,
              'tests_run': result.testsRun, 'failures': [str(test) for test, _ in result.failures],
              'errors': [str(test) for test, _ in result.errors],
              'skipped': [{'test': str(test), 'reason': reason} for test, reason in result.skipped],
              'versions': versions, 'python': sys.version, 'source_stable': before == after,
              'source_sha256': before, 'elapsed_seconds': time.monotonic() - started}
    cli.result.parent.mkdir(parents=True, exist_ok=True)
    cli.result.write_text(json.dumps(report, indent=2) + '\n')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
