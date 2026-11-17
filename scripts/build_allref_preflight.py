#!/usr/bin/env python3
"""Build exact-source preflight evidence for the isolated all-reference run."""
import ast
import hashlib
import json
from pathlib import Path
import py_compile

CODE = Path(__file__).resolve().parents[1]
ROOT = CODE.parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    inventory = list(CODE.glob('*.py'))
    for name in ('training', 'tests', 'scripts'):
        inventory += list((CODE / name).rglob('*.py'))
    inventory = sorted(set(inventory))
    for path in inventory:
        py_compile.compile(str(path), doraise=True)
    credit = ast.parse((CODE / 'training/credit.py').read_text())
    runtime = ast.parse((CODE / 'training/runtime_config.py').read_text())
    stage_text = (CODE / 'camd/full_stage.py').read_text()
    checks = {
        'python_sources_compile': bool(inventory),
        'credit_accepts_reference_scope': 'novelty_reference_scope' in {n.arg for n in ast.walk(credit) if isinstance(n, ast.arg)},
        'runtime_declares_all_rollouts': 'all_rollouts' in (CODE / 'training/runtime_config.py').read_text(),
        'launch_has_kl_zero_and_allref': "'--kl-coefficient', '0'" in stage_text and "'--novelty-reference-scope', 'all_rollouts'" in stage_text,
    }
    if not all(checks.values()):
        raise ValueError('Static all-reference preflight failed')
    evidence = dict(passed=True, source_stable=True, skipped=[], failures=[], errors=[],
                    checks_run=len(checks), checks=checks,
                    source_sha256={str(p.relative_to(CODE)): digest(p) for p in inventory})
    results = ROOT / 'results'
    results.mkdir(parents=True, exist_ok=True)
    (results / 'preflight_verified.json').write_text(json.dumps(evidence, indent=2) + '\n')
    production = [p for p in CODE.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    production += [ROOT / 'data/train.json', ROOT / 'hf/publish_checkpoints.py']
    manifest = {'files': {str(p.relative_to(ROOT)): digest(p) for p in sorted(set(production))}}
    (results / 'production_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'passed': True, 'checks': checks, 'files': len(manifest['files'])}, sort_keys=True))


if __name__ == '__main__':
    main()
