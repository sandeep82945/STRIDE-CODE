#!/usr/bin/env python3
"""Validate embedded CPU credit-reference provenance without importing Torch.

This host gate checks readiness; numerical CPU/ROCm comparison still runs on
the allocated GPU with its original tolerances and exact source checks.
"""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path


CASES = {'mixed_correctness_and_mapping', 'all_correct'}
ARMS = {'grpo', 'local_positive', 'single_step', 'token_uniform'}
TENSORS = {'main'} | {f'{kind}/{arm}' for kind in ('auxiliary', 'advantages') for arm in ARMS}


def digest(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError('Missing or unsafe credit-reference dependency: ' + str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_credit_reference(code):
    code = Path(code).resolve()
    script = code / 'camd/credit_device_parity.py'
    digest(script)
    # Read the parity script's actual literal declarations without importing
    # scientific code, initializing an accelerator, or duplicating its list.
    constants = {}
    for node in ast.parse(script.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ('SOURCE_FILES', 'ATOL', 'RTOL'):
                if name in constants:
                    raise ValueError('Duplicate credit-parity declaration: ' + name)
                constants[name] = ast.literal_eval(node.value)
    names = constants.get('SOURCE_FILES')
    if (not isinstance(names, (list, tuple)) or not names or len(set(names)) != len(names)
            or any(not isinstance(name, str) or Path(name).is_absolute() or '..' in Path(name).parts for name in names)
            or 'camd/credit_device_parity.py' not in names):
        raise ValueError('Invalid credit-parity source inventory')
    expected = {name: digest(code / name) for name in names}
    path = code / 'camd/credit_cpu_reference.json'
    reference_sha256 = digest(path)
    def invalid_constant(value):
        raise ValueError('Nonfinite CPU credit-reference JSON: ' + value)
    reference = json.loads(path.read_text(), parse_constant=invalid_constant)
    if not isinstance(reference, dict) or reference.get('passed') is not True or reference.get('mode') != 'reference':
        raise ValueError('CPU credit reference has not passed reference generation')
    if reference.get('source_hashes') != expected:
        raise ValueError('Stale CPU credit-reference source hashes; regenerate the numerical reference before submission')
    if any(key not in constants or reference.get(key.lower()) != constants[key] for key in ('ATOL', 'RTOL')):
        raise ValueError('CPU credit-reference tolerances differ from the GPU parity gate')
    cases = reference.get('cases')
    if not isinstance(cases, dict) or set(cases) != CASES:
        raise ValueError('CPU credit reference must contain both complete parity fixtures')
    for name, case in cases.items():
        if not isinstance(case, dict) or case.get('passed') is not True or case.get('output_device') != 'cpu':
            raise ValueError('CPU credit-reference fixture did not pass on CPU: ' + name)
        tensors, mask, calls = case.get('tensors'), case.get('mapped_mask'), case.get('solver_calls')
        if (not isinstance(tensors, dict) or set(tensors) != TENSORS
                or not isinstance(mask, list) or not mask or not isinstance(mask[0], list) or not mask[0]):
            raise ValueError('Incomplete CPU credit-reference tensors or mask: ' + name)
        width = len(mask[0])
        if any(not isinstance(row, list) or len(row) != width or any(type(x) is not bool for x in row) for row in mask):
            raise ValueError('Invalid CPU credit-reference mask: ' + name)
        for matrix in tensors.values():
            if (not isinstance(matrix, list) or len(matrix) != len(mask)
                    or any(not isinstance(row, list) or len(row) != width
                           or any(type(x) not in (int, float) or not math.isfinite(x) for x in row) for row in matrix)):
                raise ValueError('Invalid CPU credit-reference tensor shape/value: ' + name)
        if (not isinstance(calls, list) or not calls or any(not isinstance(c, dict) for c in calls)
                or {c.get('operation') for c in calls} != {'cholesky', 'slogdet'}
                or any(c.get('device') != 'cpu' or c.get('dtype') != 'torch.float32' for c in calls)):
            raise ValueError('CPU credit-reference solver evidence is incomplete: ' + name)
    return dict(passed=True, reference=str(path), reference_sha256=reference_sha256,
                source_files_checked=len(expected), fixture_count=len(cases), numerical_check_performed=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code-dir', type=Path, default=Path(__file__).resolve().parents[1])
    print(json.dumps(verify_credit_reference(parser.parse_args().code_dir), sort_keys=True))
