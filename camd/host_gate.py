#!/usr/bin/env python3
"""Never release a dependent job on the container's exit status alone."""
import json
from pathlib import Path
import sys

root = Path('/path/to/cluster-user/system/stride_sample100_20260915')
job, kind, arm = sys.argv[1:]
report_dir = root / 'results' / f'job-{job}' / f'{kind}-{arm}'
report = json.loads((report_dir / 'run.json').read_text())
if report['stage'] != kind or report['arm'] != arm:
    raise SystemExit('Stage or method identity mismatch')
if report['status'] != 'VERIFIED':
    raise SystemExit(f"Stage did not verify: {report['status']}; see its run.json")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_outputs import verify
verification = verify(report['verification_kind'], Path(report['output']))
if verification.get('passed') is not True:
    raise SystemExit('Host artifact validation did not pass')
(report_dir / 'host_verification.json').write_text(
    json.dumps(verification, indent=2, allow_nan=False) + '\n')
print('HOST_ARTIFACT_GATE_PASSED', sys.argv[1], flush=True)
