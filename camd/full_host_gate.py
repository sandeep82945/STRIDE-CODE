#!/usr/bin/env python3
"""Validate completed artifacts with host stdlib; native exit status is insufficient."""
import json
from pathlib import Path
import sys

root = Path('/path/to/cluster-user/system/stride_allref_correctonly_20260925')
job, kind, arm = sys.argv[1:]
report_dir = root / 'results' / f'job-{job}' / f'{kind}-{arm}'
report = json.loads((report_dir / 'run.json').read_text())
if report['stage'] != kind or report['arm'] != arm or report['job_id'] != job:
    raise SystemExit('Stage, method or job identity mismatch')
if report['status'] != 'FINISHED':
    raise SystemExit(f"Stage did not finish: {report['status']}; see its run.json")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_full import verify
verification = verify(kind, Path(report['output']))
if verification.get('passed') is not True:
    raise SystemExit('Host artifact validation did not pass')
(report_dir / 'host_verification.json').write_text(json.dumps(verification, indent=2, allow_nan=False) + '\n')
report['status'] = 'VERIFIED'
(report_dir / 'run.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
print('HOST_ARTIFACT_GATE_PASSED', job, kind, arm, flush=True)
if kind == 'train' and report.get('publisher_needs_retry'):
    raise SystemExit('Training verified; Hugging Face publishing needs retry from retained checkpoints')
