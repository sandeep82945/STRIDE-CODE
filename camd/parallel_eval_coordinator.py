#!/usr/bin/env python3
"""Hand off an active local-positive evaluation to held parallel evaluation jobs.

Run on the original batch node with host Python. Only the old host controller
is paused; its active srun and GPU sampler are allowed to finish normally.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path('/path/to/cluster-user/system/stride_sample100_20260915')
CODE = ROOT / 'code/stride_handoff'
ARMS = ('single_step', 'token_uniform', 'grpo')
TERMINAL = {'BOOT_FAIL', 'CANCELLED', 'COMPLETED', 'DEADLINE', 'FAILED',
            'NODE_FAIL', 'OUT_OF_MEMORY', 'PREEMPTED', 'REVOKED', 'TIMEOUT'}


def read(path):
    return json.loads(Path(path).read_text())


def atomic_write(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def validate_plan(plan):
    if plan.get('version') != 1 or str(plan.get('old_job')) != '189545':
        raise ValueError('Unexpected evaluation handoff plan')
    if plan.get('held') is not True or set(plan.get('jobs', {})) != set(ARMS):
        raise ValueError('All three new evaluation jobs must initially be held')
    jobs = {arm: str(plan['jobs'][arm]) for arm in ARMS}
    if any(not job.isdigit() or job == '189545' for job in jobs.values()):
        raise ValueError('Invalid parallel evaluation job ID')
    if len(set(jobs.values())) not in (1, 3):
        raise ValueError('Use one three-arm bundle or three distinct evaluation jobs')
    return jobs


def process(pid, proc_root=Path('/proc')):
    path = proc_root / str(pid)
    if path.stat().st_uid != os.getuid():
        raise PermissionError('Process belongs to another user')
    command = [part.decode() for part in (path / 'cmdline').read_bytes().split(b'\0') if part]
    values = (path / 'stat').read_text().rsplit(') ', 1)[1].split()
    environment = dict(item.split(b'=', 1) for item in (path / 'environ').read_bytes().split(b'\0')
                       if b'=' in item)
    return {'pid': int(pid), 'command': command, 'state': values[0],
            'ppid': int(values[1]), 'starttime': int(values[19]),
            'job': environment.get(b'SLURM_JOB_ID', b'').decode()}


def processes(proc_root=Path('/proc')):
    for path in proc_root.iterdir():
        if path.name.isdigit():
            try:
                yield process(path.name, proc_root)
            except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError, IndexError):
                continue


def select_controller(entries, old_job, controller_path):
    found = [entry for entry in entries if entry['job'] == str(old_job)
             and entry['command'][1:] == ['-I', '-S', str(controller_path)]
             and entry['command'] and Path(entry['command'][0]).name.startswith('python3')]
    if len(found) != 1:
        raise ValueError('Expected exactly one original host evaluation controller')
    return found[0]


def same_process(expected, actual):
    return all(actual[key] == expected[key] for key in ('pid', 'starttime', 'job', 'command'))


def has_local_child(entries, parent, stage_path):
    for entry in entries:
        command = entry['command']
        if entry['ppid'] != parent['pid'] or not command or Path(command[0]).name != 'srun':
            continue
        try:
            index = command.index(str(stage_path))
        except ValueError:
            continue
        if command[index + 1:index + 4] == ['eval', '--arm', 'local_positive']:
            return True
    return False


def report_path(job, arm):
    return ROOT / 'results' / f'job-{job}' / f'eval-{arm}' / 'run.json'


def report_state(job, arm):
    try:
        report = read(report_path(job, arm))
    except (FileNotFoundError, json.JSONDecodeError):
        # stage.py replaces this small report by a regular write. A reader can
        # briefly catch the empty/truncated file while the writer updates it.
        return None
    if (report.get('job_id') != str(job) or report.get('stage') != 'eval'
            or report.get('arm') != arm):
        raise ValueError('Evaluation report identity mismatch')
    return report.get('status')


def command(arguments):
    return subprocess.run([str(value) for value in arguments], check=True, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120).stdout


def job_state(job):
    output = command(['sacct', '-n', '-X', '-j', job, '--format=JobIDRaw,State', '--parsable2'])
    for line in output.splitlines():
        fields = line.split('|')
        if fields[0] == str(job):
            return fields[1].split()[0].rstrip('+')
    return None


def host_gate(job, arm):
    command([sys.executable, '-I', '-S', CODE / 'camd/host_gate.py', job, 'eval', arm])


def manifest(old_job, jobs):
    return {'version': 1, 'baseline': str(ROOT / 'evaluations' / f'base-{old_job}'),
            'arms': {'local_positive': {'4': str(ROOT / 'evaluations' / f'local_positive-epoch4-{old_job}')},
                     **{arm: {'4': str(ROOT / 'evaluations' / f'{arm}-epoch4-{jobs[arm]}')}
                        for arm in ARMS}}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--timeout-seconds', type=float, default=119 * 3600)
    parser.add_argument('--poll-seconds', type=float, default=30)
    args = parser.parse_args()
    if not 0 < args.timeout_seconds <= 119 * 3600 or not 1 <= args.poll_seconds <= 60:
        parser.error('Invalid bounded polling settings')
    plan = read(args.plan)
    jobs = validate_plan(plan)
    old_job = str(plan['old_job'])
    coordinator_job = os.environ['SLURM_JOB_ID']
    if not coordinator_job.isdigit() or coordinator_job in [old_job, *jobs.values()]:
        raise ValueError('Coordinator requires its own Slurm allocation')
    state_path = ROOT / 'results' / f'parallel_handoff_job-{coordinator_job}.json'
    if state_path.exists():
        raise ValueError('Coordinator report already exists; do not resume implicitly')
    state = {'version': 1, 'coordinator_job': coordinator_job, 'old_job': old_job,
             'jobs': jobs, 'status': 'STARTED', 'released': [], 'verified': [], 'failures': {}}
    deadline = time.monotonic() + args.timeout_seconds
    parent = None
    paused = False
    release_attempted = False
    terminal_seen = {}

    def save(status=None):
        if status:
            state['status'] = status
        state['updated_unix_seconds'] = time.time()
        atomic_write(state_path, state)

    def cancel_original(reason):
        command(['scancel', old_job])
        state['original_cancelled'] = True
        state['original_cancellation_reason'] = reason
        save()

    def terminal_without_artifacts(job, arm):
        # A completed GPU job can precede NFS visibility of its last report.
        # Require persistent terminal state before treating missing/unfinished
        # evidence as a failure. An explicit FAILED report needs no grace.
        key = (str(job), arm)
        if job_state(job) not in TERMINAL:
            terminal_seen.pop(key, None)
            return False
        first = terminal_seen.setdefault(key, time.monotonic())
        return time.monotonic() - first >= 90

    try:
        save()
        # Verify that releasing these jobs will not unexpectedly alter an
        # already-running allocation or another user's job.
        for job in sorted(set(jobs.values())):
            details = command(['scontrol', 'show', 'job', '-o', job])
            fields = dict(part.split('=', 1) for part in details.split() if '=' in part)
            if (fields.get('JobState') != 'PENDING' or fields.get('Reason') != 'JobHeldUser'
                    or fields.get('UserId', '').split('(')[-1].rstrip(')') != str(os.getuid())):
                raise ValueError(f'Evaluation job {job} is not held by this user')
        entries = list(processes())
        parent = select_controller(entries, old_job, CODE / 'camd/run_evaluations.py')
        if not has_local_child(entries, parent, CODE / 'camd/stage.py'):
            raise ValueError('Original controller does not have the expected local-positive srun child')
        if any(report_path(old_job, arm).parent.exists() for arm in ARMS):
            raise ValueError('A later original evaluation has already started')
        if parent['state'] in ('T', 't', 'Z') or not same_process(parent, process(parent['pid'])):
            raise ValueError('Original process is stopped, exited, or changed identity')
        os.kill(parent['pid'], signal.SIGSTOP)
        paused = True
        pause_deadline = time.monotonic() + 5
        while True:
            actual = process(parent['pid'])
            if not same_process(parent, actual):
                raise ValueError('Original process identity changed while pausing')
            if actual['state'] == 'T':
                break
            if time.monotonic() >= pause_deadline:
                raise TimeoutError('Original controller did not enter stopped state')
            time.sleep(0.05)
        if any(report_path(old_job, arm).parent.exists() for arm in ARMS):
            raise ValueError('Original controller advanced before the pause')
        if (not has_local_child(list(processes()), parent, CODE / 'camd/stage.py')
                and report_state(old_job, 'local_positive') != 'VERIFIED'):
            raise ValueError('Local-positive child exited without verified artifacts during handoff')
        state['paused_controller'] = {key: parent[key] for key in ('pid', 'starttime', 'job', 'command')}
        save('ORIGINAL_CONTROLLER_PAUSED')
        # The working node exposes /dev/kfd only to the allocation containing
        # GPU0. Preserve the active sampler, then release the parallel bundle
        # once this original allocation has been explicitly cancelled.
        while not state.get('original_cancelled'):
            if time.monotonic() >= deadline:
                raise TimeoutError('Timed out waiting for the active local-positive evaluation')
            status = report_state(old_job, 'local_positive')
            if status == 'VERIFIED':
                host_gate(old_job, 'base')
                host_gate(old_job, 'local_positive')
                state['verified'].extend(['base', 'local_positive'])
                save('LOCAL_POSITIVE_HOST_VERIFIED')
                cancel_original('Intentional parallel handoff after base and local-positive host verification')
                break
            if status == 'FAILED' or terminal_without_artifacts(old_job, 'local_positive'):
                state['failures']['local_positive'] = 'Original evaluation failed or allocation ended without verification'
                cancel_original('Original local-positive evaluation did not verify')
                raise RuntimeError('Local-positive evaluation did not verify')
            time.sleep(args.poll_seconds)
        for job in sorted(set(jobs.values())):
            release_attempted = True
            try:
                command(['scontrol', 'release', job])
                state['released'].append(job)
                save('PARALLEL_JOBS_RELEASED')
            except Exception as exc:
                for arm in ARMS:
                    if jobs[arm] == job:
                        state['failures'][arm] = 'Job release failed: ' + type(exc).__name__
                save()
        save('WAITING_FOR_PARALLEL_EVALUATIONS')
        pending = set(ARMS) - set(state['failures'])
        while pending:
            if time.monotonic() >= deadline:
                raise TimeoutError('Timed out waiting for parallel evaluations')
            for arm in sorted(pending):
                status = report_state(jobs[arm], arm)
                if status == 'VERIFIED':
                    try:
                        host_gate(jobs[arm], arm)
                        state['verified'].append(arm)
                    except Exception as exc:
                        state['failures'][arm] = 'Host verification failed: ' + type(exc).__name__
                    pending.remove(arm)
                    save()
                elif status == 'FAILED' or terminal_without_artifacts(jobs[arm], arm):
                    state['failures'][arm] = 'Parallel evaluation failed or allocation ended without verification'
                    pending.remove(arm)
                    save()
            if pending:
                time.sleep(args.poll_seconds)
        if state['failures']:
            raise RuntimeError('One or more parallel evaluations failed; completed artifacts are preserved')
        output_manifest = ROOT / 'results/evaluation_manifest.json'
        with output_manifest.open('x') as stream:
            json.dump(manifest(old_job, jobs), stream, indent=2)
            stream.write('\n')
        save('COMPARING')
        command([sys.executable, '-I', '-S', CODE / 'camd/compare_evaluations.py',
                 '--manifest', output_manifest, '--output', ROOT / 'results/comparison'])
        save('COMPLETED')
        print('ALL_FOUR_METHODS_COMPARED_IN_PARALLEL', flush=True)
    except BaseException as exc:
        state['error_type'] = type(exc).__name__
        # Before any release attempt, restoring the exact original process is
        # safe: all parallel jobs remain held. Afterwards fail closed to avoid
        # duplicate GPU work, leaving artifacts and the explicit handoff record.
        if paused and not release_attempted and not state.get('original_cancelled'):
            try:
                if same_process(parent, process(parent['pid'])):
                    os.kill(parent['pid'], signal.SIGCONT)
                    state['original_controller_restored'] = True
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                pass
        save('FAILED')
        raise


if __name__ == '__main__':
    main()
