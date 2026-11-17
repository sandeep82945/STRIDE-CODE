"""Fail-closed GRES mapping and concurrent evaluation launch checks."""
import copy
import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import parallel_eval_workers as workers
import run_parallel_evaluations as host


class ParallelEvaluationTests(unittest.TestCase):
    def setUp(self):
        content = '\n'.join('Name=gpu Type=mi210 File=' + path for path in
                            ['/dev/kfd', *(f'/dev/dri/renderD{i}' for i in range(128, 136))]) + '\n'
        self.snapshot = {'job_id': '999', 'node': workers.NODE, 'source': '/etc/slurm/gres.conf',
                         'content': content, 'sha256': hashlib.sha256(content.encode()).hexdigest()}
        self.env = {'SLURM_JOB_ID': '999', 'SLURM_STEP_ID': '0', 'SLURMD_NODENAME': workers.NODE,
                    'SLURM_JOB_GPUS': '0,1,3,4', 'SLURM_STEP_GPUS': '0,1,3,4'}
        self.devices = [{'path': '/dev/kfd', 'character_device': True, 'open_read_write': True}]
        self.devices += [{'path': f'/dev/dri/renderD{minor}', 'minor': minor,
                          'character_device': True, 'open_read_write': minor in (128, 130, 131, 135)}
                         for minor in range(128, 136)]
        self.topology = [{'gpu_id': str(minor), 'properties': {'drm_render_minor': str(minor),
                                                             'unique_id': str(minor + 1000)}}
                         for minor in range(128, 136)]

    def bind(self, env=None, snapshot=None, devices=None, topology=None):
        return workers.bindings(env or self.env, snapshot or self.snapshot,
                                devices or self.devices, topology or self.topology)

    def test_only_allocated_render_slots_selected_even_when_extra_device_accessible(self):
        selected = self.bind()
        self.assertEqual([item['arm'] for item in selected], list(workers.ARMS))
        self.assertEqual([item['gres_slot'] for item in selected], [1, 3, 4])
        self.assertEqual([item['render_device']['minor'] for item in selected], [128, 130, 131])
        self.assertEqual(len({item['uuid'] for item in selected}), 3)
        self.assertNotIn(135, [item['render_device']['minor'] for item in selected])

    def test_missing_kfd_slot_and_different_step_allocation_rejected(self):
        variants = ({'SLURM_JOB_GPUS': '1,2,3,4', 'SLURM_STEP_GPUS': '1,2,3,4'},
                    {'SLURM_STEP_GPUS': '0,1,2,4'},
                    {'SLURM_STEP_GPUS': '0,1,3'},
                    {'SLURM_JOB_GPUS': '0,1,3,4,5'},
                    {'SLURM_JOB_GPUS': '0,1,1,3', 'SLURM_STEP_GPUS': '0,1,1,3'})
        for variant in variants:
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                self.bind(env={**self.env, **variant})

    def test_inaccessible_allocated_devices_rejected(self):
        for path in ('/dev/kfd', '/dev/dri/renderD130'):
            devices = copy.deepcopy(self.devices)
            next(item for item in devices if item['path'] == path)['open_read_write'] = False
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.bind(devices=devices)

    def test_unknown_node_and_foreign_snapshot_rejected(self):
        with self.assertRaises(ValueError):
            self.bind(env={**self.env, 'SLURMD_NODENAME': 'auh7-1b-gpu-192'})
        with self.assertRaises(ValueError):
            self.bind(snapshot={**self.snapshot, 'job_id': '998'})

    def test_modified_snapshot_and_changed_gres_order_rejected(self):
        with self.assertRaises(ValueError):
            self.bind(snapshot={**self.snapshot, 'content': self.snapshot['content'] + '# changed'})
        changed = self.snapshot['content'].replace('/dev/dri/renderD128', '/dev/dri/renderD999')
        with self.assertRaises(ValueError):
            self.bind(snapshot={**self.snapshot, 'content': changed,
                                'sha256': hashlib.sha256(changed.encode()).hexdigest()})

    def test_ambiguous_topology_and_reused_uuid_rejected(self):
        topology = copy.deepcopy(self.topology)
        topology.append(copy.deepcopy(topology[0]))
        with self.assertRaises(ValueError):
            self.bind(topology=topology)
        topology = copy.deepcopy(self.topology)
        topology[2]['properties']['unique_id'] = topology[0]['properties']['unique_id']
        with self.assertRaises(ValueError):
            self.bind(topology=topology)

    def test_all_three_processes_start_before_first_wait_and_failure_does_not_skip_others(self):
        events = []

        class Child:
            def __init__(self, command, **kwargs):
                self.arm = command[0]
                events.append(('launch', self.arm))

            def wait(self):
                events.append(('wait', self.arm))
                return 1 if self.arm == 'single_step' else 0

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plans = [{'arm': arm, 'command': [arm], 'environment': {},
                      'stdout': str(root / f'{arm}.out'), 'stderr': str(root / f'{arm}.err')}
                     for arm in workers.ARMS]
            with patch.object(workers.subprocess, 'Popen', Child):
                returns = workers.launch_children(plans, root)
        self.assertEqual(events[:3], [('launch', arm) for arm in workers.ARMS])
        self.assertEqual(events[3:], [('wait', arm) for arm in workers.ARMS])
        self.assertEqual(returns, {'single_step': 1, 'token_uniform': 0, 'grpo': 0})

    def exercise_host(self, worker_artifact, gate_failure=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calls = []

            def execute(command, check):
                calls.append(command)
                if command[0] == 'srun' and worker_artifact is not None:
                    output = root / 'results/job-999/parallel_evaluations/workers.json'
                    output.write_text(json.dumps(worker_artifact))
                returncode = 1 if command[0] != 'srun' and command[-1] == gate_failure else 0
                return subprocess.CompletedProcess(command, returncode)

            with patch.object(host, 'ROOT', root), patch.dict(host.os.environ, self.env), \
                    patch.object(Path, 'read_bytes', return_value=self.snapshot['content'].encode()), \
                    patch.object(host.subprocess, 'run', execute), contextlib.redirect_stdout(io.StringIO()):
                exit_code = 0
                try:
                    host.main()
                except SystemExit as error:
                    exit_code = error.code
            report = json.loads((root / 'results/job-999/parallel_evaluations/host_verification.json').read_text())
            self.assertEqual(len(calls), 4)
            self.assertEqual([command[-1] for command in calls[1:]], list(workers.ARMS))
            return exit_code, report

    def test_zero_returncodes_without_worker_completion_still_fail(self):
        exit_code, report = self.exercise_host(None)
        self.assertEqual(exit_code, 1)
        self.assertFalse(report['passed'])

    def test_one_host_gate_failure_rejects_bundle_and_other_gates_still_run(self):
        artifact = {'job_id': '999', 'passed': True, 'children': dict.fromkeys(workers.ARMS)}
        exit_code, report = self.exercise_host(artifact, gate_failure='token_uniform')
        self.assertEqual(exit_code, 1)
        self.assertFalse(report['passed'])

    def test_all_explicit_artifacts_and_host_gates_required_for_success(self):
        artifact = {'job_id': '999', 'passed': True, 'children': dict.fromkeys(workers.ARMS)}
        exit_code, report = self.exercise_host(artifact)
        self.assertEqual(exit_code, 0)
        self.assertTrue(report['passed'])


if __name__ == '__main__':
    unittest.main()
