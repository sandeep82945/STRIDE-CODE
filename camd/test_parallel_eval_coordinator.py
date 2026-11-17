import unittest
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

import parallel_eval_coordinator as coordinator


class CoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.script = Path('/pilot/camd/run_evaluations.py')
        self.parent = {'pid': 71, 'starttime': 421, 'ppid': 9, 'state': 'S', 'job': '189545',
                       'command': ['/usr/bin/python3', '-I', '-S', str(self.script)]}

    def test_selects_exact_controller_and_ignores_other_jobs_and_similar_commands(self):
        others = [dict(self.parent, pid=72, job='189546'),
                  dict(self.parent, pid=73, command=self.parent['command'] + ['extra']),
                  dict(self.parent, pid=74, command=['srun', str(self.script)])]
        self.assertEqual(coordinator.select_controller([*others, self.parent], '189545', self.script), self.parent)

    def test_ambiguous_controller_is_rejected(self):
        with self.assertRaises(ValueError):
            coordinator.select_controller([self.parent, dict(self.parent, pid=72)], '189545', self.script)

    def test_pid_reuse_is_rejected(self):
        self.assertFalse(coordinator.same_process(self.parent, dict(self.parent, starttime=422)))

    def test_child_must_be_direct_local_positive_srun(self):
        stage = Path('/pilot/camd/stage.py')
        child = {'ppid': 71, 'command': ['srun', '--exact', '/bin/bash', '/pilot/enter.sh',
                                       'gpu', str(stage), 'eval', '--arm', 'local_positive']}
        self.assertTrue(coordinator.has_local_child([child], self.parent, stage))
        self.assertFalse(coordinator.has_local_child([dict(child, ppid=2)], self.parent, stage))
        self.assertFalse(coordinator.has_local_child([dict(child, command=child['command'][:-1] + ['grpo'])],
                                                    self.parent, stage))

    def test_requires_held_bundle_or_three_distinct_jobs(self):
        plan = {'version': 1, 'old_job': '189545', 'held': True,
                'jobs': {'single_step': '20', 'token_uniform': '21', 'grpo': '22'}}
        self.assertEqual(coordinator.validate_plan(plan), plan['jobs'])
        bundle = dict(plan, jobs={arm: '20' for arm in coordinator.ARMS})
        self.assertEqual(set(coordinator.validate_plan(bundle).values()), {'20'})
        for invalid in [dict(plan, held=False), dict(plan, old_job='99'),
                        dict(plan, jobs=dict(plan['jobs'], grpo='20'))]:
            with self.assertRaises(ValueError):
                coordinator.validate_plan(invalid)

    def test_manifest_preserves_existing_results_and_maps_new_jobs(self):
        result = coordinator.manifest('189545', {'single_step': '20', 'token_uniform': '21', 'grpo': '22'})
        self.assertTrue(result['baseline'].endswith('/base-189545'))
        self.assertTrue(result['arms']['local_positive']['4'].endswith('/local_positive-epoch4-189545'))
        self.assertTrue(result['arms']['grpo']['4'].endswith('/grpo-epoch4-22'))
        self.assertEqual(set(result['arms']), {'local_positive', 'single_step', 'token_uniform', 'grpo'})

    def test_bundle_release_occurs_once_after_both_existing_host_gates_and_cancellation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code = root / 'code/stride_handoff'
            (root / 'results').mkdir()
            plan_path = root / 'plan.json'
            plan_path.write_text(json.dumps({'version': 1, 'old_job': '189545', 'held': True,
                                            'jobs': {arm: '777' for arm in coordinator.ARMS}}))
            parent = dict(self.parent, command=['/usr/bin/python3', '-I', '-S', str(code / 'camd/run_evaluations.py')])
            child = {'ppid': parent['pid'], 'job': '189545',
                     'command': ['srun', str(code / 'camd/stage.py'), 'eval', '--arm', 'local_positive']}
            events = []

            def run(arguments):
                values = [str(value) for value in arguments]
                events.append(values)
                if values[:3] == ['scontrol', 'show', 'job']:
                    return f'JobState=PENDING Reason=JobHeldUser UserId=owner({os.getuid()})'
                return ''

            with patch.object(coordinator, 'ROOT', root), patch.object(coordinator, 'CODE', code), \
                    patch.object(coordinator.sys, 'argv', ['coordinator', '--plan', str(plan_path)]), \
                    patch.dict(os.environ, {'SLURM_JOB_ID': '778'}), \
                    patch.object(coordinator, 'processes', side_effect=lambda: iter([parent, child])), \
                    patch.object(coordinator, 'process', return_value=dict(parent, state='T')), \
                    patch.object(coordinator.os, 'kill'), \
                    patch.object(coordinator, 'report_state', return_value='VERIFIED'), \
                    patch.object(coordinator, 'command', side_effect=run):
                coordinator.main()
            release = events.index(['scontrol', 'release', '777'])
            cancel = events.index(['scancel', '189545'])
            local_gate = next(i for i, value in enumerate(events) if value[-3:] == ['189545', 'eval', 'local_positive'])
            base_gate = next(i for i, value in enumerate(events) if value[-3:] == ['189545', 'eval', 'base'])
            self.assertLess(base_gate, cancel)
            self.assertLess(local_gate, cancel)
            self.assertLess(cancel, release)
            self.assertEqual(events.count(['scontrol', 'release', '777']), 1)
            state = json.loads((root / 'results/parallel_handoff_job-778.json').read_text())
            self.assertEqual(state['status'], 'COMPLETED')
            self.assertEqual(set(state['verified']), {'base', 'local_positive', *coordinator.ARMS})


if __name__ == '__main__':
    unittest.main()
