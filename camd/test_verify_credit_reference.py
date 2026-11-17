"""Host reference readiness checks using inert files; no Torch/GPU/network."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from verify_credit_reference import CASES, TENSORS, verify_credit_reference


CAMD = Path(__file__).resolve().parent
SOURCES = ('training/credit.py', 'training/production_stride.py', 'training/experiment_credit.py',
           'training/feature_pooling.py', 'training/runtime_config.py', 'camd/credit_device_parity.py')


class CreditReferenceReadinessTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.code = Path(temporary.name)
        for name in SOURCES:
            path = self.code / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('# Inert source dependency fixture\n')
        (self.code / SOURCES[-1]).write_text(
            'SOURCE_FILES = ' + repr(SOURCES) + '\nATOL = 2e-4\nRTOL = 2e-4\n'
            'raise AssertionError("The host validator must not execute this module")\n')
        case = dict(passed=True, output_device='cpu', mapped_mask=[[True, False]],
                    tensors={key: [[0., 0.]] for key in TENSORS},
                    solver_calls=[dict(operation=op, device='cpu', dtype='torch.float32')
                                  for op in ('cholesky', 'slogdet')])
        self.reference = dict(passed=True, mode='reference', atol=2e-4, rtol=2e-4,
            source_hashes={name: hashlib.sha256((self.code / name).read_bytes()).hexdigest() for name in SOURCES},
            cases={name: copy.deepcopy(case) for name in CASES})
        self.path = self.code / 'camd/credit_cpu_reference.json'
        self.write()

    def write(self):
        self.path.write_text(json.dumps(self.reference))

    def test_valid_reference_checks_all_sources_without_executing_scientific_code(self):
        result = verify_credit_reference(self.code)
        self.assertTrue(result['passed'])
        self.assertEqual(result['source_files_checked'], 6)
        self.assertEqual(result['fixture_count'], 2)
        self.assertFalse(result['numerical_check_performed'])

    def test_every_guarded_source_edit_including_runtime_and_probe_is_rejected(self):
        for name in SOURCES:
            with self.subTest(name=name):
                path = self.code / name
                original = path.read_bytes()
                path.write_bytes(original + b'\n# Changed after reference generation\n')
                with self.assertRaisesRegex(ValueError, 'Stale CPU credit-reference source hashes'):
                    verify_credit_reference(self.code)
                path.write_bytes(original)

    def test_missing_or_extra_hash_is_not_a_partial_match(self):
        original = copy.deepcopy(self.reference)
        for change in ('missing', 'extra'):
            with self.subTest(change=change):
                self.reference = copy.deepcopy(original)
                if change == 'missing':
                    self.reference['source_hashes'].pop('training/runtime_config.py')
                else:
                    self.reference['source_hashes']['training/not_in_current_probe.py'] = '0' * 64
                self.write()
                with self.assertRaisesRegex(ValueError, 'source hashes'):
                    verify_credit_reference(self.code)

    def test_failed_wrong_mode_and_changed_tolerance_references_are_rejected(self):
        original = copy.deepcopy(self.reference)
        for key, value in [('passed', False), ('mode', 'check'), ('atol', .02), ('rtol', .02)]:
            with self.subTest(key=key):
                self.reference = copy.deepcopy(original)
                self.reference[key] = value
                self.write()
                with self.assertRaises(ValueError):
                    verify_credit_reference(self.code)

    def test_incomplete_failed_non_cpu_or_malformed_fixture_is_rejected(self):
        original = copy.deepcopy(self.reference)
        mutations = [lambda c: c.pop('all_correct'),
                     lambda c: c['all_correct'].update(passed=False),
                     lambda c: c['all_correct'].update(output_device='cuda:0'),
                     lambda c: c['all_correct']['tensors'].pop('advantages/grpo'),
                     lambda c: c['all_correct']['tensors'].update(main=[[0.]]),
                     lambda c: c['all_correct'].update(solver_calls=[])]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                self.reference = copy.deepcopy(original)
                mutate(self.reference['cases'])
                self.write()
                with self.assertRaises(ValueError):
                    verify_credit_reference(self.code)

    def test_cli_works_with_isolated_host_python_without_site_packages(self):
        result = subprocess.run([sys.executable, '-I', '-S', str(CAMD / 'verify_credit_reference.py'),
                                 '--code-dir', str(self.code)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(json.loads(result.stdout)['passed'])


if __name__ == '__main__':
    unittest.main()
