"""ROCm control-plane tests; no GPU imports, model downloads, or child workers."""
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import run
from sampler_worker import SamplerClient, make_processor


class RocmVisibilityTests(unittest.TestCase):
    def test_eight_gpu_parent_preserves_rocr_and_normalizes_local_indices(self):
        physical = 'GPU-a,GPU-b,GPU-c,GPU-d,GPU-e,GPU-f,GPU-g,GPU-h'
        original = dict(STRIDE_RUNTIME='rocm', SLURM_JOB_ID='fixture',
                        ROCR_VISIBLE_DEVICES=physical, CUDA_VISIBLE_DEVICES=physical)
        with patch.dict(os.environ, original, clear=True):
            env = run.execution_environment('train')
        self.assertEqual(env['ROCR_VISIBLE_DEVICES'], physical)
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '0,1,2,3,4,5,6,7')
        self.assertEqual(env['HIP_VISIBLE_DEVICES'], env['CUDA_VISIBLE_DEVICES'])
        self.assertEqual(env['STRIDE_SCHEDULER_CUDA_VISIBLE_DEVICES'], physical)

    def test_single_gpu_eval_uses_zero_inside_physical_allocation(self):
        with patch.dict(os.environ, dict(STRIDE_RUNTIME='rocm', SLURM_JOB_ID='fixture',
                        ROCR_VISIBLE_DEVICES='7', CUDA_VISIBLE_DEVICES='7'), clear=True):
            env = run.execution_environment('eval')
        self.assertEqual(env['ROCR_VISIBLE_DEVICES'], '7')
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '0')
        self.assertEqual(env['HIP_VISIBLE_DEVICES'], '0')

    def test_missing_duplicate_or_conflicting_masks_fail_before_gpu_import(self):
        logical = '0,1,2,3,4,5,6,7'
        base = dict(STRIDE_RUNTIME='rocm', ROCR_VISIBLE_DEVICES=logical,
                    CUDA_VISIBLE_DEVICES=logical)
        for change in ({'ROCR_VISIBLE_DEVICES': ''},
                       {'ROCR_VISIBLE_DEVICES': '0,1,2,3,4,5,6,6'},
                       {'HIP_VISIBLE_DEVICES': '0,1,2,3'},
                       {'CUDA_VISIBLE_DEVICES': '7,6,5,4,3,2,1,0'}):
            with self.subTest(change=change), patch.dict(os.environ, {**base, **change}, clear=True):
                with self.assertRaises(ValueError):
                    run.execution_environment('train')

    def test_sampler_narrows_both_hip_and_cuda_preserves_rocr(self):
        logical = '0,1,2,3,4,5,6,7'
        env = dict(STRIDE_RUNTIME='rocm', ROCR_VISIBLE_DEVICES=logical,
                   CUDA_VISIBLE_DEVICES=logical, HIP_VISIBLE_DEVICES=logical,
                   RANK='2', WORLD_SIZE='4', LOCAL_RANK='2',
                   TORCHELASTIC_USE_AGENT_STORE='True')
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, env, clear=True), \
                patch('sampler_worker.subprocess.Popen', return_value=Mock()) as popen, \
                patch.object(SamplerClient, 'wait_file', return_value={'ok': True}):
            SamplerClient.launch('/fixture', Path(tmp)/'ipc', '6')
        child = popen.call_args.kwargs['env']
        self.assertEqual(child['ROCR_VISIBLE_DEVICES'], logical)
        self.assertEqual(child['HIP_VISIBLE_DEVICES'], '6')
        self.assertEqual(child['CUDA_VISIBLE_DEVICES'], '6')
        for name in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'TORCHELASTIC_USE_AGENT_STORE'):
            self.assertNotIn(name, child)

    def test_sampler_rejects_unallocated_logical_device(self):
        env = dict(STRIDE_RUNTIME='rocm', ROCR_VISIBLE_DEVICES='7',
                   CUDA_VISIBLE_DEVICES='0', HIP_VISIBLE_DEVICES='0')
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, env, clear=True), \
                patch('sampler_worker.subprocess.Popen') as popen:
            with self.assertRaises(ValueError):
                SamplerClient.launch('/fixture', Path(tmp)/'ipc', '7')
            popen.assert_not_called()

    def test_worker_rejects_conflicting_hip_before_importing_gpu_libraries(self):
        with patch.dict(os.environ, dict(STRIDE_RUNTIME='rocm', ROCR_VISIBLE_DEVICES='0,1',
                        CUDA_VISIBLE_DEVICES='1', HIP_VISIBLE_DEVICES='0,1'), clear=True):
            with self.assertRaisesRegex(RuntimeError, 'one logical HIP device'):
                make_processor('/unused')

    def test_runtime_profile_keeps_model_library_pins_and_changes_only_vllm(self):
        path = Path(__file__).resolve().parents[1]/'scripts'/'check_runtime.py'
        spec = importlib.util.spec_from_file_location('stride_runtime_profile_fixture', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.ROCM_PACKAGES, {**module.PACKAGES, 'vllm': '0.20.1'})


if __name__ == '__main__':
    unittest.main()
