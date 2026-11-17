"""Check binding decisions with the actual successful probe's device identity."""
import copy
import unittest

from allocated_gpu import normalize, step_size


def fixture():
    env = {'SLURM_JOB_ID': '189504', 'SLURM_STEP_ID': '0',
           'SLURM_JOB_GPUS': '0', 'SLURM_STEP_GPUS': '0',
           'ROCR_VISIBLE_DEVICES': '0', 'CUDA_VISIBLE_DEVICES': '0',
           'GPU_DEVICE_ORDINAL': '0', 'OTHER_SETTING': 'preserved'}
    devices = [{'path': '/dev/kfd', 'open_read_write': True, 'character_device': True},
               {'path': '/dev/dri/renderD135', 'minor': 135,
                'open_read_write': True, 'character_device': True},
               {'path': '/dev/dri/renderD128', 'minor': 128,
                'open_read_write': False, 'character_device': True}]
    nodes = [{'node': '8', 'gpu_id': '321', 'properties':
              {'drm_render_minor': '135', 'unique_id': str(int('e29968d2bb66e9cc', 16))}}]
    return env, devices, nodes


class AllocationTests(unittest.TestCase):
    def test_success_uses_permitted_render_uuid_not_slurm_ordinal(self):
        env, devices, nodes = fixture()
        before = copy.deepcopy(env)
        updated, evidence = normalize(env, devices, nodes)
        self.assertEqual(updated['ROCR_VISIBLE_DEVICES'], 'GPU-e29968d2bb66e9cc')
        self.assertEqual(updated['HIP_VISIBLE_DEVICES'], '0')
        self.assertEqual(updated['CUDA_VISIBLE_DEVICES'], '0')
        self.assertNotIn('GPU_DEVICE_ORDINAL', updated)
        self.assertEqual(updated['OTHER_SETTING'], 'preserved')
        self.assertEqual(env, before)
        self.assertEqual(evidence['original_environment']['GPU_DEVICE_ORDINAL'], '0')
        self.assertTrue(evidence['mapping_verified'])
        self.assertFalse(evidence['gpu_execution_tested'])

    def test_eight_gpu_step_is_unchanged_and_needs_no_device_scan(self):
        env, _, _ = fixture()
        env.update(SLURM_JOB_GPUS='0,1,2,3,4,5,6,7', SLURM_STEP_GPUS='0,1,2,3,4,5,6,7')
        updated, evidence = normalize(env)
        self.assertEqual(updated, env)
        self.assertEqual(evidence['action'], 'eight_gpu_passthrough')

    def test_ambiguous_or_missing_render_access_rejected(self):
        for available in (0, 2):
            env, devices, nodes = fixture()
            devices[1]['open_read_write'] = available > 0
            devices[2]['open_read_write'] = available > 1
            with self.subTest(available=available), self.assertRaises(ValueError):
                normalize(env, devices, nodes)

    def test_kfd_permission_required(self):
        env, devices, nodes = fixture()
        devices[0]['open_read_write'] = False
        with self.assertRaises(ValueError):
            normalize(env, devices, nodes)

    def test_missing_ambiguous_or_invalid_topology_rejected(self):
        for kind in ('missing', 'duplicate', 'bad_uuid', 'cpu_node'):
            env, devices, nodes = fixture()
            if kind == 'missing':
                nodes.clear()
            elif kind == 'duplicate':
                nodes.append(copy.deepcopy(nodes[0]))
            elif kind == 'bad_uuid':
                nodes[0]['properties']['unique_id'] = '0'
            else:
                nodes[0]['gpu_id'] = '0'
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                normalize(env, devices, nodes)

    def test_invalid_or_unallocated_step_metadata_rejected(self):
        for key, value in [('SLURM_JOB_ID', ''), ('SLURM_STEP_ID', 'batch'),
                           ('SLURM_STEP_GPUS', ''), ('SLURM_STEP_GPUS', '0,0'),
                           ('SLURM_STEP_GPUS', '1'), ('SLURM_STEP_GPUS', '0,1')]:
            env, _, _ = fixture()
            env[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                step_size(env)


if __name__ == '__main__':
    unittest.main()
