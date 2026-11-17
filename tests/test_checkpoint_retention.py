"""Filesystem retention of all 321 committed snapshots for the full run."""
from pathlib import Path
import tempfile
import unittest

import resumable_checkpoint as checkpoints
from trainer_resumable import checkpoint_pins


def complete_pair(root, step):
    """Small opaque payloads exercise real hash/commit/pruning without Torch."""
    policy = root / f'adapter_{step:03d}'
    policy.mkdir()
    (policy / 'adapter_model.safetensors').write_bytes(f'fixture-{step}'.encode())
    (policy / 'adapter_config.json').write_text('{}')
    checkpoints.write_json(root / f'adapter_{step:03d}_manifest.json',
                           {p.name: checkpoints.digest(p) for p in policy.iterdir()})
    stage = checkpoints.prepare_state(root, step)
    (stage / 'optimizer.pt').write_bytes(b'opaque optimizer fixture')
    (stage / 'rng_rank_0.pt').write_bytes(b'opaque RNG fixture')
    checkpoints.commit_state(root, step, 1, {'fixture': 'retention'}, 'lora')


class CheckpointRetentionTests(unittest.TestCase):
    def test_latest_epoch_prunes_only_state_payloads_and_archives_exact_commit_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            originals = {}
            for step in range(9):
                complete_pair(root, step)
                originals[step] = {name: (root / f'state_{step:03d}' / name).read_bytes()
                                   for name in ('metadata.json', 'COMPLETE.json')}
                checkpoints.retain_latest_epoch_states(root, step, 4)
                self.assertEqual({p.name for p in root.glob('state_*')},
                                 {f'state_{i:03d}' for i in {step, step // 4 * 4}})
                self.assertEqual({p.name for p in root.glob('adapter_*') if p.is_dir()},
                                 {f'adapter_{i:03d}' for i in range(step + 1)})
                for previous in range(step + 1):
                    receipt = checkpoints.inspect_checkpoint_receipt(root, previous)
                    self.assertFalse(receipt['resumable'])
                    self.assertFalse(receipt['optimizer_payload_verified'])
                    path = root / 'checkpoint_receipts' / f'state_{previous:03d}'
                    self.assertEqual({name: (path / name).read_bytes() for name in originals[previous]}, originals[previous])
            self.assertEqual(checkpoints.inspect_checkpoint(root / 'state_008')['completed_step'], 8)
            with self.assertRaises(ValueError):
                checkpoints.inspect_checkpoint(root / 'checkpoint_receipts/state_007')

    def test_damaged_receipt_aborts_pruning_and_does_not_remove_adapter(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for step in range(3):
                complete_pair(root, step)
                checkpoints.retain_latest_epoch_states(root, step, 4)
            path = root / 'checkpoint_receipts/state_002/metadata.json'
            path.write_bytes(b'corrupted')
            complete_pair(root, 3)
            with self.assertRaisesRegex(ValueError, 'receipt metadata hash'):
                checkpoints.retain_latest_epoch_states(root, 3, 4)
            self.assertTrue((root / 'state_002/COMPLETE.json').is_file())
            self.assertTrue((root / 'adapter_002/adapter_model.safetensors').is_file())

    def test_full_schedule_retains_every_committed_pair_including_zero(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            pins = checkpoint_pins(320, 32, keep_all=True)
            for step in range(321):
                complete_pair(root, step)
                self.assertEqual(checkpoints.prune_complete_pairs(root, step, pins), [])
            self.assertEqual({p.name for p in root.glob('state_*')},
                             {f'state_{step:03d}' for step in range(321)})
            for step in range(321):
                verified = checkpoints.inspect_checkpoint(root / f'state_{step:03d}')
                self.assertEqual(verified['completed_step'], step)

    def test_default_retention_still_prunes_non_epoch_pairs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            pins = checkpoint_pins(6, 3)
            for step in range(7):
                complete_pair(root, step)
                checkpoints.prune_complete_pairs(root, step, pins)
            self.assertEqual({p.name for p in root.glob('state_*')},
                             {'state_003', 'state_005', 'state_006'})
            self.assertFalse((root / 'adapter_000').exists())
            self.assertFalse((root / 'adapter_004').exists())


if __name__ == '__main__':
    unittest.main()
