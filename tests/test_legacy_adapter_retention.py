"""Retention checks for the preserved, currently unused adapter-only helper."""
from pathlib import Path
import tempfile
import unittest

from trainer_resumable import prune_synchronization_adapters


class LegacyAdapterRetentionTests(unittest.TestCase):
    def adapter(self, root, step, manifest=True):
        folder = root / f'adapter_{step:03d}'
        folder.mkdir()
        (folder / 'adapter_model.safetensors').write_bytes(b'test adapter')
        if manifest:
            folder.with_name(folder.name + '_manifest.json').write_text('{}')
        return folder

    def test_retains_epoch_current_previous_and_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for step in range(5):
                self.adapter(root, step)
            prune_synchronization_adapters(root, 4, {1}, finetune_mode='lora')
            self.assertEqual({p.name for p in root.glob('adapter_*') if p.is_dir()},
                             {'adapter_001', 'adapter_003', 'adapter_004'})
            self.assertEqual(len(list(root.glob('*_manifest.json'))), 5)

    def test_missing_manifest_refuses_deletion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = self.adapter(root, 0, manifest=False)
            with self.assertRaisesRegex(ValueError, 'without its manifest'):
                prune_synchronization_adapters(root, 4, set(), finetune_mode='lora')
            self.assertTrue(folder.is_dir())

    def test_symlink_or_file_refuses_deletion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outside = root / 'outside'
            outside.mkdir()
            for kind in ('symlink', 'file'):
                candidate = root / 'adapter_000'
                if kind == 'symlink':
                    candidate.symlink_to(outside, target_is_directory=True)
                else:
                    candidate.write_text('unexpected file')
                with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, 'Unexpected'):
                    prune_synchronization_adapters(root, 4, set(), finetune_mode='lora')
                self.assertTrue(outside.is_dir())
                candidate.unlink()


if __name__ == '__main__':
    unittest.main()
