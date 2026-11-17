"""CPU checkpoints: immutable publication, tamper rejection, RNG, and retention.

CUDA RNG APIs are explicitly mocked; these tests make no GPU claim.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

import resumable_checkpoint as cp


CONTRACT = {'arm': 'local_positive', 'seed': 17, 'batch_size': 4}


def adapter(root, step):
    directory = root / f'adapter_{step:03d}'
    directory.mkdir()
    (directory / 'adapter_model.safetensors').write_bytes(b'fixture; never loaded as a model')
    (directory / 'adapter_config.json').write_text('{"fixture": "cpu"}')
    cp.write_json(root / f'adapter_{step:03d}_manifest.json',
                  {p.name: cp.digest(p) for p in directory.iterdir()})
    return directory


def staged(root, step=1, world_size=1):
    adapter(root, step)
    stage = cp.prepare_state(root, step)
    cp.save_torch(stage / 'optimizer.pt', {'state': {}, 'param_groups': []})
    for rank in range(world_size):
        cp.save_torch(stage / f'rng_rank_{rank}.pt', cp.capture_rng())
    return stage


def checkpoint(root, step=1, world_size=1):
    staged(root, step, world_size)
    return cp.commit_state(root, step, world_size, CONTRACT)


def rewrite_metadata(state, mutate):
    """Rehash deliberately altered metadata to exercise deeper validation."""
    path = state / 'metadata.json'; meta = json.loads(path.read_text())
    mutate(meta); path.write_text(json.dumps(meta))
    complete_path = state / 'COMPLETE.json'; complete = json.loads(complete_path.read_text())
    complete['metadata_sha256'] = cp.digest(path)
    complete_path.write_text(json.dumps(complete))


class AtomicCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_digest_streams_entire_file_including_multiple_blocks(self):
        path = self.root / 'large'; payload = b'abcd' * (2 * 1024 * 1024 + 3)
        path.write_bytes(payload)
        self.assertEqual(cp.digest(path), hashlib.sha256(payload).hexdigest())

    def test_directory_and_file_guards_reject_missing_wrong_type_and_symlink(self):
        directory = self.root / 'dir'; directory.mkdir()
        regular = self.root / 'file'; regular.write_bytes(b'x')
        file_link = self.root / 'file_link'; file_link.symlink_to(regular)
        dir_link = self.root / 'dir_link'; dir_link.symlink_to(directory)
        self.assertEqual(cp._regular(regular), regular)
        self.assertEqual(cp._directory(directory), directory)
        for bad in (directory, file_link, self.root / 'missing'):
            with self.subTest(bad=bad.name), self.assertRaises(ValueError): cp._regular(bad)
        for bad in (regular, dir_link, self.root / 'missing'):
            with self.subTest(bad=bad.name), self.assertRaises(ValueError): cp._directory(bad)

    def test_fsync_directory_closes_descriptor_when_fsync_fails(self):
        real_close = os.close
        with patch.object(cp.os, 'fsync', side_effect=OSError('injected')), patch.object(cp.os, 'close', wraps=real_close) as close:
            with self.assertRaisesRegex(OSError, 'injected'): cp._fsync_directory(self.root)
        close.assert_called_once()

    def test_json_is_immutable_and_serialization_failure_remains_ineligible(self):
        path = self.root / 'value.json'; cp.write_json(path, {'b': 2, 'a': 'λ'})
        self.assertEqual(json.loads(path.read_text()), {'a': 'λ', 'b': 2})
        self.assertFalse(path.with_name('value.json.writing').exists())
        with self.assertRaises(FileExistsError): cp.write_json(path, {'overwrite': True})
        invalid = self.root / 'invalid.json'
        with self.assertRaises(ValueError): cp.write_json(invalid, {'bad': float('nan')})
        self.assertFalse(invalid.exists())
        self.assertTrue(invalid.with_name('invalid.json.writing').exists())
        with self.assertRaises(FileExistsError): cp.write_json(invalid, {'retry': 1})

    def test_json_competing_final_file_is_never_overwritten(self):
        path = self.root / 'value.json'; real_link = os.link
        def competing_link(source, destination):
            Path(destination).write_text('{"winner": true}')
            return real_link(source, destination)
        with patch.object(cp.os, 'link', side_effect=competing_link):
            with self.assertRaises(FileExistsError): cp.write_json(path, {'loser': True})
        self.assertEqual(json.loads(path.read_text()), {'winner': True})

    def test_save_torch_weights_only_roundtrip_and_immutability(self):
        path = self.root / 'state.pt'; value = {'tensor': torch.tensor([2, 3]), 'scalar': 7}
        cp.save_torch(path, value)
        loaded = torch.load(path, weights_only=True)
        self.assertTrue(torch.equal(loaded['tensor'], value['tensor']))
        self.assertEqual(loaded['scalar'], 7)
        original = path.read_bytes()
        with self.assertRaises(FileExistsError): cp.save_torch(path, {})
        self.assertEqual(path.read_bytes(), original)

    def test_interrupted_torch_save_cannot_be_replayed_over_partial_evidence(self):
        path = self.root / 'partial.pt'
        def fail(value, stream):
            stream.write(b'partial'); raise OSError('interrupted')
        with patch.object(torch, 'save', side_effect=fail):
            with self.assertRaisesRegex(OSError, 'interrupted'): cp.save_torch(path, {})
        self.assertEqual(path.read_bytes(), b'partial')
        with self.assertRaises(FileExistsError): cp.save_torch(path, {})

    def test_prepare_state_validates_step_and_rejects_replay(self):
        for bad in (True, False, -1, 1., '1', None):
            with self.subTest(bad=bad), self.assertRaises(ValueError): cp.prepare_state(self.root, bad)
        stage = cp.prepare_state(self.root, 0)
        self.assertEqual(stage.name, 'state_000.writing')
        with self.assertRaises(FileExistsError): cp.prepare_state(self.root, 0)
        stage.rename(self.root / 'state_000')
        with self.assertRaises(FileExistsError): cp.prepare_state(self.root, 0)
        self.assertEqual(cp.prepare_state(self.root, 1234).name, 'state_1234.writing')

    def test_commit_requires_strict_step_and_world_size(self):
        for bad in (True, False, -1, 1., '1', None):
            with self.subTest(step=bad), self.assertRaises(ValueError): cp.commit_state(self.root, bad, 1, CONTRACT)
        for bad in (True, False, 0, -1, 1., '1', None):
            with self.subTest(world=bad), self.assertRaises(ValueError): cp.commit_state(self.root, 1, bad, CONTRACT)

    def test_commit_requires_exact_files_and_does_not_publish_incomplete_state(self):
        stage = staged(self.root, world_size=2)
        (stage / 'rng_rank_1.pt').unlink()
        with self.assertRaisesRegex(ValueError, 'Incomplete'): cp.commit_state(self.root, 1, 2, CONTRACT)
        self.assertFalse((self.root / 'state_001').exists())
        cp.save_torch(stage / 'rng_rank_1.pt', cp.capture_rng())
        (stage / 'unexpected').write_bytes(b'')
        with self.assertRaisesRegex(ValueError, 'unexpected'): cp.commit_state(self.root, 1, 2, CONTRACT)
        (stage / 'unexpected').unlink()
        state = cp.commit_state(self.root, 1, 2, CONTRACT)
        self.assertFalse(stage.exists())
        self.assertEqual(cp.inspect_checkpoint(state, CONTRACT, 2)['completed_step'], 1)

    def test_crash_before_complete_preserves_final_but_cannot_resume(self):
        staged(self.root); original_write = cp.write_json
        def injected(path, value):
            if Path(path).name == 'COMPLETE.json': raise OSError('power loss')
            return original_write(path, value)
        with patch.object(cp, 'write_json', side_effect=injected):
            with self.assertRaisesRegex(OSError, 'power loss'): cp.commit_state(self.root, 1, 1, CONTRACT)
        state = self.root / 'state_001'
        self.assertTrue((state / 'metadata.json').is_file())
        self.assertFalse((state / 'COMPLETE.json').exists())
        with self.assertRaises(ValueError): cp.inspect_checkpoint(state)
        with self.assertRaises(FileExistsError): cp.prepare_state(self.root, 1)

    def test_crash_before_metadata_publication_keeps_staging_ineligible(self):
        stage = staged(self.root)
        with patch.object(cp, 'write_json', side_effect=OSError('injected metadata failure')):
            with self.assertRaises(OSError): cp.commit_state(self.root, 1, 1, CONTRACT)
        self.assertTrue(stage.is_dir())
        self.assertFalse((self.root / 'state_001').exists())
        with self.assertRaises(ValueError): cp.inspect_checkpoint(stage)


class CheckpointValidationTests(unittest.TestCase):
    def test_adapter_manifest_requires_weights_config_and_exact_directory(self):
        for damage in ('not_dict', 'missing_required', 'unexpected_file', 'wrong_hash', 'symlink_file', 'symlink_manifest', 'symlink_directory'):
            with self.subTest(damage=damage), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); path = adapter(root, 1); manifest = root / 'adapter_001_manifest.json'
                if damage == 'not_dict': manifest.write_text('[]')
                elif damage == 'missing_required': manifest.write_text('{}')
                elif damage == 'unexpected_file': (path / 'extra').write_text('x')
                elif damage == 'wrong_hash': (path / 'adapter_model.safetensors').write_bytes(b'changed')
                elif damage == 'symlink_file':
                    original = path / 'adapter_model.safetensors'; outside = root / 'outside'; original.rename(outside); original.symlink_to(outside)
                elif damage == 'symlink_manifest':
                    outside = root / 'manifest'; manifest.rename(outside); manifest.symlink_to(outside)
                else:
                    outside = root / 'outside'; path.rename(outside); path.symlink_to(outside)
                with self.assertRaises(ValueError): cp.verify_adapter(path)

    def test_adapter_hashes_and_manifest_hash_are_independently_pinned(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); state = checkpoint(root)
            manifest = root / 'adapter_001_manifest.json'
            manifest.write_text(json.dumps(json.loads(manifest.read_text()), separators=(',', ':')))
            with self.assertRaisesRegex(ValueError, 'provenance'): cp.inspect_checkpoint(state)

    def test_checkpoint_name_complete_version_and_step_must_match(self):
        for field, value in [('format_version', 999), ('completed_step', 2), ('metadata_sha256', 'bad')]:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                state = checkpoint(Path(directory)); path = state / 'COMPLETE.json'
                data = json.loads(path.read_text()); data[field] = value; path.write_text(json.dumps(data))
                with self.assertRaises(ValueError): cp.inspect_checkpoint(state)
        with tempfile.TemporaryDirectory() as directory:
            state = checkpoint(Path(directory)); wrong = state.with_name('state_1'); state.rename(wrong)
            with self.assertRaisesRegex(ValueError, 'state_NNN'): cp.inspect_checkpoint(wrong)

    def test_rehashed_metadata_cannot_bypass_semantic_contract_checks(self):
        mutations = {'version': lambda m: m.update(format_version=999),
                     'step': lambda m: m.update(completed_step=2),
                     'world bool': lambda m: m.update(world_size=True),
                     'world zero': lambda m: m.update(world_size=0),
                     'world float': lambda m: m.update(world_size=1.),
                     'missing state': lambda m: m.update(state_files={}),
                     'path escape': lambda m: m.update(adapter_name='../adapter_001'),
                     'wrong provenance': lambda m: m.update(adapter={})}
        for label, mutate in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                state = checkpoint(Path(directory)); rewrite_metadata(state, mutate)
                with self.assertRaises(ValueError): cp.inspect_checkpoint(state)

    def test_missing_extra_symlink_and_tampered_state_files_are_rejected(self):
        for damage in ('missing_complete', 'missing_metadata', 'missing_rng', 'extra', 'rng_hash', 'optimizer_hash', 'symlink_rng', 'symlink_state'):
            with self.subTest(damage=damage), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); state = checkpoint(root)
                if damage.startswith('missing_'):
                    name = {'missing_complete': 'COMPLETE.json', 'missing_metadata': 'metadata.json', 'missing_rng': 'rng_rank_0.pt'}[damage]
                    (state / name).unlink()
                elif damage == 'extra': (state / 'unexpected').write_bytes(b'')
                elif damage.endswith('_hash'):
                    name = 'rng_rank_0.pt' if damage == 'rng_hash' else 'optimizer.pt'
                    with (state / name).open('ab') as stream: stream.write(b'tamper')
                elif damage == 'symlink_rng':
                    path = state / 'rng_rank_0.pt'; outside = root / 'rng'; path.rename(outside); path.symlink_to(outside)
                else:
                    outside = root / 'outside'; state.rename(outside); state.symlink_to(outside)
                with self.assertRaises(ValueError): cp.inspect_checkpoint(state)

    def test_contract_topology_and_load_rank_are_checked_before_deserialization(self):
        with tempfile.TemporaryDirectory() as directory:
            state = checkpoint(Path(directory), world_size=2)
            with self.assertRaisesRegex(ValueError, 'world size'): cp.inspect_checkpoint(state, expected_world_size=1)
            with self.assertRaisesRegex(ValueError, 'contract'): cp.inspect_checkpoint(state, expected_contract={'arm': 'other'})
            meta = cp.inspect_checkpoint(state); optimizer = Mock()
            for bad in (-1, 2, True, False, 0., '0', None):
                with self.subTest(rank=bad), patch.object(torch, 'load') as load:
                    with self.assertRaisesRegex(ValueError, 'Rank'): cp.load_training_state(meta, optimizer, bad)
                    load.assert_not_called()
            rng = cp.load_training_state(meta, optimizer, 1)
            optimizer.load_state_dict.assert_called_once_with({'state': {}, 'param_groups': []})
            self.assertEqual(set(rng), {'python', 'numpy', 'torch_cpu', 'cuda'})
            (state / 'optimizer.pt').write_bytes(b'tampered after inspect')
            with patch.object(torch, 'load') as load:
                with self.assertRaisesRegex(ValueError, 'hash mismatch'): cp.load_training_state(meta, optimizer, 0)
                load.assert_not_called()


class RNGAndDistributedSaveTests(unittest.TestCase):
    def setUp(self):
        self.saved_rng = cp.capture_rng(); self.addCleanup(cp.restore_rng, self.saved_rng)

    def test_cpu_rng_replays_python_numpy_gaussian_cache_and_torch(self):
        random.seed(13); np.random.seed(14); torch.manual_seed(15)
        np.random.normal()  # Leave a cached Gaussian in NumPy's MT19937 state.
        state = cp.capture_rng()
        self.assertEqual(state['numpy'][3], 1)
        self.assertIsNone(state['cuda'])
        expected = (random.random(), np.random.normal(size=9), torch.randn(9))
        random.seed(1); np.random.seed(2); torch.manual_seed(3)
        cp.restore_rng(state)
        actual = (random.random(), np.random.normal(size=9), torch.randn(9))
        self.assertEqual(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])
        self.assertTrue(torch.equal(actual[2], expected[2]))

    def test_cuda_rng_is_explicitly_mocked_and_targets_only_requested_device(self):
        fake_cuda = torch.tensor([1, 2, 3], dtype=torch.uint8)
        with patch.object(torch.cuda, 'get_rng_state', return_value=fake_cuda) as get:
            state = cp.capture_rng(device='cuda:2')
        get.assert_called_once_with('cuda:2')
        with patch.object(torch.cuda, 'set_rng_state') as set_state:
            cp.restore_rng(state, device='cuda:2')
        self.assertTrue(torch.equal(set_state.call_args.args[0], fake_cuda))
        self.assertEqual(set_state.call_args.args[1], 'cuda:2')
        with self.assertRaisesRegex(ValueError, 'CUDA RNG contract'): cp.restore_rng(state)
        with self.assertRaisesRegex(ValueError, 'CUDA RNG contract'): cp.restore_rng(cp.capture_rng(), device='cuda:0')

    def test_cpu_capture_does_not_call_cuda(self):
        with patch.object(torch.cuda, 'get_rng_state') as get, patch.object(torch.cuda, 'set_rng_state') as set_state:
            cp.restore_rng(cp.capture_rng())
        get.assert_not_called(); set_state.assert_not_called()

    def test_single_rank_distributed_save_commits_after_three_barriers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); adapter(root, 1)
            events = []
            dist = NSDist(rank=0, world_size=1, events=events)
            optimizer = Mock(); optimizer.state_dict.return_value = {'state': {}, 'param_groups': []}
            original_commit = cp.commit_state
            def commit(*args):
                self.assertEqual(events, ['barrier', 'barrier'])
                return original_commit(*args)
            with patch.object(cp, 'commit_state', side_effect=commit):
                result = cp.save_distributed_state(root, 1, object(), optimizer, dist, CONTRACT)
            self.assertEqual(events, ['barrier'] * 3)
            self.assertEqual(cp.inspect_checkpoint(result)['completed_step'], 1)
            optimizer.state_dict.assert_called_once()

    def test_nonzero_rank_writes_only_its_rng_and_never_optimizer_or_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); stage = cp.prepare_state(root, 1); events = []
            dist = NSDist(rank=1, world_size=2, events=events); optimizer = Mock()
            with patch.object(cp, 'prepare_state') as prepare, patch.object(cp, 'commit_state') as commit:
                result = cp.save_distributed_state(root, 1, object(), optimizer, dist, CONTRACT)
            self.assertEqual(result, str(root / 'state_001'))
            self.assertEqual([p.name for p in stage.iterdir()], ['rng_rank_1.pt'])
            self.assertEqual(events, ['barrier'] * 3)
            prepare.assert_not_called(); commit.assert_not_called(); optimizer.state_dict.assert_not_called()


class NSDist:
    def __init__(self, rank, world_size, events): self.rank, self.world_size, self.events = rank, world_size, events
    def get_rank(self): return self.rank
    def get_world_size(self): return self.world_size
    def barrier(self): self.events.append('barrier')


class PruningTests(unittest.TestCase):
    def test_pruning_keeps_latest_two_pinned_future_and_incomplete_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for step in (0, 1, 2, 3, 4, 5): checkpoint(root, step)
            (root / 'state_006.writing').mkdir(); (root / 'state_007').mkdir()
            removed = cp.prune_complete_pairs(root, current_step=4, pinned_steps={1})
            self.assertEqual(removed, [0, 2])
            for step in (1, 3, 4, 5): cp.inspect_checkpoint(root / f'state_{step:03d}')
            for step in removed:
                self.assertFalse((root / f'state_{step:03d}').exists())
                self.assertFalse((root / f'adapter_{step:03d}').exists())
                self.assertTrue((root / f'adapter_{step:03d}_manifest.json').is_file())
            self.assertTrue((root / 'state_006.writing').is_dir())
            self.assertTrue((root / 'state_007').is_dir())
            self.assertEqual(cp.prune_complete_pairs(root, 4, {1}), [])

    def test_corrupt_old_checkpoint_is_not_deleted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); state = checkpoint(root, 0)
            (state / 'optimizer.pt').write_bytes(b'tampered')
            with self.assertRaises(ValueError): cp.prune_complete_pairs(root, 4, set())
            self.assertTrue((state / 'COMPLETE.json').exists())
            self.assertTrue((root / 'adapter_000').is_dir())

    def test_symlinked_checkpoint_cannot_escape_attempt_during_pruning(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); attempt = root / 'attempt'; attempt.mkdir()
            outside = root / 'outside'; outside.mkdir(); state = checkpoint(outside, 0)
            (attempt / 'state_000').symlink_to(state)
            with self.assertRaises(ValueError): cp.prune_complete_pairs(attempt, 4, set())
            self.assertTrue((state / 'COMPLETE.json').exists())
            self.assertTrue((outside / 'adapter_000').is_dir())

    def test_pruning_removes_completion_marker_before_state_or_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); state = checkpoint(root, 0)
            with patch.object(cp.shutil, 'rmtree', side_effect=OSError('interrupted deletion')):
                with self.assertRaisesRegex(OSError, 'interrupted'): cp.prune_complete_pairs(root, 4, set())
            self.assertFalse((state / 'COMPLETE.json').exists())
            self.assertTrue(state.is_dir())
            self.assertTrue((root / 'adapter_000').is_dir())


if __name__ == '__main__':
    unittest.main()
