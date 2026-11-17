"""CPU resume continuity, transactional checkpoints, and unchanged update math."""
import ast
import copy
import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

import resumable_checkpoint as cp


ROOT = Path(__file__).resolve().parents[1] / "training"
CONTRACT = dict(arm='local_positive', batch_size=64, K=8, epochs=4, seed=42)


def fixture_adapter(root, step, model):
    """The weights-only torch fixture uses the production expected filename."""
    adapter = root / f'adapter_{step:03d}'
    adapter.mkdir()
    cp.save_torch(adapter / 'adapter_model.safetensors', model.state_dict())
    cp.write_json(adapter / 'adapter_config.json', {'fixture': 'tiny_cpu_model'})
    cp.write_json(adapter.with_name(adapter.name + '_manifest.json'),
                  {path.name: cp.digest(path) for path in adapter.iterdir()})
    return adapter


def tiny_model():
    return torch.nn.Sequential(torch.nn.Linear(4, 7), torch.nn.Dropout(.35),
                               torch.nn.Tanh(), torch.nn.Linear(7, 2))


def stochastic_update(model, optimizer, step):
    optimizer.zero_grad(set_to_none=True)
    x = torch.randn(5, 4) * (1 + random.random()) + float(np.random.normal())
    target = torch.randn(5, 2) + step / 10
    loss = (model(x) - target).square().mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    optimizer.step()
    return float(loss.detach())


def assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        if not torch.equal(left, right):
            raise AssertionError('Tensor mismatch')
    elif isinstance(left, dict):
        if left.keys() != right.keys():
            raise AssertionError('Dictionary fields mismatch')
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        if len(left) != len(right):
            raise AssertionError('Sequence length mismatch')
        for a, b in zip(left, right):
            assert_nested_equal(a, b)
    elif left != right:
        raise AssertionError(f'Value mismatch: {left} != {right}')


def make_checkpoint(root, step, world_size=1):
    model = tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.)
    stochastic_update(model, optimizer, 0)
    fixture_adapter(root, step, model)
    stage = cp.prepare_state(root, step)
    cp.save_torch(stage / 'optimizer.pt', optimizer.state_dict())
    for rank in range(world_size):
        torch.manual_seed(100 + rank)
        cp.save_torch(stage / f'rng_rank_{rank}.pt', cp.capture_rng())
    return cp.commit_state(root, step, world_size, CONTRACT)


def distributed_resume_worker(rank, init_file, directory):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + init_file, rank=rank, world_size=2)
    try:
        root = Path(directory)
        torch.manual_seed(42)
        actor = DistributedDataParallel(tiny_model())
        optimizer = torch.optim.AdamW(actor.parameters(), lr=1e-4, weight_decay=0.)
        random.seed(12 + rank)
        np.random.seed(23 + rank)
        torch.manual_seed(34 + rank)
        stochastic_update(actor, optimizer, 0)
        if rank == 0:
            fixture_adapter(root, 1, actor.module)
        dist.barrier()
        cp.save_distributed_state(root, 1, actor, optimizer, dist, CONTRACT)
        expected_losses = [stochastic_update(actor, optimizer, step) for step in (1, 2, 3)]
        expected_weights = copy.deepcopy(actor.module.state_dict())
        expected_optimizer = copy.deepcopy(optimizer.state_dict())
        expected_rng = cp.capture_rng()

        # New initialization intentionally disturbs all three CPU random streams.
        torch.manual_seed(999)
        random.seed(998)
        np.random.seed(997)
        resumed = DistributedDataParallel(tiny_model())
        resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1e-4, weight_decay=0.)
        checkpoint = cp.inspect_checkpoint(root / 'state_001', CONTRACT, 2)
        resumed.module.load_state_dict(torch.load(Path(checkpoint['adapter_path']) /
               'adapter_model.safetensors', map_location='cpu', weights_only=True))
        rng = cp.load_training_state(checkpoint, resumed_optimizer, rank)
        cp.restore_rng(rng)
        actual_losses = [stochastic_update(resumed, resumed_optimizer, step)
                         for step in range(checkpoint['completed_step'], 4)]
        assert_nested_equal(expected_losses, actual_losses)
        assert_nested_equal(expected_weights, resumed.module.state_dict())
        assert_nested_equal(expected_optimizer, resumed_optimizer.state_dict())
        assert_nested_equal(expected_rng, cp.capture_rng())
        cp.write_json(root / f'result_rank_{rank}.json', dict(exact=True, losses=actual_losses,
                    resumed_indices=list(range(checkpoint['completed_step'], 4))))
    finally:
        dist.destroy_process_group()


class ResumeTests(unittest.TestCase):
    def test_adamw_dropout_and_each_rank_rng_resume_exactly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mp.spawn(distributed_resume_worker, args=(str(root / 'gloo_init'), directory),
                     nprocs=2, join=True)
            results = [json.loads((root / f'result_rank_{rank}.json').read_text()) for rank in range(2)]
            self.assertTrue(all(result['exact'] for result in results))
            self.assertEqual(results[0]['resumed_indices'], [1, 2, 3])
            self.assertNotEqual(results[0]['losses'], results[1]['losses'])

    def test_missing_complete_is_rejected_even_if_data_are_present(self):
        with tempfile.TemporaryDirectory() as directory:
            state = make_checkpoint(Path(directory), 1)
            (state / 'COMPLETE.json').unlink()
            with self.assertRaises(ValueError):
                cp.inspect_checkpoint(state)

    def test_incomplete_rank_state_cannot_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage = cp.prepare_state(root, 1)
            cp.save_torch(stage / 'optimizer.pt', {})
            cp.save_torch(stage / 'rng_rank_0.pt', cp.capture_rng())
            with self.assertRaises(ValueError):
                cp.commit_state(root, 1, 2, CONTRACT)
            self.assertFalse((root / 'state_001' / 'COMPLETE.json').exists())

    def test_corrupt_optimizer_adapter_and_metadata_rejected(self):
        for damaged in ('optimizer.pt', 'metadata.json', 'adapter'):
            with self.subTest(damaged=damaged), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                state = make_checkpoint(root, 1)
                target = root / 'adapter_001' / 'adapter_model.safetensors' if damaged == 'adapter' else state / damaged
                with target.open('ab') as stream:
                    stream.write(b'corruption')
                with self.assertRaises(ValueError):
                    cp.inspect_checkpoint(state)

    def test_changed_training_contract_or_world_size_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            state = make_checkpoint(Path(directory), 1, world_size=2)
            for changed in ({**CONTRACT, 'batch_size':32}, {**CONTRACT, 'arm':'dqo'}):
                with self.assertRaises(ValueError):
                    cp.inspect_checkpoint(state, changed, 2)
            with self.assertRaises(ValueError):
                cp.inspect_checkpoint(state, CONTRACT, 4)

    def test_missing_rank_rng_and_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = make_checkpoint(root, 1, world_size=2)
            rank_file = state / 'rng_rank_1.pt'
            rank_file.unlink()
            with self.assertRaises(ValueError):
                cp.inspect_checkpoint(state)
            rank_file.symlink_to(state / 'rng_rank_0.pt')
            with self.assertRaises(ValueError):
                cp.inspect_checkpoint(state)

    def test_retention_preserves_epochs_latest_two_and_incomplete_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for step in (0, 7, 8, 16, 23, 24, 30, 31, 32):
                make_checkpoint(root, step)
            (root / 'state_033.writing').mkdir()
            (root / 'state_034').mkdir()
            self.assertEqual(cp.prune_complete_pairs(root, 32, {8, 16, 24, 32}), [0, 7, 23, 30])
            expected = [8, 16, 24, 31, 32]
            for step in expected:
                cp.inspect_checkpoint(root / f'state_{step:03d}')
            self.assertTrue((root / 'state_033.writing').is_dir())
            self.assertTrue((root / 'state_034').is_dir())
            self.assertEqual(len(list(root.glob('adapter_*_manifest.json'))), 9)
            self.assertEqual(len(list(root.glob('adapter_[0-9][0-9][0-9]'))), len(expected))

    def test_all_frozen_math_helpers_are_ast_identical(self):
        source = ast.parse((ROOT / 'trainer_distributed.py').read_text())
        derived = ast.parse((ROOT / 'trainer_resumable.py').read_text())
        original = {node.name: ast.dump(node) for node in source.body if isinstance(node, ast.FunctionDef)}
        updated = {node.name: ast.dump(node) for node in derived.body if isinstance(node, ast.FunctionDef)}
        for name in original.keys() - {'main'}:
            self.assertEqual(original[name], updated[name], name)


if __name__ == '__main__':
    unittest.main(verbosity=2)
