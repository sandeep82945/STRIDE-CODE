"""CPU contract tests and a real two-process gloo DDP/serial gradient comparison."""
from __future__ import annotations

import json
import importlib
import os
from datetime import timedelta
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

import trainer_distributed as module
from sampler_worker import SamplerClient


class ScalarPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([.07, -.02]))

    def forward(self, token_ids):
        x = token_ids.float() / 10.
        return self.weight[0] * x + self.weight[1]


def group_for_rank(rank):
    ids = ([1], [2, 3, 4]) if rank == 0 else ([5, 6, 7, 8, 9], [10, 11])
    # Unequal token counts and active positive/negative PPO clipping ensure
    # this test catches averaging rank means or response means accidentally.
    old = ([0.], [.4, 0., -.4]) if rank == 0 else ([.4, -.4, 0., .2, -.2], [.1, 0.])
    advantage = ([.2, 100., 100., 100., 100.], [-.1, .3, -.4, 100., 100.]) if rank == 0 else (
        [.4, -.3, .2, -.1, .5], [.1, -.5, 100., 100., 100.])
    return [dict(prompt_token_ids=[7], records=[dict(token_ids=list(t), logprobs=list(p)) for t, p in zip(ids, old)],
                 advantages=torch.tensor(advantage))]


def fake_logps(actor, prompt, response):
    return actor(torch.tensor(response, dtype=torch.float32))


def ddp_worker(rank, init_file, out_file):
    dist.init_process_group('gloo', init_method='file://' + init_file, rank=rank, world_size=2)
    try:
        actor = DistributedDataParallel(ScalarPolicy())
        optimizer = torch.optim.SGD(actor.parameters(), lr=.3)
        module.pilot.selected_logps = fake_logps
        stats = module.update_distributed(actor, optimizer, group_for_rank(rank), dist=dist)
        gathered = [None, None]
        dist.all_gather_object(gathered, actor.module.weight.detach().tolist())
        if rank == 0:
            Path(out_file).write_text(json.dumps(dict(parameters=gathered, stats=stats)))
    finally:
        dist.destroy_process_group()


class ScheduleTests(unittest.TestCase):
    def test_epoch_coverage_and_rank_partition(self):
        rows = [dict(id=i) for i in range(512)]
        first = []
        for epoch in range(4):
            seen = []
            for within in range(16):
                step = epoch * 16 + within
                batch = module.epoch_batch(rows, step, 32, seed=42, epoch_offset=1)
                shards = [module.rank_batch(batch, rank, 4) for rank in range(4)]
                self.assertEqual([len(shard) for shard in shards], [8] * 4)
                self.assertEqual(sum(shards, []), batch)
                seen.extend(row['id'] for row in batch)
            self.assertEqual(sorted(seen), list(range(512)))
            if epoch == 0:
                first = seen
            else:
                self.assertNotEqual(seen, first)
        self.assertEqual(module.epoch_batch(rows, 29, 32), module.epoch_batch(rows, 29, 32))

    def test_too_short_tail_rejected_without_silent_drop(self):
        with self.assertRaises(ValueError):
            module.epoch_batch(list(range(33)), 0, 32)
        with self.assertRaises(ValueError):
            module.rank_batch(list(range(3)), 0, 4)

    def test_allocation_uses_actual_slurm_ids(self):
        visible = 'GPU-a,GPU-c,GPU-e,GPU-g,GPU-b,GPU-d,GPU-f,GPU-h'
        self.assertEqual(module.allocation(visible, 2, 4), ('GPU-e', 'GPU-f'))
        self.assertEqual(module.allocation('3,4,5,6,8,9,10,11', 0, 4), ('3', '8'))
        for bad in ('', '0,1,2,3', '0,1,2,3,4,5,6,6'):
            with self.assertRaises(ValueError):
                module.allocation(bad, 0, 4)

    def test_validate_rows_checks_duplicate_identity(self):
        good = [dict(id=i, question=f'x {i}', ground_truth='1') for i in range(4)]
        module.validate_rows(good)
        with self.assertRaises(ValueError):
            module.validate_rows(good + [good[0]])


class PruningTests(unittest.TestCase):
    def make_checkpoint(self, root, step):
        path = root / f'adapter_{step:03d}'
        path.mkdir()
        (path / 'adapter_model.safetensors').write_bytes(b'fixture')
        path.with_name(path.name + '_manifest.json').write_text('{}')
        return path

    def test_epoch_checkpoints_and_latest_two_survive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for step in (0, 15, 16, 30, 31, 32):
                self.make_checkpoint(root, step)
            module.prune_synchronization_adapters(root, 32, {16, 32, 48, 64}, finetune_mode='lora')
            self.assertEqual(sorted(p.name for p in root.iterdir() if p.is_dir()),
                             ['adapter_016', 'adapter_031', 'adapter_032'])
            self.assertEqual(len(list(root.glob('*_manifest.json'))), 6)

    def test_symlink_and_unmanifested_dir_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'adapter_000').mkdir()
            with self.assertRaises(ValueError):
                module.prune_synchronization_adapters(root, 3, set(), finetune_mode='lora')
            (root / 'adapter_000').rmdir()
            target = root / 'target'
            target.mkdir()
            (root / 'adapter_000').symlink_to(target, target_is_directory=True)
            with self.assertRaises(ValueError):
                module.prune_synchronization_adapters(root, 3, set(), finetune_mode='lora')


class SamplerIsolationTests(unittest.TestCase):
    def test_sampler_removes_parent_distributed_environment(self):
        captured = {}
        class Process:
            def __init__(self, command, **kwargs):
                captured.update(command=command, **kwargs)
            def poll(self):
                return None
        parent = {'RANK': '2', 'LOCAL_RANK': '2', 'WORLD_SIZE': '4', 'MASTER_PORT': '12345',
                  'MASTER_ADDR': 'localhost', 'TORCHELASTIC_RUN_ID': 'parent',
                  'TORCHELASTIC_USE_AGENT_STORE': 'True', 'TORCHELASTIC_FUTURE_FLAG': 'parent',
                  'CUDA_VISIBLE_DEVICES': '1,2,3,4,5,6,7,8'}
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, parent), \
                patch('sampler_worker.subprocess.Popen', Process), \
                patch.object(SamplerClient, 'wait_file', return_value={'ok': True}):
            SamplerClient.launch('/fake/model', Path(directory) / 'ipc', sampler_device='GPU-allocated')
        self.assertEqual(captured['env']['CUDA_VISIBLE_DEVICES'], 'GPU-allocated')
        for key in parent:
            if key != 'CUDA_VISIBLE_DEVICES':
                self.assertNotIn(key, captured['env'])
        # Exercise this environment against the installed PyTorch store path.
        # Rank zero must HOST vLLM's private store, not wait as an orphan client.
        rendezvous = importlib.import_module('torch.distributed.rendezvous')
        with patch.dict(os.environ, parent, clear=True), patch.object(rendezvous, 'TCPStore') as before:
            rendezvous._create_c10d_store('127.0.0.1', 12345, 0, 1, timedelta(seconds=1))
            self.assertFalse(before.call_args.kwargs['is_master'])
        with patch.dict(os.environ, captured['env'], clear=True), patch.object(rendezvous, 'TCPStore') as after:
            rendezvous._create_c10d_store('127.0.0.1', 12345, 0, 1, timedelta(seconds=1))
            self.assertTrue(after.call_args.kwargs['is_master'])


class DiagnosticTests(unittest.TestCase):
    def test_accuracy_only_grpo_reports_available_group_activation(self):
        groups = [dict(records=[dict(token_ids=[1, 2], finish_reason='stop',
                                      score=dict(answer_correct=True, step_eligible=True)) for _ in range(2)],
                       diagnostics=dict(correct_responses=2, eligible_responses=2),
                       advantages=torch.zeros(2, 2))]
        result = module.diagnostics(groups, 'grpo')
        self.assertEqual(result['gate_active'], 1)
        self.assertEqual(result['correct'], 2)
        self.assertEqual(result['nonzero_advantage_tokens'], 0)
        self.assertEqual(result['aux_mass'], 0)
        metrics = module.metrics({}, result, epoch=.0625)
        self.assertEqual(metrics['credit/activation_fraction'], 1.)
        self.assertEqual(metrics['rollouts/mean_response_tokens'], 2.)


class DistributedParityTests(unittest.TestCase):
    def test_global_real_token_loss_and_update_equal_serial(self):
        actor = ScalarPolicy()
        optimizer = torch.optim.SGD(actor.parameters(), lr=.3)
        all_groups = group_for_rank(0) + group_for_rank(1)
        denominator = sum(len(r['token_ids']) for g in all_groups for r in g['records'])
        loss_sum = 0.
        for group in all_groups:
            for index, record in enumerate(group['records']):
                new = fake_logps(actor, group['prompt_token_ids'], record['token_ids'])
                old = torch.tensor(record['logprobs'])
                advantage = group['advantages'][index, :len(old)]
                loss, _, _ = module.pilot.loss_terms(new, old, advantage)
                (loss / denominator).backward()
                loss_sum += float(loss.detach())
        norm = float(torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.))
        optimizer.step()
        with tempfile.TemporaryDirectory() as directory:
            rendezvous = str(Path(directory) / 'rendezvous')
            result_file = str(Path(directory) / 'result.json')
            mp.spawn(ddp_worker, args=(rendezvous, result_file), nprocs=2, join=True)
            result = json.loads(Path(result_file).read_text())
        for weights in result['parameters']:
            self.assertTrue(torch.allclose(torch.tensor(weights), actor.weight.detach(), atol=1e-7, rtol=0))
        self.assertEqual(result['stats']['response_tokens'], 11)
        self.assertEqual(result['stats']['rollout_count'], 4)
        self.assertAlmostEqual(result['stats']['loss'], loss_sum / denominator, places=7)
        self.assertAlmostEqual(result['stats']['gradient_norm'], norm, places=7)


if __name__ == '__main__':
    unittest.main()
