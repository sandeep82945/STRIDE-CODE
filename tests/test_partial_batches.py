"""Exact sample coverage and real four-rank uneven-batch gradient parity."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

import runtime_config as config
import trainer
import trainer_distributed as distributed
import trainer_resumable as resumable
from tests.test_distributed import ScalarPolicy
from tests.test_seed_comparability import ReversibleTokenizer, arguments


def fixture_groups(rank):
    """The requested final batch: 22 questions, eight responses per question."""
    rows = list(range(22))
    groups = []
    for identity in distributed.rank_batch(rows, rank, 4):
        records, advantages = [], []
        for sample in range(8):
            width = 1 + (identity + sample) % 4
            records.append(dict(token_ids=list(range(1, width + 1)),
                                logprobs=[.1 * ((identity + sample) % 3 - 1)] * width))
            advantages.append([.07 * ((identity + sample) % 5 - 2)] * 4)
        groups.append(dict(id=identity, prompt_token_ids=[7], records=records,
                           advantages=torch.tensor(advantages)))
    return groups


def fake_logps(actor, prompt, response):
    return actor(torch.tensor(response, dtype=torch.float32))


def unequal_worker(rank, rendezvous, result, force_signal):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=4)
    try:
        actor = DistributedDataParallel(ScalarPolicy())
        optimizer = torch.optim.SGD(actor.parameters(), lr=.3)
        calls = []

        def counted_logps(*args):
            calls.append(1)
            return fake_logps(*args)

        with patch.object(distributed.pilot, 'selected_logps', side_effect=counted_logps):
            stats = distributed.update_distributed(actor, optimizer, fixture_groups(rank),
                                                  dist=dist, force_signal=force_signal)
        gathered = [None] * 4
        dist.all_gather_object(gathered, dict(weights=actor.module.weight.detach().tolist(),
                                            calls=len(calls)))
        if rank == 0:
            Path(result).write_text(json.dumps(dict(ranks=gathered, stats=stats)))
    finally:
        dist.destroy_process_group()


class PartialBatchTests(unittest.TestCase):
    def test_150_questions_once_per_epoch_and_shared_arm_seeds(self):
        rows = [dict(id=i, question=f'Calculate {i} plus seven.', ground_truth=i+7)
                for i in range(150)]
        tok = ReversibleTokenizer()
        histories, contracts = [], []
        for driver in (distributed, resumable):
            for arm in config.ARMS:
                args = arguments(arm, 42)
                self.assertEqual(config.schedule_steps(args, len(rows)), 12)
                with patch('trainer_resumable.version', return_value='fixture'):
                    contract = resumable.training_contract(args, 4, len(rows), {'content_sha256': 'same'})
                self.assertEqual(contract['version'], 7)
                self.assertEqual(contract['updates_per_epoch'], 3)
                comparable = copy.deepcopy(contract)
                comparable.pop('arm')
                contracts.append(comparable)
                history = []
                for epoch in range(4):
                    seen = []
                    for within, size in enumerate((64, 64, 22)):
                        step = 3 * epoch + within
                        batch = driver.epoch_batch(rows, step, 64, seed=42)
                        self.assertEqual(len(batch), size)
                        shards = [driver.rank_batch(batch, rank, 4) for rank in range(4)]
                        self.assertEqual([len(x) for x in shards], [16]*4 if size == 64 else [6, 6, 5, 5])
                        self.assertEqual(sum(shards, []), batch)
                        base = 20260980 + 42 * 1000000 + step * 64
                        global_prompts = trainer.request(tok, batch, 'global', None, 0, base)['prompts']
                        local_prompts = []
                        for rank, shard in enumerate(shards):
                            offset = driver.rank_offset(len(batch), rank, 4)
                            local_prompts.extend(trainer.request(tok, shard, 'rank', None, 0,
                                                                 base + offset)['prompts'])
                        self.assertEqual(local_prompts, global_prompts)
                        seen.extend(row['id'] for row in batch)
                        history.extend((row['id'], row['seed']) for row in global_prompts)
                    self.assertEqual(sorted(seen), list(range(150)))
                    self.assertEqual(config.epoch_progress((epoch+1)*3, 150, 64), epoch+1)
                self.assertEqual(len(history), 600)
                self.assertEqual(len({seed for _, seed in history}), 600)
                histories.append(history)
        self.assertTrue(all(item == histories[0] for item in histories))
        self.assertTrue(all(item == contracts[0] for item in contracts))
        self.assertEqual(config.epoch_progress(1, 150, 64), 64/150)
        self.assertEqual(config.epoch_progress(2, 150, 64), 128/150)

    def test_reference_and_credit_source_hashes_remain_original(self):
        root = Path(__file__).resolve().parents[1]
        original = json.loads((root/'RELEASE_MANIFEST.json').read_text())['source_files']
        for name in ('credit.py', 'production_stride.py', 'experiment_credit.py', 'feature_pooling.py'):
            relative = 'training/' + name
            self.assertEqual(hashlib.sha256((root/relative).read_bytes()).hexdigest(), original[relative])

    def test_unequal_rank_loss_gradient_and_counts_equal_serial(self):
        for force_signal in (False, True):
            with self.subTest(force_signal=force_signal):
                actor = ScalarPolicy()
                optimizer = torch.optim.SGD(actor.parameters(), lr=.3)
                groups = sum((fixture_groups(rank) for rank in range(4)), [])
                denominator = sum(len(record['token_ids']) for group in groups for record in group['records'])
                loss_sum = 0.
                for group in groups:
                    for index, record in enumerate(group['records']):
                        new = fake_logps(actor, group['prompt_token_ids'], record['token_ids'])
                        old = torch.tensor(record['logprobs'])
                        advantage = group['advantages'][index, :len(old)]
                        if force_signal:
                            advantage = torch.ones_like(advantage)
                        loss, _, _ = trainer.loss_terms(new, old, advantage)
                        (loss / denominator).backward()
                        loss_sum += float(loss.detach())
                norm = float(torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.))
                optimizer.step()
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp)
                    mp.spawn(unequal_worker, args=(str(path/'rendezvous'), str(path/'result.json'), force_signal),
                             nprocs=4, join=True)
                    actual = json.loads((path/'result.json').read_text())
                self.assertEqual([rank['calls'] for rank in actual['ranks']], [48] * 4)
                for rank in actual['ranks']:
                    torch.testing.assert_close(torch.tensor(rank['weights']), actor.weight.detach(),
                                               atol=1e-7, rtol=0)
                self.assertEqual(actual['stats']['rollout_count'], 176)
                self.assertEqual(actual['stats']['collective_padding_forwards'], 16)
                self.assertEqual(actual['stats']['response_tokens'], denominator)
                self.assertAlmostEqual(actual['stats']['loss'], loss_sum / denominator, places=7)
                self.assertAlmostEqual(actual['stats']['gradient_norm'], norm, places=6)


if __name__ == '__main__':
    unittest.main()
