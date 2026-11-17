"""Real tiny Qwen3 padding/gradient checks and four-rank DDP update parity."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

import batched_learner
import trainer
import trainer_resumable


def tiny_actor():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(573)
    model = Qwen3ForCausalLM(Qwen3Config(vocab_size=31, hidden_size=16,
        intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=1, head_dim=8, max_position_embeddings=128,
        attention_dropout=0.))
    model.config._attn_implementation = 'sdpa'
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    return model


def fixture_groups(actor, rank):
    groups = []
    # Deliberately unequal rank counts exercise differentiable collective padding.
    for i in range(rank + 1):
        prompt = [1, 2, 3][:1 + (rank + i) % 3]
        response = [4 + (rank + i + j) % 20 for j in range(1 + (rank * 3 + i) % 7)]
        with torch.no_grad():
            old = trainer.selected_logps(actor, prompt, response).tolist()
        advantages = torch.tensor([[(-1 if j % 2 else 1) * .1 * (i + 1) for j in range(len(response))]])
        # Include clipped actor/behavior ratios and a zero-advantage response.
        old = [x + (.3 if j % 3 == 0 else -.4 if j % 3 == 1 else 0) for j, x in enumerate(old)]
        if rank == 1 and i == 0:
            advantages.zero_()
        groups.append(dict(prompt_token_ids=prompt, records=[dict(token_ids=response, logprobs=old)],
                           advantages=advantages))
    return groups


def distributed_worker(rank, init_file, result_dir):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + init_file, rank=rank, world_size=4)
    try:
        initial = tiny_actor()
        groups = fixture_groups(initial, rank)
        results = []
        for size in (1, 2):
            actor = DistributedDataParallel(copy.deepcopy(initial), broadcast_buffers=False)
            optimizer = torch.optim.SGD(actor.parameters(), lr=.03)
            stats = trainer_resumable.update_learner(actor, optimizer, groups, dist=dist,
                                                    learner_microbatch_size=size)
            results.append((stats, [p.detach().clone() for p in actor.parameters()],
                            [p.grad.detach().clone() for p in actor.parameters()]))
            del actor, optimizer
        a, b = results
        error = max(float((x-y).abs().max()) for x, y in zip(a[1], b[1]))
        grad_error = max(float((x-y).abs().max()) for x, y in zip(a[2], b[2]))
        assert error < 1e-6, error
        assert grad_error < 2e-6, grad_error
        assert abs(a[0]['loss'] - b[0]['loss']) < 1e-7
        assert a[0]['response_tokens'] == b[0]['response_tokens']
        assert a[0]['rollout_count'] == b[0]['rollout_count'] == 10
        assert b[0]['collective_padding_forwards'] == 2
        assert abs(a[0]['ratio_clip_fraction'] - b[0]['ratio_clip_fraction']) < 1e-12
        Path(result_dir, f'rank_{rank}.json').write_text(json.dumps(dict(weight_error=error,
            gradient_error=grad_error, loss_error=abs(a[0]['loss'] - b[0]['loss']),
            response_tokens=b[0]['response_tokens'], rollout_count=b[0]['rollout_count'])))
    finally:
        dist.destroy_process_group()


class BatchedLearnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_variable_length_logps_loss_and_gradients_match_unpadded(self):
        # Includes one-token prompt/response, unequal prompt and response lengths,
        # and a large left-padding gap: logits slices must align with responses.
        for sequences in ([([1], [7]), ([2, 3, 4, 5], [8, 9, 10, 11, 12, 13])],
                          [([1, 2, 3, 4, 5, 6], [7, 8]), ([2], [9, 10, 11, 12])]):
            with self.subTest(sequences=sequences):
                single = tiny_actor().train()
                paired = copy.deepcopy(single)
                left = [trainer.selected_logps(single, p, r) for p, r in sequences]
                right = batched_learner.selected_logps(paired, sequences)
                left_losses, right_losses = [], []
                for a, b in zip(left, right):
                    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)
                    old = a.detach() + torch.linspace(-.4, .4, len(a))
                    advantage = torch.linspace(-.8, .9, len(a))
                    left_losses.append(trainer.loss_terms(a, old, advantage)[0])
                    right_losses.append(trainer.loss_terms(b, old, advantage)[0])
                sum(left_losses).backward()
                sum(right_losses).backward()
                torch.testing.assert_close(sum(left_losses), sum(right_losses), atol=1e-6, rtol=1e-6)
                for a, b in zip(single.parameters(), paired.parameters()):
                    torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=1e-5)

    def test_real_four_rank_ddp_single_vs_pair_update(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(distributed_worker, args=(str(Path(directory) / 'gloo'), directory), nprocs=4, join=True)
            records = [json.loads(Path(directory, f'rank_{rank}.json').read_text()) for rank in range(4)]
            self.assertTrue(all(r['gradient_error'] < 2e-6 for r in records))

    def test_length_sort_preserves_all_records_once(self):
        records = [(dict(prompt_token_ids=[1] * i), i, dict(token_ids=[2] * (7-i))) for i in (1, 3, 5, 2, 4)]
        packs = batched_learner.packed_records(records, 2)
        self.assertEqual([len(p) for p in packs], [2, 2, 1])
        self.assertEqual(sorted(id(x) for p in packs for x in p), sorted(map(id, records)))

    def test_invalid_inputs_fail(self):
        actor = tiny_actor()
        for sequences in ([], [([], [1])], [([1], [])]):
            with self.assertRaises(ValueError):
                batched_learner.selected_logps(actor, sequences)
        with self.assertRaises(ValueError):
            batched_learner.packed_records([], 3)

    def test_flag_is_opt_in_lora_only_and_resume_contract_records_packing(self):
        import runtime_config as config
        from tests.test_training_config import arguments
        contracts = []
        for size in (1, 2):
            args = arguments('train', '--finetune-mode', 'lora', '--learner-microbatch-size', str(size))
            config.validate_options(args, resumable=True)
            contracts.append(config.scientific_contract(args, world_size=4, training_questions=2048,
                                                        identity={'content_sha256': 'fixture'}))
            self.assertEqual(contracts[-1]['learner_microbatch_size'], size)
        self.assertNotEqual(contracts[0], contracts[1])
        self.assertEqual(contracts[1]['learner_sequence_packing'], 'length_sorted_left_pad')
        self.assertEqual(arguments().learner_microbatch_size, 1)
        args = arguments('train', '--finetune-mode', 'full', '--learner-microbatch-size', '2')
        with self.assertRaisesRegex(ValueError, 'resumable LoRA'):
            config.validate_options(args, resumable=True)


if __name__ == '__main__':
    unittest.main()
