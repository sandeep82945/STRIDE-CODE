"""Cross-arm seed contracts, rank partitioning, and tiny CPU initialization.

Identical seeds do not promise identical GPU arithmetic or identical trajectories
after the methods update their policies. No pretrained weights are downloaded.
"""
import argparse
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

import torch

import runtime_config as config
import trainer
import trainer_distributed
import trainer_resumable


def arguments(arm, seed):
    parser = argparse.ArgumentParser()
    config.add_training_arguments(parser, resumable=True)
    args = parser.parse_args(['train', '--model-path', '/fixture/model', '--data', '/fixture/data',
                              '--data-sha256', 'a'*64, '--out', f'/fixture/{arm}',
                              '--arm', arm, '--seed', str(seed)])
    config.validate_options(args, resumable=True)
    return args


class ReversibleTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return messages[0]['content']
    def __call__(self, text, **kwargs):
        return {'input_ids': list(text.encode('utf-8'))}
    def decode(self, ids, **kwargs):
        return bytes(ids).decode('utf-8')


class SeedComparabilityTests(unittest.TestCase):
    def test_all_arm_contracts_share_seed_and_sampling_settings(self):
        contracts = []
        for arm in config.ARMS:
            args = arguments(arm, 43)
            contract = config.scientific_contract(args, world_size=4,
                training_questions=128, identity={'content_sha256': 'same-base'})
            self.assertEqual(contract['seed'], 43)
            self.assertEqual(contract['sampler_engine_seed'], 43)
            self.assertEqual(contract['python_hash_seed'], 43)
            self.assertEqual(contract['data_sha256'], 'a'*64)
            self.assertEqual(contract['arm'], arm)
            comparable = copy.deepcopy(contract); comparable.pop('arm')
            contracts.append(comparable)
        self.assertTrue(all(contract == contracts[0] for contract in contracts))
        changed = config.scientific_contract(arguments('grpo', 44), world_size=4,
            training_questions=128, identity={'content_sha256': 'same-base'})
        self.assertNotEqual(changed['seed'], contracts[0]['seed'])
        self.assertEqual(changed['data_sha256'], contracts[0]['data_sha256'])

    def test_global_prompt_seeds_and_order_are_arm_and_rank_partition_independent(self):
        rows = [dict(id=i, question=f'Calculate {i} plus 7.', ground_truth=i+7) for i in range(128)]
        tok = ReversibleTokenizer()
        histories = {}
        for driver in (trainer_distributed, trainer_resumable):
            for experiment_seed in (42, 43):
                for arm in config.ARMS:
                    args = arguments(arm, experiment_seed)
                    contract = config.scientific_contract(args, world_size=4,
                        training_questions=len(rows), identity={'content_sha256': 'same-base'})
                    history = []
                    for step in range(4):
                        batch = driver.epoch_batch(rows, step, args.batch_size, seed=args.seed)
                        base = (contract['sampler_seed_base'] + contract['sampler_seed_stride']*args.seed
                                + step*args.batch_size)
                        global_prompts = trainer.request(tok, batch, 'whole', None, 0, base)['prompts']
                        partitioned = []
                        for rank in range(4):
                            local = driver.rank_batch(batch, rank, 4)
                            partitioned.extend(trainer.request(tok, local, f'rank-{rank}', None, 0,
                                               base + rank*(args.batch_size//4))['prompts'])
                        self.assertEqual(partitioned, global_prompts)
                        self.assertEqual(len({p['seed'] for p in global_prompts}), args.batch_size)
                        history.extend((p['id'], p['seed'], p['prompt_token_ids']) for p in global_prompts)
                    key = (driver.__name__, experiment_seed)
                    if key in histories: self.assertEqual(history, histories[key])
                    else: histories[key] = history
                first_epoch = histories[(driver.__name__, experiment_seed)][:128]
                self.assertEqual(sorted(item[0] for item in first_epoch), list(range(128)))
            self.assertNotEqual(histories[(driver.__name__, 42)], histories[(driver.__name__, 43)])
            self.assertNotEqual([x[0] for x in histories[(driver.__name__, 42)]],
                                [x[0] for x in histories[(driver.__name__, 43)]])
        for seed in (42, 43):
            self.assertEqual(histories[('trainer_distributed', seed)], histories[('trainer_resumable', seed)])

    @unittest.skipUnless(all(importlib.util.find_spec(name) for name in ('transformers', 'peft')),
                         'Install transformers and peft for tiny CPU LoRA initialization')
    def test_same_seed_recreates_identical_lora_initialization_for_all_arms(self):
        from transformers import Qwen3Config, Qwen3ForCausalLM
        with torch.random.fork_rng(devices=[]), tempfile.TemporaryDirectory() as tmp:
            torch.manual_seed(9)
            model = Qwen3ForCausalLM(Qwen3Config(vocab_size=64, hidden_size=16, intermediate_size=24,
                num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                head_dim=8, max_position_embeddings=128))
            model.save_pretrained(Path(tmp)/'model')
            reference = None
            for arm in config.ARMS:
                args = arguments(arm, 43)
                torch.manual_seed(args.seed)
                actor = trainer.load_actor(str(Path(tmp)/'model'), device='cpu', finetune_mode='lora')
                actual = {name: p.detach().clone() for name, p in actor.named_parameters() if p.requires_grad}
                self.assertTrue(actual)
                self.assertTrue(all('lora_' in name for name in actual))
                if reference is None: reference = actual
                else:
                    self.assertEqual(set(actual), set(reference))
                    self.assertTrue(all(torch.equal(actual[name], reference[name]) for name in actual))
            torch.manual_seed(44)
            different = trainer.load_actor(str(Path(tmp)/'model'), device='cpu', finetune_mode='lora')
            self.assertTrue(any(not torch.equal(p, reference[name]) for name, p in different.named_parameters()
                                if p.requires_grad))


if __name__ == '__main__':
    unittest.main()
