"""Real tiny random Qwen3 + PEFT integration on CPU, not a pretrained-model eval."""
import contextlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

import trainer
import trainer_distributed as distributed
import trainer_resumable as resumable
from scoring import annotate

ML_AVAILABLE = all(importlib.util.find_spec(n) for n in ('transformers', 'peft'))


class AsciiTokenizer:
    """Synthetic reversible ByteLevel vocabulary; actual tokenizer tests are separate."""
    eos_token = None
    all_special_ids = []
    def __call__(self, text, **kwargs):
        return {'input_ids': self.encode(text)}
    def encode(self, text, **kwargs):
        return list(text.encode('ascii'))
    def decode(self, ids, **kwargs):
        return bytes(ids).decode('ascii')
    def convert_ids_to_tokens(self, ids):
        from native_step_parser import _byte_decoder
        encoder = {value: key for key, value in _byte_decoder().items()}
        return encoder[ids] if isinstance(ids, int) else [encoder[i] for i in ids]
    def get_added_vocab(self):
        return {}
    def apply_chat_template(self, messages, **kwargs):
        return messages[0]['content']


@unittest.skipUnless(ML_AVAILABLE, 'Install transformers and peft for real CPU actor integration')
class ActorCPUIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import Qwen3Config, Qwen3ForCausalLM
        cls.folder = tempfile.TemporaryDirectory(prefix='stride-tiny-qwen-')
        cls.model_path = Path(cls.folder.name)/'model'
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        torch.manual_seed(19)
        config = Qwen3Config(vocab_size=128, hidden_size=32, intermediate_size=64,
                            num_hidden_layers=2, num_attention_heads=4,
                            num_key_value_heads=2, head_dim=8, max_position_embeddings=1024)
        Qwen3ForCausalLM(config).save_pretrained(cls.model_path)
        cls.tok = AsciiTokenizer()

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()
        torch.set_num_threads(cls.threads)

    def actor(self, adapter=None):
        torch.manual_seed(23)
        actor = trainer.load_actor(str(self.model_path), adapter, device='cpu', finetune_mode='lora')
        actor.eval()
        return actor

    def example(self, actor, arm='local_positive'):
        row = dict(id='q', question='What is 1 + 1?', ground_truth=2)
        prompt = trainer.render(self.tok, row)
        texts = [r'Step 1: Add 1 and 1.'+'\n'+r'Step 2: The sum is 2.'+'\n'+r'Final Answer: \boxed{2}',
                 r'Step 1: Double one.'+'\n'+r'Step 2: Doubling gives 2.'+'\n'+r'Final Answer: \boxed{2}',
                 r'Step 1: Add the numbers.'+'\n'+r'Step 2: Guess 3.'+'\n'+r'Final Answer: \boxed{3}']
        samples = []
        for index, text in enumerate(texts):
            ids = self.tok.encode(text)
            with torch.no_grad(): old = trainer.selected_logps(actor, prompt, ids).tolist()
            samples.append(dict(sample_index=index, token_ids=ids, logprobs=old,
                                text=text, finish_reason='stop'))
        generated = dict(rows=[dict(id='q', prompt_token_ids=prompt, max_tokens=8192-len(prompt), samples=samples)])
        groups = trainer.annotate_groups(self.tok, [row], generated,
                                        actor=actor if arm != 'grpo' else None, arm=arm)
        return groups

    def test_actor_initialization_logps_and_reference_features(self):
        actor = self.actor()
        trainable = [(n, p) for n, p in actor.named_parameters() if p.requires_grad]
        self.assertTrue(trainable)
        self.assertTrue(all('lora_' in n and p.dtype == torch.float32 for n, p in trainable))
        prompt, response = [1, 4, 3], [7, 8, 9, 10]
        with torch.no_grad():
            actual = trainer.selected_logps(actor, prompt, response)
            logits = actor(input_ids=torch.tensor([prompt+response[:-1]]), use_cache=False).logits[0]
            expected = logits[len(prompt)-1:].float().log_softmax(-1).gather(1, torch.tensor(response)[:, None])[:, 0]
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=0))
        spans = [(0, 0), (2, 3)]
        before = trainer.features(actor, prompt, response, spans, include_pooled=True)
        with torch.no_grad():
            for name, p in trainable:
                if 'lora_B' in name: p.add_(.25)
        after = trainer.features(actor, prompt, response, spans, include_pooled=True)
        for x, y in zip(before, after): self.assertTrue(torch.equal(x, y))
        with torch.no_grad(): self.assertFalse(torch.equal(actual, trainer.selected_logps(actor, prompt, response)))

    def test_each_arm_real_forward_backward_save_reload(self):
        for arm in ('local_positive', 'single_step', 'token_uniform', 'grpo'):
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as folder:
                actor = self.actor()
                groups = self.example(actor, arm)
                self.assertEqual([r['score']['answer_correct'] for r in groups[0]['records']], [True, True, False])
                self.assertTrue(all(r['score']['step_eligible'] for r in groups[0]['records']))
                if arm != 'grpo': self.assertTrue(groups[0]['diagnostics']['gate_active'])
                before = distributed.trainable_fingerprint(actor)
                optimizer = torch.optim.AdamW([p for p in actor.parameters() if p.requires_grad], lr=.01)
                stats = trainer.update(actor, optimizer, groups)
                self.assertGreater(stats['gradient_norm'], 0)
                self.assertNotEqual(distributed.trainable_fingerprint(actor), before)
                self.assertEqual(stats['peak_gpu_bytes'], 0)
                self.assertFalse(actor.training)
                adapter = trainer.save_adapter(actor, Path(folder)/'adapter_001')
                with self.assertRaises(AssertionError): trainer.save_adapter(actor, adapter)
                restored = self.actor(adapter)
                ids = groups[0]['records'][0]['token_ids']
                with torch.no_grad():
                    a = trainer.selected_logps(actor, groups[0]['prompt_token_ids'], ids)
                    b = trainer.selected_logps(restored, groups[0]['prompt_token_ids'], ids)
                self.assertTrue(torch.equal(a, b))
                serialized = trainer.serialize_groups(groups)
                json.dumps(serialized, allow_nan=False)
                self.assertEqual(serialized[0]['advantages'], groups[0]['advantages'].tolist())

    def test_adapter_reload_honors_explicit_cpu_without_auto_selecting_an_accelerator(self):
        import peft.peft_model as peft_model
        actor = self.actor()
        with torch.no_grad():
            for name, parameter in actor.named_parameters():
                if 'lora_B' in name:
                    parameter.add_(.125)
        expected = {name: parameter.detach().clone() for name, parameter in actor.named_parameters()
                    if parameter.requires_grad}
        with tempfile.TemporaryDirectory() as folder:
            adapter = trainer.save_adapter(actor, Path(folder) / 'adapter_001')
            for device in ('cpu', torch.device('cpu')):
                with self.subTest(device=str(device)), \
                     patch.object(peft_model, 'infer_device', side_effect=AssertionError(
                         'An explicit actor device must never auto-select an accelerator')) as infer, \
                     patch.object(peft_model, 'load_peft_weights', wraps=peft_model.load_peft_weights) as load:
                    restored = trainer.load_actor(str(self.model_path), adapter,
                                                   device=device, finetune_mode='lora')
                infer.assert_not_called()
                self.assertEqual(load.call_args.kwargs['device'], 'cpu')
                self.assertTrue(all(parameter.device.type == 'cpu' for parameter in restored.parameters()))
                actual = {name: parameter for name, parameter in restored.named_parameters()
                          if parameter.requires_grad}
                self.assertEqual(actual.keys(), expected.keys())
                for name in expected:
                    self.assertTrue(torch.equal(actual[name], expected[name]), name)

    def test_both_distributed_update_functions_with_real_qwen_gloo(self):
        for module in (distributed, resumable):
            with self.subTest(module=module.__name__), tempfile.TemporaryDirectory() as folder:
                dist.init_process_group('gloo', init_method='file://'+str(Path(folder)/'init'), rank=0, world_size=1)
                try:
                    actor = self.actor()
                    groups = self.example(actor)
                    ddp = DistributedDataParallel(actor, find_unused_parameters=False)
                    optimizer = torch.optim.AdamW([p for p in ddp.parameters() if p.requires_grad], lr=.01)
                    before = module.trainable_fingerprint(actor)
                    stats = module.update_distributed(ddp, optimizer, groups, dist=dist)
                    self.assertGreater(stats['gradient_norm'], 0)
                    self.assertNotEqual(before, module.trainable_fingerprint(actor))
                    adapter = module.save_synchronized(ddp, folder, 1, dist, finetune_mode='lora')
                    self.assertTrue(Path(adapter, 'adapter_model.safetensors').is_file())
                    totals = module.diagnostics(groups, 'local_positive')
                    self.assertEqual(totals['correct'], 2)
                    self.assertEqual(module.metrics(stats, totals, epoch=1)['credit/activation_fraction'], 1)
                finally:
                    dist.destroy_process_group()

    def test_legacy_schedule_request_and_pruning_helpers(self):
        rows = [dict(id=i, question=f'Question {i}') for i in range(8)]
        self.assertEqual(sorted(r['id'] for i in range(2) for r in trainer.schedule(rows, i)), list(range(8)))
        request = trainer.request(self.tok, rows[:2], 'request', None, 0, 12, n=4)
        self.assertEqual([p['seed'] for p in request['prompts']], [12, 13])
        self.assertEqual(request['context_limit'], 8192)
        with tempfile.TemporaryDirectory() as folder:
            for i in range(4): Path(folder,f'adapter_{i:03d}').mkdir()
            trainer.prune_adapters(folder, 3)
            self.assertEqual(sorted(p.name for p in Path(folder).iterdir()), ['adapter_002','adapter_003'])


if __name__ == '__main__':
    unittest.main()
