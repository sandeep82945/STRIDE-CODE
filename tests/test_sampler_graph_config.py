"""CPU-only tests: graph capture concurrency matches the sampler configuration."""
import os
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from sampler_worker import decode_graph_capture_sizes, make_processor, SamplerClient


class DecodeGraphConfigurationTests(unittest.TestCase):
    def test_engine_seed_rejects_invalid_values_before_gpu_or_ipc_access(self):
        for value in (True, -1, 2**32, 1.5, '42', None):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, 'engine_seed'):
                    make_processor('/unused', engine_seed=value)
                with self.assertRaisesRegex(ValueError, 'engine_seed'):
                    SamplerClient.launch('/unused', '/must-not-be-created', 'GPU-fixture', engine_seed=value)

    def test_existing_32_configuration_is_unchanged(self):
        self.assertEqual(decode_graph_capture_sizes(32), [1, 2, 4, 8, 16, 24, 32])

    def test_64_captures_new_large_decode_batches(self):
        sizes = decode_graph_capture_sizes(64)
        self.assertEqual(sizes, [1, 2, 4, 8, 16, 24, 32, 48, 64])
        self.assertEqual(max(sizes), 64)
        self.assertEqual(sizes, sorted(set(sizes)))

    def test_16_does_not_capture_unreachable_shapes(self):
        self.assertEqual(decode_graph_capture_sizes(16), [1, 2, 4, 8, 16])

    def test_invalid_concurrency_rejected(self):
        for value in (True, False, 0, 1, 24, 48, 128, 64., '64', None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                decode_graph_capture_sizes(value)

    def test_processor_passes_64_graphs_and_preserves_memory_budget(self):
        captured = {}
        class FakeLLM:
            def __init__(self, **kwargs):
                captured.update(kwargs)
            def get_tokenizer(self):
                return object()
        modules = {name: ModuleType(name) for name in
                   ('vllm', 'vllm.lora', 'vllm.lora.request', 'vllm.config', 'vllm.config.compilation')}
        modules['vllm'].LLM = FakeLLM
        modules['vllm'].SamplingParams = object
        modules['vllm.lora.request'].LoRARequest = object
        compilation = modules['vllm.config.compilation']
        compilation.CompilationConfig = lambda **kwargs: kwargs
        compilation.CompilationMode = SimpleNamespace(NONE='none')
        compilation.CUDAGraphMode = SimpleNamespace(FULL_DECODE_ONLY='decode')
        with patch.dict(sys.modules, modules), patch.dict(os.environ, CUDA_VISIBLE_DEVICES='GPU-allocated'):
            make_processor('/fixture/model', max_num_seqs=64, engine_seed=43, finetune_mode='lora')
        self.assertEqual(captured['seed'], 43)
        self.assertTrue(captured['enable_lora'])
        self.assertEqual(captured['max_num_seqs'], 64)
        self.assertEqual(captured['gpu_memory_utilization'], .8)
        self.assertEqual(captured['max_model_len'], 8192)
        self.assertEqual(captured['max_num_batched_tokens'], 8192)
        self.assertEqual(captured['compilation_config']['max_cudagraph_capture_size'], 64)
        self.assertEqual(captured['compilation_config']['cudagraph_capture_sizes'], decode_graph_capture_sizes(64))


if __name__ == '__main__':
    unittest.main()
