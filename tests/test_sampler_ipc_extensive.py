"""CPU sampler contract tests; fake vLLM and real, bounded file-IPC processes.

No test loads a model, imports real vLLM, or initializes a GPU.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import sampler_worker as sw


def request(**changes):
    value = dict(request_id='req_1', prompts=[dict(id='q', prompt_token_ids=[1, 2], seed=42)],
                 n=2, max_tokens=4)
    value.update(changes)
    return value


class FakeTokenizer:
    def decode(self, ids, **kwargs):
        assert kwargs == dict(skip_special_tokens=False, clean_up_tokenization_spaces=False)
        return '|'.join(map(str, ids))


def completion(index=0, ids=None):
    ids = [3, 4] if ids is None else ids
    return NS(index=index, token_ids=ids, logprobs=[{token: NS(logprob=-.25)} for token in ids],
              text='backend text', finish_reason='stop', stop_reason=None)


class FakeLLM:
    def __init__(self):
        self.calls = []
        self.llm_engine = NS(remove_lora=Mock(return_value=True))
        self.mutate = None

    def generate(self, prompts, params, **kwargs):
        self.calls.append((prompts, params, kwargs))
        result = [NS(prompt_token_ids=list(p['prompt_token_ids']),
                     outputs=[completion(i) for i in reversed(range(param.n))])
                  for p, param in zip(prompts, params)]
        if self.mutate:
            self.mutate(result)
        return result


def processor(context_limit=8192):
    return sw.SamplerProcessor(FakeLLM(), FakeTokenizer(), lambda **kw: NS(**kw),
                               lambda *args: args, context_limit=context_limit, finetune_mode='lora')


def adapter(root, name='adapter', **changes):
    path = root / name
    path.mkdir()
    config = dict(peft_type='LORA', r=16, lora_alpha=32, lora_dropout=0)
    config.update(changes)
    sw.atomic_json(path / 'adapter_config.json', config)
    (path / 'adapter_model.safetensors').write_bytes(b'fake weights; no GPU or safetensors reader')
    return path


class RequestValidationTests(unittest.TestCase):
    def test_defaults_and_context_budget_are_per_prompt(self):
        value = request(prompts=[dict(id=1, prompt_token_ids=[1], seed=0),
                                 dict(id='1', prompt_token_ids=[1] * 8191, seed=2**63 - 1)])
        value.pop('n'); value.pop('max_tokens')
        before = copy.deepcopy(value)
        actual = sw.validate_request(value)
        self.assertEqual([p['max_tokens'] for p in actual['prompts']], [8191, 1])
        self.assertEqual(actual['n'], 8)
        self.assertEqual(actual['temperature'], 1.)
        self.assertEqual(actual['adapter_id'], 0)
        self.assertEqual(value, before)

    def test_explicit_small_context_and_completion_cap(self):
        actual = sw.validate_request(request(max_tokens=1, context_limit=8), context_limit=8)
        self.assertEqual(actual['prompts'][0]['max_tokens'], 1)
        self.assertEqual(sw.validate_request(request(adapter_id=None))['adapter_id'], 0)

    def test_invalid_top_level_fields(self):
        for bad in (None, [], 3, {'answer': 'secret'}, request(answer='secret'), request(correct=True)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                sw.validate_request(bad)

    def test_integer_boundaries(self):
        cases = {'n': [0, 65, True, 2., '2', None],
                 'max_tokens': [0, 8193, False, 2., '2', None],
                 'context_limit': [0, 8191, 8193, 8192., True, None],
                 'adapter_id': [True, False, 0., 1, -1, '0']}
        for key, values in cases.items():
            for bad in values:
                with self.subTest(key=key, bad=bad), self.assertRaises(ValueError):
                    sw.validate_request(request(**{key: bad}))
        self.assertEqual(sw.require_int(64, 'n', 1, 64), 64)
        for bad in (True, 1., '1', -1):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                sw.require_int(bad, 'token', 0)

    def test_request_id_is_safe_for_filesystem(self):
        for bad in ('', '../escape', 'a/b', 'a\\b', 'a.b', 'a' * 129, 'a\n', 1, None):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                sw.validate_request(request(request_id=bad))
        self.assertEqual(sw.validate_request(request(request_id='A_9-x'))['request_id'], 'A_9-x')

    def test_temperatures(self):
        for bad in (0, -1, float('inf'), float('-inf'), float('nan'), True, '1', None):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                sw.validate_request(request(temperature=bad))
        self.assertEqual(sw.validate_request(request(temperature=.25))['temperature'], .25)

    def test_prompt_schema_identity_tokens_and_seeds(self):
        valid = request()['prompts'][0]
        cases = [None, [], {}, [None], [{**valid, 'answer': 3}],
                 [{k: v for k, v in valid.items() if k != 'seed'}], [valid, valid.copy()]]
        for bad in cases:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                sw.validate_request(request(prompts=bad))
        for key, bad_values in {'id': [True, 2., [], None],
                                'prompt_token_ids': [[], (), [True], [-1], [1.], [1] * 8192],
                                'seed': [True, -1, 2**63, 1., '1']}.items():
            for bad in bad_values:
                with self.subTest(key=key, bad=str(bad)[:30]), self.assertRaises(ValueError):
                    sw.validate_request(request(prompts=[{**valid, key: bad}]))

    def test_adapter_request_path_and_id(self):
        for path, identifier in [('relative', 1), (3, 1), ('/absolute', 0), ('/absolute', True),
                                 ('/absolute', 2**31), ('/absolute', None)]:
            with self.subTest(path=path, identifier=identifier), self.assertRaises(ValueError):
                sw.validate_request(request(adapter_path=path, adapter_id=identifier))
        self.assertEqual(sw.validate_request(request(adapter_path='/absolute', adapter_id=2**31 - 1))['adapter_id'], 2**31 - 1)


class AtomicJSONTests(unittest.TestCase):
    def test_roundtrip_digest_and_replace(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'nested' / 'value.json'
            sw.atomic_json(path, {'text': 'λ'})
            self.assertEqual(sw.read_json(path), {'text': 'λ'})
            self.assertEqual(sw.file_sha256(path), hashlib.sha256(path.read_bytes()).hexdigest())
            sw.atomic_json(path, {'next': 1})
            self.assertEqual(sw.read_json(path), {'next': 1})
            self.assertEqual([p.name for p in path.parent.iterdir()], ['value.json'])

    def test_serialization_and_publish_failures_leave_original_and_no_temp(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'value.json'
            sw.atomic_json(path, {'original': True})
            for bad in (float('nan'), object()):
                with self.subTest(bad=type(bad)), self.assertRaises((ValueError, TypeError)):
                    sw.atomic_json(path, bad)
            with patch.object(sw.os, 'replace', side_effect=OSError('injected')):
                with self.assertRaisesRegex(OSError, 'injected'):
                    sw.atomic_json(path, {'new': True})
            self.assertEqual(sw.read_json(path), {'original': True})
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_exclusive_publication_preserves_competing_request(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'request.json'
            sw.atomic_json(path, {'winner': 1}, overwrite=False)
            with self.assertRaises(FileExistsError):
                sw.atomic_json(path, {'loser': 2}, overwrite=False)
            self.assertEqual(sw.read_json(path), {'winner': 1})
            self.assertEqual(list(path.parent.iterdir()), [path])


class ProcessorTests(unittest.TestCase):
    def test_generation_preserves_groups_indices_tokens_and_behavior_logprobs(self):
        proc = processor(10)
        value = request(context_limit=10, prompts=[dict(id='a', prompt_token_ids=[1, 2], seed=9),
                                                   dict(id=7, prompt_token_ids=[1] * 8, seed=11)])
        result = proc.handle(value)
        self.assertTrue(result['ok'])
        self.assertEqual(result['protocol_version'], sw.PROTOCOL_VERSION)
        self.assertEqual([r['id'] for r in result['rows']], ['a', 7])
        self.assertEqual([r['max_tokens'] for r in result['rows']], [4, 2])
        self.assertGreaterEqual(result['generation_seconds'], 0)
        for row in result['rows']:
            self.assertEqual([s['sample_index'] for s in row['samples']], [0, 1])
            for sample in row['samples']:
                self.assertEqual(sample['text'], '3|4')
                self.assertEqual(sample['token_ids'], [3, 4])
                self.assertEqual(sample['logprobs'], [-.25, -.25])
                self.assertEqual(sample['generated_tokens'], 2)
                self.assertFalse(sample['backend_text_matches_original_decode'])
        prompts, params, kwargs = proc.llm.calls[0]
        self.assertEqual(prompts, [{'prompt_token_ids': [1, 2]}, {'prompt_token_ids': [1] * 8}])
        self.assertEqual([p.seed for p in params], [9, 11])
        self.assertTrue(all(p.n == 2 and p.logprobs == 0 and p.top_p == 1 and p.top_k == -1 for p in params))
        self.assertEqual(kwargs, dict(lora_request=None, use_tqdm=False))

    def test_invalid_backend_results_fail_closed(self):
        mutations = {
            'question count': lambda rows: rows.clear(),
            'prompt ids': lambda rows: setattr(rows[0], 'prompt_token_ids', [9]),
            'duplicate index': lambda rows: setattr(rows[0].outputs[0], 'index', 0),
            'missing sample': lambda rows: rows[0].outputs.pop(),
            'empty completion': lambda rows: setattr(rows[0].outputs[0], 'token_ids', []),
            'over budget': lambda rows: setattr(rows[0].outputs[0], 'token_ids', [3] * 5),
            'missing logprobs': lambda rows: setattr(rows[0].outputs[0], 'logprobs', None),
            'short logprobs': lambda rows: setattr(rows[0].outputs[0], 'logprobs', []),
            'null candidate': lambda rows: setattr(rows[0].outputs[0], 'logprobs', [None, {}]),
            'absent chosen token': lambda rows: setattr(rows[0].outputs[0], 'logprobs', [{9: NS(logprob=-1)}, {}]),
            'infinite logprob': lambda rows: setattr(rows[0].outputs[0].logprobs[0][3], 'logprob', float('inf')),
            'nan logprob': lambda rows: setattr(rows[0].outputs[0].logprobs[0][3], 'logprob', float('nan')),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                proc = processor(); proc.llm.mutate = mutate
                with self.assertRaises(ValueError):
                    proc.handle(request())
                self.assertEqual(proc.active_adapter_id, 0)

    def test_adapter_identity_hashes_staleness_and_eviction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); one = adapter(root, 'one'); two = adapter(root, 'two')
            proc = processor()
            first = request(adapter_path=str(one), adapter_id=1)
            actual = proc.handle(first)
            self.assertEqual(actual['adapter']['files']['adapter_model.safetensors'], sw.file_sha256(one / 'adapter_model.safetensors'))
            self.assertIsNone(actual['previous_adapter_evicted'])
            proc.handle(first)
            proc.llm.llm_engine.remove_lora.assert_not_called()
            with self.assertRaisesRegex(ValueError, 'original id'):
                proc.handle(request(adapter_path=str(one), adapter_id=2))
            with self.assertRaisesRegex(ValueError, 'reused'):
                proc.handle(request(adapter_path=str(two), adapter_id=1))
            second = proc.handle(request(adapter_path=str(two), adapter_id=2))
            self.assertTrue(second['previous_adapter_evicted'])
            proc.llm.llm_engine.remove_lora.assert_called_once_with(1)
            proc.handle(request())
            self.assertEqual(proc.llm.llm_engine.remove_lora.call_args.args, (2,))
            (one / 'adapter_model.safetensors').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'reused'):
                proc.handle(first)

    def test_adapter_config_rejections_and_incomplete_files(self):
        configs = [dict(peft_type='OTHER'), dict(r=8), dict(lora_alpha=16), dict(lora_dropout=.1),
                   dict(use_dora=True), dict(use_rslora=True), dict(modules_to_save=['x']),
                   dict(rank_pattern={'a': 8}), dict(alpha_pattern={'a': 8})]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for i, changes in enumerate(configs):
                with self.subTest(changes=changes):
                    path = adapter(root, str(i), **changes)
                    with self.assertRaises(ValueError):
                        processor().handle(request(adapter_path=str(path), adapter_id=1))
            missing = adapter(root, 'missing'); (missing / 'adapter_model.safetensors').unlink()
            with self.assertRaisesRegex(ValueError, 'completed'):
                processor().handle(request(adapter_path=str(missing), adapter_id=1))
            with self.assertRaises(FileNotFoundError):
                processor().handle(request(adapter_path=str(root / 'absent'), adapter_id=1))

    def test_make_processor_eager_uses_fake_vllm_and_requires_one_gpu(self):
        for visible in ('', '0,1'):
            with self.subTest(visible=visible), patch.dict(os.environ, CUDA_VISIBLE_DEVICES=visible):
                with self.assertRaisesRegex(RuntimeError, 'exactly one'):
                    sw.make_processor('/fake')
        modules = {name: ModuleType(name) for name in ('vllm', 'vllm.lora', 'vllm.lora.request', 'vllm.config', 'vllm.config.compilation')}
        llm = Mock(return_value=NS(get_tokenizer=lambda: FakeTokenizer()))
        modules['vllm'].LLM = llm; modules['vllm'].SamplingParams = Mock()
        modules['vllm.lora.request'].LoRARequest = Mock()
        compilation = modules['vllm.config.compilation']
        compilation.CompilationConfig = Mock()
        compilation.CompilationMode = NS(NONE='none'); compilation.CUDAGraphMode = NS(FULL_DECODE_ONLY='decode')
        with patch.dict(sys.modules, modules), patch.dict(os.environ, CUDA_VISIBLE_DEVICES='GPU-mocked'):
            sw.make_processor('/fake', eager=True, max_num_seqs=16, gpu_memory_utilization=.7,
                              finetune_mode='lora')
        compilation.CompilationConfig.assert_not_called()
        self.assertNotIn('compilation_config', llm.call_args.kwargs)
        self.assertTrue(llm.call_args.kwargs['enforce_eager'])
        self.assertEqual(llm.call_args.kwargs['gpu_memory_utilization'], .7)
        self.assertTrue(llm.call_args.kwargs['enable_lora'])


class FileIPCLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.children = []
        self.addCleanup(self.reap)
        self.worker = self.root / 'fake_worker.py'
        self.worker.write_text(textwrap.dedent('''\
            import argparse, json, os, pathlib, subprocess, sys, time
            sys.path.insert(0, TRAINING_PATH)
            import sampler_worker as sw
            parser = argparse.ArgumentParser()
            parser.add_argument('--model-path'); parser.add_argument('--ipc-dir')
            parser.add_argument('--gpu-memory-utilization'); parser.add_argument('--max-num-seqs')
            parser.add_argument('--finetune-mode', choices=['full', 'lora'])
            parser.add_argument('--tokenizer-path'); parser.add_argument('--model-version', type=int)
            parser.add_argument('--engine-seed', type=int)
            parser.add_argument('--eager', action='store_true')
            args = parser.parse_args(); root = pathlib.Path(args.ipc_dir)
            if args.model_path == 'descendant':
                code = "import fcntl, os, pathlib, signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); lock=open(" + repr(str(root / 'child.lock')) + ", 'w'); fcntl.flock(lock, fcntl.LOCK_EX); pathlib.Path(" + repr(str(root / 'child_ready.json')) + ").write_text(str(os.getpid())); time.sleep(30)"
                subprocess.Popen([sys.executable, '-c', code])
            sw.atomic_json(root / 'environment.json', dict(args=vars(args), env={k:v for k,v in os.environ.items() if k in ('CUDA_VISIBLE_DEVICES', 'RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT', 'TORCHELASTIC_USE_AGENT_STORE', 'VLLM_WORKER_MULTIPROC_METHOD', 'TOKENIZERS_PARALLELISM', 'PYTHONHASHSEED')}))
            if args.model_path == 'crash': sys.exit(7)
            if args.model_path == 'startup_timeout': time.sleep(30)
            if args.model_path == 'reject':
                sw.atomic_json(root / 'ready.json', {'ok': False, 'error': 'injected startup failure'})
                time.sleep(30)
            class Processor:
                def handle(self, payload):
                    valid = sw.validate_request(payload)
                    if args.model_path == 'error': raise ValueError('injected request failure')
                    if args.model_path == 'block': time.sleep(30)
                    return dict(ok=True, request_id=('wrong' if args.model_path == 'wrong_id' else valid['request_id']), validated=valid)
            sw.serve(root, Processor())
        ''').replace('TRAINING_PATH', repr(str(Path(sw.__file__).parent))))

    def reap(self):
        for process in self.children:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=2)

    def launch(self, mode='good', **kwargs):
        real_popen = subprocess.Popen
        def recording_popen(*args, **options):
            process = real_popen(*args, **options); self.children.append(process); return process
        with patch.object(sw.subprocess, 'Popen', side_effect=recording_popen):
            return sw.SamplerClient.launch(mode, self.root / ('ipc_' + mode), 'GPU-mocked',
                                           python=sys.executable, worker_path=self.worker,
                                           startup_timeout=kwargs.pop('startup_timeout', 3),
                                           finetune_mode='lora', **kwargs)

    def test_actual_subprocess_environment_request_and_graceful_close(self):
        with patch.dict(os.environ, RANK='3', WORLD_SIZE='4', LOCAL_RANK='3', MASTER_ADDR='localhost',
                        MASTER_PORT='1', TORCHELASTIC_USE_AGENT_STORE='True'):
            client = self.launch(eager=True, max_num_seqs=16, engine_seed=43)
        environment = sw.read_json(client.ipc_dir / 'environment.json')
        self.assertEqual(environment['env'], dict(CUDA_VISIBLE_DEVICES='GPU-mocked',
                         VLLM_WORKER_MULTIPROC_METHOD='spawn', TOKENIZERS_PARALLELISM='false', PYTHONHASHSEED='43'))
        self.assertEqual(environment['args']['engine_seed'], 43)
        self.assertEqual(environment['args']['finetune_mode'], 'lora')
        self.assertEqual(environment['args']['model_version'], 0)
        self.assertEqual(environment['args']['tokenizer_path'], 'good')
        self.assertTrue(environment['args']['eager'])
        original = request(); result = client.generate(original, timeout=3)
        self.assertEqual(result['request_id'], original['request_id'])
        self.assertEqual(result['validated']['prompts'][0]['seed'], 42)
        self.assertEqual(sw.read_json(client.ipc_dir / 'requests' / 'req_1.json'), original)
        with self.assertRaises(FileExistsError): client.generate(original, timeout=0)
        auto_id = request(); auto_id.pop('request_id')
        self.assertEqual(len(client.generate(auto_id, timeout=3)['request_id']), 32)
        self.assertNotIn('request_id', auto_id)
        client.close(timeout=3)
        self.assertEqual(client.process.returncode, 0)
        self.assertTrue(sw.read_json(client.ipc_dir / 'stopped.json')['ok'])

    def test_processor_failure_and_response_identity_are_propagated(self):
        for mode, message in [('error', 'request failed'), ('wrong_id', 'id mismatch')]:
            with self.subTest(mode=mode):
                client = self.launch(mode)
                with self.assertRaisesRegex(RuntimeError, message): client.generate(request(), timeout=3)
                client.close(timeout=3)

    def test_server_rejects_filename_mismatch_and_malformed_json_and_keeps_serving(self):
        client = self.launch()
        sw.atomic_json(client.ipc_dir / 'requests' / 'different.json', request())
        mismatch = client.wait_file(client.ipc_dir / 'responses' / 'different.json', timeout=3)
        self.assertEqual(mismatch['error_type'], 'ValueError')
        (client.ipc_dir / 'requests' / 'broken.json').write_text('{broken')
        malformed = client.wait_file(client.ipc_dir / 'responses' / 'broken.json', timeout=3)
        self.assertEqual(malformed['error_type'], 'JSONDecodeError')
        self.assertTrue(client.generate(request(), timeout=3)['ok'])
        observer = sw.SamplerClient(client.ipc_dir)
        observer.close(timeout=3)
        client.process.wait(timeout=3)

    def test_startup_crash_rejection_and_timeout_reap_child(self):
        for mode, error in [('crash', RuntimeError), ('reject', RuntimeError), ('startup_timeout', TimeoutError)]:
            with self.subTest(mode=mode), self.assertRaises(error):
                self.launch(mode, startup_timeout=.3)
            self.assertIsNotNone(self.children[-1].returncode)

    def test_request_and_shutdown_timeouts_are_bounded_and_reap_worker(self):
        client = self.launch('block')
        with self.assertRaises(TimeoutError): client.generate(request(), timeout=.2)
        with self.assertRaises(subprocess.TimeoutExpired): client.close(timeout=.1)
        self.assertIsNotNone(client.process.returncode)

    @unittest.skipUnless(os.name == 'posix', 'Owned POSIX process groups are used by the Linux sampler')
    def test_close_terminates_real_descendant_that_ignores_sigterm(self):
        import fcntl
        client = self.launch('descendant')
        child_pid = client.wait_file(client.ipc_dir / 'child_ready.json', timeout=3)
        self.assertNotEqual(os.getpgid(child_pid), os.getpgrp())
        try:
            with (client.ipc_dir / 'child.lock').open('r') as lock:
                with self.assertRaises(BlockingIOError): fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                client.close(timeout=3)
                deadline = time.monotonic() + 2
                while True:
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        if time.monotonic() >= deadline: self.fail('Sampler descendant still holds its resource lock')
                        time.sleep(.01)
            self.assertIsNotNone(client.process.returncode)
            self.assertFalse(client._owns_process_group)
        finally:
            try: os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError: pass

    def test_external_process_cleanup_never_signals_callers_process_group(self):
        process = Mock(); process.poll.return_value = None
        client = sw.SamplerClient(self.root, process)
        with patch.object(sw.os, 'killpg') as kill_group:
            client._stop_process(timeout=.1)
        kill_group.assert_not_called(); process.terminate.assert_called_once()
        process.wait.assert_called_once_with(timeout=.1)

    def test_wait_file_reports_exit_or_timeout_and_reads_completed_file(self):
        client = sw.SamplerClient(self.root)
        with self.assertRaises(TimeoutError): client.wait_file(self.root / 'absent', timeout=0)
        client.process = NS(poll=lambda: 9, returncode=9)
        with self.assertRaisesRegex(RuntimeError, 'exited 9'): client.wait_file(self.root / 'absent', timeout=0)
        sw.atomic_json(self.root / 'present', {'done': 1})
        self.assertEqual(client.wait_file(self.root / 'present', timeout=0), {'done': 1})

    def test_invalid_device_does_not_create_ipc_directory(self):
        for device in ('', '0,1'):
            with self.subTest(device=device), self.assertRaises(ValueError):
                sw.SamplerClient.launch('/fake', self.root / 'invalid', device)
            self.assertFalse((self.root / 'invalid').exists())

    def test_force_stop_kills_uncooperative_child_and_reaps(self):
        process = Mock(); process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired('worker', .01), 0]
        client = sw.SamplerClient(self.root, process)
        client._stop_process(timeout=.01)
        process.terminate.assert_called_once(); process.kill.assert_called_once()
        self.assertEqual(process.wait.call_count, 2)
        sw.SamplerClient(self.root)._stop_process()

    def test_serve_finishes_current_response_before_obeying_shutdown(self):
        ipc = self.root / 'direct'; ipc.mkdir()
        (ipc / 'requests').mkdir(); (ipc / 'responses').mkdir()
        for identifier in ('a', 'b'):
            sw.atomic_json(ipc / 'requests' / (identifier + '.json'), request(request_id=identifier))
        def handle(value):
            sw.atomic_json(ipc / 'shutdown.json', {'shutdown': True})
            return {'ok': True, 'request_id': value['request_id']}
        proc = NS(handle=Mock(side_effect=handle))
        sw.serve(ipc, proc)
        proc.handle.assert_called_once()
        self.assertTrue(sw.read_json(ipc / 'responses' / 'a.json')['ok'])
        self.assertFalse((ipc / 'responses' / 'b.json').exists())
        self.assertTrue((ipc / 'stopped.json').exists())

    def test_serve_never_reexecutes_completed_request_and_publishes_errors(self):
        ipc = self.root / 'direct'; ipc.mkdir()
        (ipc / 'requests').mkdir(); (ipc / 'responses').mkdir()
        sw.atomic_json(ipc / 'requests' / 'done.json', request(request_id='done'))
        sw.atomic_json(ipc / 'responses' / 'done.json', {'ok': True, 'request_id': 'done'})
        sw.atomic_json(ipc / 'requests' / 'bad.json', request(request_id='different'))
        def idle_sleep(duration): sw.atomic_json(ipc / 'shutdown.json', {'shutdown': True})
        proc = NS(handle=Mock())
        with patch.object(sw.time, 'sleep', side_effect=idle_sleep), patch.object(sw.traceback, 'print_exc') as traceback:
            sw.serve(ipc, proc)
        proc.handle.assert_not_called(); traceback.assert_called_once()
        self.assertEqual(sw.read_json(ipc / 'responses' / 'bad.json')['error_type'], 'ValueError')

    def test_main_success_and_startup_failure_publish_status(self):
        args = ['sampler', '--model-path', '/mock', '--ipc-dir', str(self.root), '--eager']
        with patch.object(sys, 'argv', args), patch.object(sw, 'make_processor', return_value='mock') as make, patch.object(sw, 'serve') as serve:
            sw.main()
        make.assert_called_once_with('/mock', gpu_memory_utilization=.8, max_num_seqs=32, eager=True,
                                     engine_seed=42, finetune_mode='full', tokenizer_path=None,
                                     model_version=0)
        serve.assert_called_once_with(self.root, 'mock')
        with patch.object(sys, 'argv', args), patch.object(sw, 'make_processor', side_effect=RuntimeError('GPU explicitly mocked unavailable')):
            with self.assertRaisesRegex(RuntimeError, 'mocked unavailable'): sw.main()
        self.assertFalse(sw.read_json(self.root / 'ready.json')['ok'])


if __name__ == '__main__':
    unittest.main()
