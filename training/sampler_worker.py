"""Persistent, isolated vLLM sampler with atomic file IPC and exact token IDs.

Importing this module does not import torch or vLLM. The client launches a new
process with exactly one visible sampler GPU before those libraries initialize.
The worker never receives answer keys or evaluates correctness.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import traceback
import uuid


PROTOCOL_VERSION = 2
CONTEXT_LIMIT = 8192


def require_finetune_mode(value):
    if value not in ('full', 'lora'):
        raise ValueError('finetune_mode must be full or lora')
    return value


def full_checkpoint_identity(checkpoint_path, *, verify_file_hashes=True):
    """Read only a completed immutable HF model export with a sibling manifest.

    Model exports deliberately exclude optimizer state. Keep the original base
    tokenizer pinned separately when loading these weights into vLLM.
    """
    original = Path(checkpoint_path)
    if not original.is_absolute() or original.is_symlink():
        raise ValueError('Full checkpoint must be an absolute pinned directory')
    checkpoint = original.resolve(strict=True)
    if not checkpoint.is_dir():
        raise ValueError('Full checkpoint must be a directory')
    manifest_path = checkpoint.with_name(checkpoint.name + '_manifest.json')
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError('Completed sibling full-model manifest is required')
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict) or 'config.json' not in manifest:
        raise ValueError('Full-model manifest must contain config.json')
    for name, expected in manifest.items():
        if not isinstance(name, str) or Path(name).name != name or name in {'.', '..'}:
            raise ValueError('Unsafe full-model manifest filename')
        if not isinstance(expected, str) or not re.fullmatch(r'[a-f0-9]{64}', expected):
            raise ValueError('Invalid full-model digest')
        path = checkpoint / name
        if path.is_symlink() or not path.is_file() or (verify_file_hashes and file_sha256(path) != expected):
            raise ValueError('Full-model files do not match the completed manifest')
    if {p.name for p in checkpoint.iterdir()} != set(manifest):
        raise ValueError('Full-model inventory differs from its manifest')
    if any(name.startswith('adapter_') for name in manifest):
        raise ValueError('A LoRA adapter is not a full-model checkpoint')
    config = read_json(checkpoint / 'config.json')
    if not isinstance(config, dict) or not config.get('model_type'):
        raise ValueError('Full-model config lacks a model type')
    index_name = 'model.safetensors.index.json'
    if index_name in manifest:
        index = read_json(checkpoint / index_name)
        weights = index.get('weight_map') if isinstance(index, dict) else None
        if not isinstance(weights, dict) or not weights:
            raise ValueError('Full-model shard index lacks a nonempty weight map')
        for name in weights.values():
            if not isinstance(name, str) or name not in manifest or not name.endswith('.safetensors'):
                raise ValueError('Full-model shard index references an absent or unsafe shard')
    elif 'model.safetensors' not in manifest:
        raise ValueError('Completed full-model safetensors weights are required')
    return dict(path=str(checkpoint), manifest_path=str(manifest_path),
                manifest_sha256=file_sha256(manifest_path), files=manifest)


def atomic_json(path, payload, *, overwrite=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('x', encoding='utf-8') as stream:
            json.dump(payload, stream, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            # A concurrent client must never replace a live request with the
            # same id after both clients have passed an existence check.
            os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require_int(value, name, minimum, maximum=None):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f'Invalid {name}')
    return value


def validate_request(request, context_limit=CONTEXT_LIMIT):
    require_int(context_limit, 'context_limit', 2, CONTEXT_LIMIT)
    permitted = {'request_id', 'prompts', 'adapter_path', 'adapter_id', 'n',
                 'model_checkpoint', 'model_version',
                 'max_tokens', 'temperature', 'context_limit'}
    if not isinstance(request, dict) or set(request) - permitted:
        raise ValueError('Unknown request fields; no answer or grading fields are accepted')
    request_id = request.get('request_id')
    if not isinstance(request_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', request_id):
        raise ValueError('Invalid request_id')
    requested_context = require_int(request.get('context_limit', context_limit), 'context_limit', 2, CONTEXT_LIMIT)
    if requested_context != context_limit:
        raise ValueError('Request and worker context limits differ')
    n = require_int(request.get('n', 8), 'n', 1, 64)
    cap = require_int(request.get('max_tokens', 8192), 'max_tokens', 1, 8192)
    temperature = request.get('temperature', 1.0)
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('temperature must be positive and finite')
    prompts = request.get('prompts')
    if not isinstance(prompts, list) or not prompts:
        raise ValueError('At least one prompt is required')
    seen = set()
    result = []
    for prompt in prompts:
        if not isinstance(prompt, dict) or set(prompt) != {'id', 'prompt_token_ids', 'seed'}:
            raise ValueError('Prompts contain only id, prompt_token_ids, and seed')
        identifier = prompt['id']
        if isinstance(identifier, bool) or not isinstance(identifier, (int, str)):
            raise ValueError('Prompt id must be a string or integer')
        identity = (type(identifier).__name__, identifier)
        if identity in seen:
            raise ValueError('Duplicate prompt id')
        seen.add(identity)
        ids = prompt['prompt_token_ids']
        if not isinstance(ids, list) or not ids or len(ids) >= context_limit:
            raise ValueError('Prompt leaves no room within the total context limit')
        for token in ids:
            require_int(token, 'prompt token', 0)
        seed = require_int(prompt['seed'], 'seed', 0, 2**63 - 1)
        result.append(dict(id=identifier, prompt_token_ids=ids, seed=seed,
                           max_tokens=min(cap, context_limit - len(ids))))
    adapter_path = request.get('adapter_path')
    adapter_id = request.get('adapter_id', 0)
    if adapter_path is None:
        if adapter_id is not None and (type(adapter_id) is not int or adapter_id != 0):
            raise ValueError('Base sampling uses adapter_id=0 or None')
        adapter_id = 0
    else:
        if not isinstance(adapter_path, str) or not Path(adapter_path).is_absolute():
            raise ValueError('Adapter path must be absolute')
        require_int(adapter_id, 'adapter_id', 1, 2**31 - 1)
    model_checkpoint = request.get('model_checkpoint')
    model_version = request.get('model_version', 0)
    if model_checkpoint is None:
        if model_version is not None and (type(model_version) is not int or model_version != 0):
            raise ValueError('Unchanged base sampling uses model_version=0 or None')
        model_version = 0
    else:
        if not isinstance(model_checkpoint, str) or not Path(model_checkpoint).is_absolute():
            raise ValueError('Full model checkpoint path must be absolute')
        require_int(model_version, 'model_version', 1, 2**31 - 1)
        if adapter_path is not None:
            raise ValueError('A request cannot combine full-model weights and a LoRA adapter')
    return dict(request_id=request_id, prompts=result, adapter_path=adapter_path,
                adapter_id=adapter_id, n=n, temperature=float(temperature),
                context_limit=context_limit, model_checkpoint=model_checkpoint,
                model_version=model_version)


class SamplerProcessor:
    def __init__(self, llm, tokenizer, sampling_class, lora_class, *, context_limit=CONTEXT_LIMIT,
                 finetune_mode='full', model_identity=None, model_version=0):
        self.llm, self.tokenizer = llm, tokenizer
        self.sampling_class, self.lora_class = sampling_class, lora_class
        self.context_limit = context_limit
        self.seen_adapters = {}
        self.path_ids = {}
        self.active_adapter_id = 0
        self.finetune_mode = require_finetune_mode(finetune_mode)
        self.model_identity, self.model_version = model_identity, model_version

    def adapter_request(self, request):
        if not request['adapter_path']:
            return None, None
        path = Path(request['adapter_path']).resolve(strict=True)
        files = [path / 'adapter_config.json', path / 'adapter_model.safetensors']
        if not all(p.is_file() for p in files):
            raise ValueError('A completed safetensors adapter and config are required')
        cfg = read_json(files[0])
        if cfg.get('peft_type') != 'LORA' or cfg.get('r') != 16 or cfg.get('lora_alpha') != 32 or cfg.get('lora_dropout') != 0:
            raise ValueError('Expected rank16, alpha32, dropout0 LoRA')
        if cfg.get('use_dora') or cfg.get('use_rslora') or cfg.get('modules_to_save') or cfg.get('rank_pattern') or cfg.get('alpha_pattern'):
            raise ValueError('This pilot supports ordinary fixed-rank LoRA without saved base modules')
        fingerprint = {p.name: file_sha256(p) for p in files}
        identity = dict(path=str(path), files=fingerprint)
        adapter_id = request['adapter_id']
        if adapter_id in self.seen_adapters and self.seen_adapters[adapter_id] != identity:
            raise ValueError('An adapter id was reused for changed weights or another path')
        if str(path) in self.path_ids and self.path_ids[str(path)] != adapter_id:
            raise ValueError('An immutable adapter path must retain its original id')
        self.seen_adapters[adapter_id] = identity
        self.path_ids[str(path)] = adapter_id
        return self.lora_class(f'pilot_adapter_{adapter_id}', adapter_id, str(path)), identity

    def handle(self, payload):
        request = validate_request(payload, self.context_limit)
        if self.finetune_mode == 'full':
            if request['adapter_path'] is not None:
                raise ValueError('Full-model sampler does not accept LoRA adapters')
            requested_path = (str(Path(request['model_checkpoint']).resolve(strict=True))
                              if request['model_checkpoint'] else None)
            loaded_path = self.model_identity['path'] if self.model_identity else None
            if requested_path != loaded_path or request['model_version'] != self.model_version:
                raise ValueError('Full-model worker does not contain the requested checkpoint/version')
        elif request['model_checkpoint'] is not None:
            raise ValueError('LoRA sampler does not accept full-model checkpoints')
        lora_request, adapter_identity = self.adapter_request(request)
        params = [self.sampling_class(
            n=request['n'], temperature=request['temperature'], top_p=1.0,
            top_k=-1, max_tokens=prompt['max_tokens'], seed=prompt['seed'],
            skip_special_tokens=False, logprobs=0,
        ) for prompt in request['prompts']]
        prompts = [{'prompt_token_ids': p['prompt_token_ids']} for p in request['prompts']]
        before = time.monotonic()
        outputs = self.llm.generate(prompts, params, lora_request=lora_request, use_tqdm=False)
        elapsed = time.monotonic() - before
        if len(outputs) != len(prompts):
            raise ValueError('Sampler output question count changed')
        rows = []
        for prompt, output in zip(request['prompts'], outputs):
            if list(output.prompt_token_ids) != prompt['prompt_token_ids']:
                raise ValueError('vLLM changed the original prompt token ids')
            completions = sorted(output.outputs, key=lambda item: item.index)
            if [s.index for s in completions] != list(range(request['n'])):
                raise ValueError('Sampler did not return each requested sample exactly once')
            samples = []
            for sample in completions:
                ids = list(sample.token_ids)
                if not ids or len(ids) > prompt['max_tokens']:
                    raise ValueError('Empty or over-budget completion')
                if sample.logprobs is None or len(sample.logprobs) != len(ids):
                    raise ValueError('Missing chosen-token behavior log probabilities')
                probabilities = []
                for token, candidates in zip(ids, sample.logprobs):
                    if candidates is None or token not in candidates:
                        raise ValueError('Chosen token missing from vLLM logprobs=0 result')
                    value = float(candidates[token].logprob)
                    if not math.isfinite(value):
                        raise ValueError('Nonfinite chosen-token behavior log probability')
                    probabilities.append(value)
                text = self.tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                samples.append(dict(sample_index=sample.index, token_ids=ids, text=text,
                                    finish_reason=sample.finish_reason, stop_reason=sample.stop_reason,
                                    logprobs=probabilities, generated_tokens=len(ids),
                                    backend_text_matches_original_decode=sample.text == text))
            rows.append(dict(id=prompt['id'], prompt_token_ids=prompt['prompt_token_ids'],
                             seed=prompt['seed'], max_tokens=prompt['max_tokens'], samples=samples))
        # Synchronous generate has finished. Unload an obsolete GPU/CPU adapter
        # through the engine API, never by deleting or overwriting its files.
        previous = self.active_adapter_id
        current = request['adapter_id']
        evicted = None
        if previous and previous != current:
            evicted = bool(self.llm.llm_engine.remove_lora(previous))
        self.active_adapter_id = current
        return dict(protocol_version=PROTOCOL_VERSION, request_id=request['request_id'],
                    ok=True, rows=rows, adapter_id=current, adapter=adapter_identity,
                    previous_adapter_evicted=evicted, generation_seconds=elapsed,
                    context_limit=self.context_limit, finetune_mode=self.finetune_mode,
                    model_checkpoint=self.model_identity, model_version=self.model_version)


def decode_graph_capture_sizes(max_num_seqs):
    """Cover the configured sampler concurrency without enlarging its KV budget."""
    if isinstance(max_num_seqs, bool) or not isinstance(max_num_seqs, int) or max_num_seqs not in (16, 32, 64):
        raise ValueError('Supported sampler concurrency is 16,32,or64')
    return [size for size in (1, 2, 4, 8, 16, 24, 32, 48, 64) if size <= max_num_seqs]


def make_processor(model_path, *, gpu_memory_utilization=.8, max_num_seqs=32, eager=False,
                   engine_seed=42, finetune_mode='full', tokenizer_path=None,
                   model_version=0):
    require_int(engine_seed, 'engine_seed', 0, 2**32 - 1)
    require_finetune_mode(finetune_mode)
    require_int(model_version, 'model_version', 0, 2**31 - 1)
    if finetune_mode == 'lora' and model_version:
        raise ValueError('LoRA engine cannot declare a full-model version')
    identity = full_checkpoint_identity(model_path) if model_version else None
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not visible or ',' in visible:
        raise RuntimeError('Sampler process requires exactly one explicitly visible GPU')
    if os.environ.get('STRIDE_RUNTIME') == 'rocm':
        allocated = os.environ.get('ROCR_VISIBLE_DEVICES', '').split(',')
        if (not all(allocated) or len(set(allocated)) != len(allocated)
                or visible not in [str(i) for i in range(len(allocated))]
                or os.environ.get('HIP_VISIBLE_DEVICES') != visible):
            raise RuntimeError('ROCm sampler requires one logical HIP device inside the unchanged ROCR allocation')
        import torch
        if not torch.version.hip or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError('ROCm sampler requires exactly one usable allocated GPU')
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from vllm.config.compilation import CompilationConfig, CompilationMode, CUDAGraphMode
    capture_sizes = decode_graph_capture_sizes(max_num_seqs)
    graph_options = {} if eager else {'compilation_config': CompilationConfig(
        mode=CompilationMode.NONE, cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
        cudagraph_capture_sizes=capture_sizes,max_cudagraph_capture_size=max_num_seqs,
        cudagraph_specialize_lora=False)}
    lora_options = dict(enable_lora=True, max_lora_rank=16, max_loras=1, max_cpu_loras=2) if finetune_mode == 'lora' else dict(enable_lora=False)
    llm = LLM(
        model=model_path, tokenizer=tokenizer_path or model_path, dtype='bfloat16', seed=engine_seed,
        tensor_parallel_size=1, **lora_options,
        max_model_len=CONTEXT_LIMIT, max_num_seqs=max_num_seqs,
        max_num_batched_tokens=8192,
        gpu_memory_utilization=gpu_memory_utilization, enforce_eager=eager,
        enable_prefix_caching=False, safetensors_load_strategy='eager', **graph_options,
    )
    return SamplerProcessor(llm, llm.get_tokenizer(), SamplingParams, LoRARequest,
                            finetune_mode=finetune_mode, model_identity=identity,
                            model_version=model_version)


def serve(ipc_dir, processor):
    ipc_dir = Path(ipc_dir)
    requests = ipc_dir / 'requests'
    responses = ipc_dir / 'responses'
    requests.mkdir(exist_ok=True)
    responses.mkdir(exist_ok=True)
    atomic_json(ipc_dir / 'ready.json', dict(ok=True, protocol_version=PROTOCOL_VERSION,
                pid=os.getpid(), context_limit=CONTEXT_LIMIT,
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')))
    while not (ipc_dir / 'shutdown.json').exists():
        pending = [p for p in sorted(requests.glob('*.json')) if not (responses / p.name).exists()]
        if not pending:
            time.sleep(.1)
            continue
        for path in pending:
            if (ipc_dir / 'shutdown.json').exists():
                break
            request_id = path.stem
            try:
                payload = read_json(path)
                if payload.get('request_id') != request_id:
                    raise ValueError('Request filename and id differ')
                result = processor.handle(payload)
            except Exception as exc:
                traceback.print_exc()
                result = dict(ok=False, request_id=request_id,
                              error_type=type(exc).__name__, error=str(exc))
            atomic_json(responses / path.name, result)
    atomic_json(ipc_dir / 'stopped.json', dict(ok=True, pid=os.getpid()))


class SamplerClient:
    """Trainer helper with an isolated worker and explicit checkpoint freshness.

    LoRA keeps a persistent base engine. Full tuning restarts the complete vLLM
    process for every new weight version; no private in-place weight API or
    stale CUDA graph/KV state is relied on. This costs model load/graph capture
    time per update and is intended as the conservative portable implementation.
    """
    def __init__(self, ipc_dir, process=None, log_path=None, *, owns_process_group=False):
        self.ipc_dir = Path(ipc_dir)
        self.process = process
        self.log_path = Path(log_path) if log_path else self.ipc_dir / 'worker.log'
        self._owns_process_group = owns_process_group
        self.finetune_mode = 'full'
        self._active_model = None
        self._active_model_version = 0
        self._seen_models = {}
        self._model_path_versions = {}
        self._submitted_request_ids = set()
        self._launch_options = None
        self._ipc_root = self.ipc_dir

    @classmethod
    def launch(cls, model_path, ipc_dir, sampler_device, *, python=sys.executable,
               worker_path=__file__, gpu_memory_utilization=.8, max_num_seqs=32,
               startup_timeout=600, eager=False, engine_seed=42, finetune_mode='full',
               tokenizer_path=None, model_version=0):
        require_int(engine_seed, 'engine_seed', 0, 2**32 - 1)
        require_finetune_mode(finetune_mode)
        require_int(model_version, 'model_version', 0, 2**31 - 1)
        if finetune_mode == 'lora' and model_version:
            raise ValueError('LoRA launch cannot specify a full-model version')
        # The replacement worker verifies all weight bytes before announcing
        # readiness; the client reads the declared identity without rereading
        # the same multi-GB weights once more during its launch.
        model_identity = full_checkpoint_identity(model_path, verify_file_hashes=False) if model_version else None
        device = str(sampler_device)
        if not device or ',' in device:
            raise ValueError('Pass exactly one GPU id from the Slurm allocation')
        ipc_dir = Path(ipc_dir)
        ipc_dir.mkdir(parents=True, exist_ok=False)
        (ipc_dir / 'requests').mkdir()
        (ipc_dir / 'responses').mkdir()
        env = os.environ.copy()
        # This sampler is an independent single-GPU vLLM process, not a
        # participant in the parent's four-rank torchrun learner group.
        learner_variables = {'RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE',
                             'GROUP_RANK', 'ROLE_RANK', 'ROLE_WORLD_SIZE',
                             'MASTER_ADDR', 'MASTER_PORT'}
        for name in list(env):
            # In particular, TORCHELASTIC_USE_AGENT_STORE=True forces even
            # vLLM's rank zero to create a TCPStore CLIENT. Its private port
            # has no agent server, so keeping this inherited flag deadlocks.
            if name in learner_variables or name.startswith('TORCHELASTIC_'):
                env.pop(name)
        env.update(CUDA_VISIBLE_DEVICES=device, VLLM_WORKER_MULTIPROC_METHOD='spawn',
                   TOKENIZERS_PARALLELISM='false', PYTHONHASHSEED=str(engine_seed))
        if env.get('STRIDE_RUNTIME') == 'rocm':
            # The parent launcher has normalized CUDA/HIP to ROCR-relative
            # logical indices. HIP takes precedence over CUDA on ROCm, so the
            # sampler must narrow HIP as well while retaining ROCR unchanged.
            allocated = env.get('ROCR_VISIBLE_DEVICES', '').split(',')
            logical = [str(i) for i in range(len(allocated))]
            if (len(allocated) not in (1, 8) or not all(allocated)
                    or len(set(allocated)) != len(allocated) or device not in logical
                    or os.environ.get('CUDA_VISIBLE_DEVICES') != ','.join(logical)
                    or os.environ.get('HIP_VISIBLE_DEVICES') != ','.join(logical)):
                raise ValueError('ROCm sampler needs a normalized parent GPU allocation')
            env['HIP_VISIBLE_DEVICES'] = device
        command = [python, '-u', str(Path(worker_path).resolve()), '--model-path', str(model_path),
                   '--ipc-dir', str(ipc_dir.resolve()), '--gpu-memory-utilization', str(gpu_memory_utilization),
                   '--max-num-seqs', str(max_num_seqs), '--engine-seed', str(engine_seed),
                   '--finetune-mode', finetune_mode, '--model-version', str(model_version),
                   '--tokenizer-path', str(tokenizer_path or model_path)]
        if eager: command.append('--eager')
        log_path = ipc_dir / 'worker.log'
        with log_path.open('xb') as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, start_new_session=True)
        client = cls(ipc_dir, process, log_path, owns_process_group=True)
        client.finetune_mode = finetune_mode
        client._active_model = model_identity
        client._active_model_version = model_version
        client._launch_options = dict(
            sampler_device=sampler_device, python=python, worker_path=worker_path,
            gpu_memory_utilization=gpu_memory_utilization, max_num_seqs=max_num_seqs,
            startup_timeout=startup_timeout, eager=eager, engine_seed=engine_seed,
            finetune_mode=finetune_mode, tokenizer_path=str(tokenizer_path or model_path))
        if model_identity is not None:
            client._seen_models[model_version] = model_identity
            client._model_path_versions[model_identity['path']] = model_version
        try:
            ready = client.wait_file(ipc_dir / 'ready.json', timeout=startup_timeout)
            if not ready.get('ok'):
                raise RuntimeError(f'Sampler startup failed: {ready}; log={log_path}')
        except BaseException:
            client._stop_process()
            raise
        return client

    def _ensure_full_checkpoint(self, request):
        """Wait for the old worker to release its GPU before loading new weights."""
        if self.finetune_mode == 'lora':
            if request['model_checkpoint'] is not None:
                raise ValueError('LoRA sampler does not accept full-model checkpoints')
            return 0.0
        if request['adapter_path'] is not None:
            raise ValueError('Full-model sampler does not accept LoRA adapters')
        if request['model_checkpoint'] is None:
            if self._active_model is not None:
                raise ValueError('A full-model checkpoint/version is required after loading trained weights')
            return 0.0
        identity = full_checkpoint_identity(request['model_checkpoint'], verify_file_hashes=False)
        version = request['model_version']
        if version in self._seen_models and self._seen_models[version] != identity:
            raise ValueError('A full-model version was reused for changed weights or another path')
        previous_version = self._model_path_versions.get(identity['path'])
        if previous_version is not None and previous_version != version:
            raise ValueError('An immutable full-model path must retain its original version')
        if self._active_model == identity and self._active_model_version == version:
            full_checkpoint_identity(request['model_checkpoint'])
            return 0.0
        if self._launch_options is None:
            raise RuntimeError('Full-model checkpoint refresh requires an owned launched worker')
        began = time.monotonic()
        try:
            self.close(timeout=5)
        except subprocess.TimeoutExpired:
            # close() already terminates and reaps the owned process group.
            # Some vLLM versions do not finish Python teardown promptly after
            # synchronous generation; that must not prevent a clean reload.
            if self.process is None or self.process.poll() is None:
                raise
        # Each worker has an independent mailbox so shutdown files and old
        # ready/results can never be mistaken for the replacement's state.
        next_ipc = self._ipc_root / 'full_weight_workers' / f'model_{version:06d}_{uuid.uuid4().hex}'
        replacement = type(self).launch(identity['path'], next_ipc,
                                        model_version=version, **self._launch_options)
        self.ipc_dir = replacement.ipc_dir
        self.process, self.log_path = replacement.process, replacement.log_path
        self._owns_process_group = replacement._owns_process_group
        replacement._owns_process_group = False
        self._active_model, self._active_model_version = identity, version
        self._seen_models[version] = identity
        self._model_path_versions[identity['path']] = version
        return time.monotonic() - began

    def _stop_process(self, *, timeout=5):
        """Reap the worker and terminate descendants only in a group we own."""
        if self.process is None:
            return
        if self._owns_process_group:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        elif self.process.poll() is None:
            self.process.terminate()
        try:
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            if self._owns_process_group:
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                self.process.kill()
            self.process.wait(timeout=timeout)
        finally:
            if self._owns_process_group:
                # A worker can exit while a vLLM descendant ignores SIGTERM.
                # Reaping only the worker would leave that GPU process alive.
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self._owns_process_group = False

    def wait_file(self, path, *, timeout=900):
        deadline = time.monotonic() + timeout
        path = Path(path)
        while True:
            if path.exists():
                return read_json(path)
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(f'Sampler exited {self.process.returncode}; log={self.log_path}')
            if time.monotonic() >= deadline:
                raise TimeoutError(f'Sampler timeout waiting for {path.name}; log={self.log_path}')
            time.sleep(.1)

    def generate(self, request, *, timeout=900):
        request = dict(request)
        request.setdefault('request_id', uuid.uuid4().hex)
        validated = validate_request(request)
        if validated['request_id'] in self._submitted_request_ids:
            raise FileExistsError('Request id already exists; do not replay across model versions')
        refresh_seconds = self._ensure_full_checkpoint(validated)
        name = validated['request_id'] + '.json'
        target = self.ipc_dir / 'requests' / name
        response = self.ipc_dir / 'responses' / name
        if target.exists() or response.exists():
            raise FileExistsError('Request id already exists; do not overwrite/replay a live request')
        atomic_json(target, request, overwrite=False)
        self._submitted_request_ids.add(validated['request_id'])
        result = self.wait_file(response, timeout=timeout)
        if result.get('request_id') != validated['request_id']:
            raise RuntimeError('Sampler response id mismatch')
        if not result.get('ok'):
            raise RuntimeError(f'Sampler request failed: {result}; log={self.log_path}')
        if self.finetune_mode == 'full':
            if result.get('model_version') != validated['model_version'] or result.get('model_checkpoint') != self._active_model:
                raise RuntimeError('Sampler returned a different full-model checkpoint/version')
        result['full_model_refresh_seconds'] = refresh_seconds
        return result

    def close(self, *, timeout=60):
        atomic_json(self.ipc_dir / 'shutdown.json', dict(shutdown=True))
        if self.process is not None:
            try:
                self.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self._stop_process()
                raise
            if self._owns_process_group:
                self._stop_process()
        else:
            self.wait_file(self.ipc_dir / 'stopped.json', timeout=timeout)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--ipc-dir', required=True)
    parser.add_argument('--gpu-memory-utilization', type=float, default=.8)
    parser.add_argument('--max-num-seqs', type=int, default=32)
    parser.add_argument('--engine-seed', type=int, default=42)
    parser.add_argument('--finetune-mode', choices=['full', 'lora'], default='full')
    parser.add_argument('--tokenizer-path')
    parser.add_argument('--model-version', type=int, default=0)
    parser.add_argument('--eager', action='store_true')
    args = parser.parse_args()
    directory = Path(args.ipc_dir)
    directory.mkdir(parents=True, exist_ok=True)
    try:
        processor = make_processor(args.model_path,
                                   gpu_memory_utilization=args.gpu_memory_utilization,
                                   max_num_seqs=args.max_num_seqs,eager=args.eager,
                                   engine_seed=args.engine_seed, finetune_mode=args.finetune_mode,
                                   tokenizer_path=args.tokenizer_path, model_version=args.model_version)
        serve(directory, processor)
    except Exception as exc:
        atomic_json(directory / 'ready.json', dict(ok=False, error_type=type(exc).__name__, error=str(exc)))
        raise


if __name__ == '__main__':
    main()
