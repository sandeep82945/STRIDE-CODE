"""Make immutable, inference-only Hugging Face exports without network access.

Completion metadata authenticates the epoch policy. Optimizer/RNG files are
checked for presence, but never loaded, copied or hashed for this inference
export. Resume validation remains the stricter responsibility of the trainer.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile

import resumable_checkpoint as checkpoints


# The pinned upstream identities in scripts/download_model.py. Kept here too
# because the trainer's frozen source bundle contains only training/*.py.
BASE_MODELS = {
    'qwen4b': ('Qwen/Qwen3-4B-Instruct-2507', 'cdbee75f17c01a7cc42f958dc650907174af0554'),
    'qwen17b': ('Qwen/Qwen3-1.7B', '70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'),
}
METHODS = {
    'local_positive': 'STRIDE: nonnegative local step-diversity credit on eligible reasoning tokens.',
    'single_step': 'STRIDE ablation: a single pooled reasoning embedding per eligible response.',
    'token_uniform': 'STRIDE ablation: spread each response’s step-diversity credit uniformly over its eligible tokens.',
    'grpo': 'GRPO baseline: group-standardized answer correctness without a diversity bonus.',
}
TOKENIZER_FILES = {
    'tokenizer.json', 'tokenizer.model', 'tokenizer_config.json', 'special_tokens_map.json',
    'added_tokens.json', 'vocab.json', 'merges.txt', 'chat_template.jinja',
    'chat_template.jinja2',
}
FULL_FILES = TOKENIZER_FILES | {'config.json', 'generation_config.json', 'model.safetensors',
                               'model.safetensors.index.json'}
LORA_FILES = {'adapter_config.json', 'adapter_model.safetensors', 'README.md'}
SHARD = re.compile(r'model-\d{5}-of-\d{5}\.safetensors\Z')
SHA = re.compile(r'[0-9a-f]{64}\Z')
PRIVATE_PATH_KEYS = {'_name_or_path', 'name_or_path', 'tokenizer_file', 'vocab_file',
                     'merges_file', 'special_tokens_map_file', 'added_tokens_file'}


def _object(path):
    value = json.loads(checkpoints._regular(path).read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'Expected JSON object: {Path(path).name}')
    return value


def _encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                       allow_nan=False) + '\n').encode('utf-8')


def _safe_text(payload):
    for name in ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'WANDB_API_KEY'):
        secret = os.environ.get(name, '').strip()
        if secret and secret.encode() in payload:
            raise ValueError('Credentials must not appear in checkpoint export content')
    return payload


def _clean_config(value):
    if isinstance(value, dict):
        return {key: _clean_config(item) for key, item in value.items()
                if key not in PRIVATE_PATH_KEYS}
    if isinstance(value, list):
        return [_clean_config(item) for item in value]
    return value


def _validate_contract(contract):
    if not isinstance(contract, dict):
        raise ValueError('Training contract is missing')
    expected = {'mode': 'train', 'warmup_steps': 0,
                'format_coefficient': 0., 'context_limit': 8192,
                'epoch_offset': 0, 'diversity_alpha_schedule': 'constant'}
    if (type(contract.get('version')) is not int or contract['version'] not in (5, 6)
            or any(contract.get(key) != value for key, value in expected.items())):
        raise ValueError('Export requires the current direct-training contract')
    if (contract.get('model') not in BASE_MODELS or contract.get('arm') not in METHODS
            or contract.get('finetune_mode') not in ('full', 'lora')):
        raise ValueError('Unsupported model, arm or fine-tuning mode')
    for key in ('training_questions', 'batch_size', 'epochs', 'world_size', 'K'):
        if type(contract.get(key)) is not int or contract[key] < 1:
            raise ValueError(f'Invalid training contract {key}')
    if (contract['batch_size'] % contract['world_size']
            or not 2 <= contract['K'] <= 64):
        raise ValueError('Invalid training batch/group schedule')
    tail = contract['training_questions'] % contract['batch_size']
    if contract['version'] == 5:
        if tail:
            raise ValueError('Legacy training contract requires complete batches')
    else:
        if (contract.get('batch_schedule') != 'epoch_shuffle_keep_partial'
                or contract.get('collective_padding') != 'zero_loss_no_generated_samples'
                or type(contract.get('updates_per_epoch')) is not int
                or contract['updates_per_epoch'] != _updates_per_epoch(contract)
                or contract['training_questions'] < contract['batch_size']
                or 0 < tail < contract['world_size']):
            raise ValueError('Invalid partial training batch schedule')
    if type(contract.get('seed')) is not int or not 0 <= contract['seed'] < 2**32:
        raise ValueError('Invalid training seed')
    alpha = contract.get('alpha')
    if type(alpha) not in (int, float) or not math.isfinite(alpha) or alpha < 0:
        raise ValueError('Invalid diversity coefficient')
    if not isinstance(contract.get('data_sha256'), str) or not SHA.fullmatch(contract['data_sha256']):
        raise ValueError('Missing training dataset hash')
    identity = contract.get('model_identity')
    if not isinstance(identity, dict) or not isinstance(identity.get('files'), dict) or not identity['files']:
        raise ValueError('Missing base-model content identity')
    for name, value in identity['files'].items():
        if (not isinstance(name, str) or not name or '\\' in name or Path(name).is_absolute()
                or any(part in ('', '.', '..') for part in name.split('/'))
                or not isinstance(value, str) or not SHA.fullmatch(value)):
            raise ValueError('Invalid base-model file identity')
    identity_hash = hashlib.sha256(json.dumps(identity['files'], sort_keys=True,
                                      separators=(',', ':')).encode()).hexdigest()
    if identity.get('content_sha256') != identity_hash:
        raise ValueError('Base-model content identity mismatch')
    return contract


def _updates_per_epoch(contract):
    """Version six keeps every question in a final, smaller real batch."""
    count, size = contract['training_questions'], contract['batch_size']
    return (count + size - 1) // size


def _run(path, expected=None):
    path = checkpoints._directory(path).resolve()
    started = _object(path / 'started.json')
    contract = _validate_contract(started.get('training_contract'))
    if expected is not None and contract != expected:
        raise ValueError('Resume ancestry training contract differs')
    if started.get('model_identity') != contract['model_identity']:
        raise ValueError('Started model identity disagrees with training contract')
    if started.get('data_sha256') != contract['data_sha256']:
        raise ValueError('Started dataset identity disagrees with training contract')
    if not isinstance(started.get('args'), dict) or started['args'].get('mode') != 'train':
        raise ValueError('Only actual training runs can be exported')
    return path, started, contract


def _committed(run, step, contract):
    """Verify completion/provenance without reading large unused Adam shards."""
    state = checkpoints._directory(run / f'state_{step:03d}')
    marker = _object(state / 'COMPLETE.json')
    metadata_path = checkpoints._regular(state / 'metadata.json')
    if marker.get('metadata_sha256') != checkpoints.digest(metadata_path):
        raise ValueError('Checkpoint completion metadata hash mismatch')
    metadata = _object(metadata_path)
    if (marker.get('format_version') != 2 or metadata.get('format_version') != 2
            or marker.get('completed_step') != step or metadata.get('completed_step') != step
            or metadata.get('world_size') != contract['world_size']
            or metadata.get('contract') != contract):
        raise ValueError('Checkpoint completion or training contract mismatch')
    mode = contract['finetune_mode']
    names = checkpoints._state_names(contract['world_size'], mode)
    state_hashes = metadata.get('state_files', {})
    if (not isinstance(state_hashes, dict) or set(state_hashes) != names
            or any(not isinstance(value, str) or not SHA.fullmatch(value)
                   for value in state_hashes.values())
            or {path.name for path in state.iterdir()} != names | {'metadata.json', 'COMPLETE.json'}):
        raise ValueError('Incomplete or unexpected optimizer/RNG checkpoint inventory')
    for name in names:
        checkpoints._regular(state / name)
    policy_name = f'{"model" if mode == "full" else "adapter"}_{step:03d}'
    layout = 'fsdp_local_v1' if mode == 'full' else 'replicated_v1'
    if (metadata.get('finetune_mode') != mode or metadata.get('optimizer_layout') != layout
            or metadata.get('policy_name') != policy_name):
        raise ValueError('Checkpoint policy type mismatch')
    policy = checkpoints.verify_policy(run / policy_name, mode)
    if metadata.get('policy') != policy:
        raise ValueError('Checkpoint policy provenance mismatch')
    if mode == 'lora' and (metadata.get('adapter_name') != policy_name or metadata.get('adapter') != policy):
        raise ValueError('Checkpoint adapter provenance mismatch')
    return run / policy_name, policy['files']


def _ancestry(run, started, contract):
    result, seen = [run], {run}
    for _ in range(32):
        source = started.get('resume_from')
        if source is None:
            return result
        if not isinstance(source, str) or not source:
            raise ValueError('Invalid resume ancestry path')
        source = Path(source).expanduser()
        if not source.is_absolute():
            source = run / source
        checkpoints._directory(source)
        previous = checkpoints._directory(source.parent).resolve()
        if previous in seen:
            raise ValueError('Cyclic resume ancestry')
        match = checkpoints.STATE_NAME.fullmatch(source.name)
        if match is None or started.get('start_step') != int(match.group(1)):
            raise ValueError('Resume ancestry starting step mismatch')
        _committed(previous, int(match.group(1)), contract)
        run, started, _ = _run(previous, contract)
        seen.add(run)
        result.append(run)
    raise ValueError('Resume ancestry exceeds the bounded traversal limit')


def _base_source(started, contract):
    value = started['args'].get('model_path')
    if not isinstance(value, str) or not value or not Path(value).is_absolute():
        raise ValueError('Saved absolute base-model path is required for export provenance')
    source = checkpoints._directory(Path(value))
    manifest_path = checkpoints._regular(source / 'model_manifest.json')
    identity = contract['model_identity']
    if checkpoints.digest(manifest_path) != identity.get('preparation_manifest_sha256'):
        raise ValueError('Pinned base-model preparation manifest changed or is missing')
    manifest = _object(manifest_path)
    repo, revision = BASE_MODELS[contract['model']]
    if (manifest.get('status') != 'download_complete' or manifest.get('alias') != contract['model']
            or manifest.get('repo_id') != repo or manifest.get('revision') != revision):
        raise ValueError('Base-model preparation manifest does not match the pinned upstream model')
    assets = manifest.get('files')
    if not isinstance(assets, list) or any(not isinstance(entry, dict) for entry in assets):
        raise ValueError('Invalid downloaded base-model inventory')
    downloaded = {entry.get('file'): entry.get('sha256') for entry in assets}
    if len(downloaded) != len(assets) or any(downloaded.get(name) != value
                                           for name, value in identity['files'].items()):
        raise ValueError('Training base-model identity differs from the pinned download')
    return source


def _tokenizer_names(names):
    names = set(names) & TOKENIZER_FILES
    if ('tokenizer_config.json' not in names
            or not ('tokenizer.json' in names or 'tokenizer.model' in names
                    or {'vocab.json', 'merges.txt'} <= names)):
        raise ValueError('Export requires a complete tokenizer')
    return names


def _card(info, description):
    lora = info['finetune_mode'] == 'lora'
    library = 'peft' if lora else 'transformers'
    loading = ('from peft import AutoPeftModelForCausalLM\n'
               'model = AutoPeftModelForCausalLM.from_pretrained(repo_id, torch_dtype="auto")'
               if lora else 'from transformers import AutoModelForCausalLM\n'
               'model = AutoModelForCausalLM.from_pretrained(repo_id, torch_dtype="auto")')
    details = description.strip()
    tail = info['training_questions'] % info['batch_size']
    batch_note = (f'The last optimizer batch of each epoch contains {tail} real questions; '
                  'all training questions are retained without generated-sample padding.'
                  if tail else '')
    return f'''---
library_name: {library}
base_model: {info['base_model']}
pipeline_tag: text-generation
tags:
- reinforcement-learning
- math
- {'grpo' if info['arm'] == 'grpo' else 'stride'}
---

# {info['model']} / {info['arm']} / {info['finetune_mode']} / epoch {info['epoch']}

{METHODS[info['arm']]}

This is an inference checkpoint from {'LoRA adapter training' if lora else 'full-parameter training'},
saved after epoch {info['epoch']} (optimizer update {info['step']}).
{details}

| Setting | Value |
|---|---|
| Base model | `{info['base_model']}` |
| Base revision | `{info['base_revision']}` |
| Distinct training questions | {info['training_questions']} |
| Planned/completed training epochs | {info['total_epochs']} |
| Nominal global prompt batch | {info['batch_size']} |
| Rollouts per question (K) | {info['K']} |
| Seed | {info['seed']} |
| Diversity alpha | {info['alpha']} |
| Prompt plus response limit | 8,192 tokens |
| Structural reward / warmup | None / none |
| Training-file SHA256 | `{info['dataset_sha256']}` |

{batch_note}

All compared methods use the same dataset protocol; the hash above identifies
this run's exact training file. No evaluation or convergence claim is made here.
Correct final answers and parseable steps do not verify intermediate reasoning.

## Loading

Set `repo_id` to this repository's Hugging Face ID.

```python
{loading}
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(repo_id)
```

{'The adapter configuration pins the upstream base model and revision. Original base weights are not included.' if lora else 'The repository contains full trained weights and tokenizer files.'}
Training metadata and payload hashes are in `training_metadata.json` and
`export_manifest.json`. Optimizer states, RNG states, prompts, rollouts, logs and
credentials are excluded. This inference export cannot resume optimizer state.
'''


def _transfer(source, destination, expected):
    source = checkpoints._regular(source)
    if destination.suffix == '.safetensors':
        try:
            os.link(source, destination)
        except OSError:
            with source.open('rb') as incoming, destination.open('xb') as outgoing:
                shutil.copyfileobj(incoming, outgoing, 8 * 1024 * 1024)
    else:
        with source.open('rb') as incoming, destination.open('xb') as outgoing:
            payload = incoming.read()
            if destination.suffix in ('.json', '.txt', '.jinja', '.jinja2'):
                _safe_text(payload)
            outgoing.write(payload)
    if checkpoints.digest(destination) != expected:
        raise ValueError('Checkpoint changed while creating export')


def _export_one(destination, info, policy, hashes, base_source, contract, description):
    mode = info['finetune_mode']
    if mode == 'lora':
        if not set(hashes) <= LORA_FILES:
            raise ValueError('Unexpected adapter files; refusing to export extra content')
        config = _object(policy / 'adapter_config.json')
        lora = contract.get('lora')
        if (not isinstance(lora, dict) or config.get('peft_type') != 'LORA'
                or config.get('task_type') != 'CAUSAL_LM' or config.get('r') != lora.get('rank')
                or config.get('lora_alpha') != lora.get('alpha')
                or config.get('lora_dropout') != lora.get('dropout')
                or set(config.get('target_modules') or ()) != set(lora.get('targets') or ())):
            raise ValueError('Adapter configuration differs from the training contract')
        config = _clean_config(config)
        config.update(base_model_name_or_path=info['base_model'], revision=info['base_revision'])
        copies = {'adapter_model.safetensors': (policy / 'adapter_model.safetensors', hashes['adapter_model.safetensors'])}
        identity_files = contract['model_identity']['files']
        for name in _tokenizer_names(identity_files):
            if checkpoints.digest(checkpoints._regular(base_source / name)) != identity_files[name]:
                raise ValueError('Base tokenizer changed since training')
            copies[name] = base_source / name, identity_files[name]
        generated = {'adapter_config.json': _encoded(config)}
    else:
        if any(name not in FULL_FILES and not SHARD.fullmatch(name) for name in hashes):
            raise ValueError('Unexpected full-model checkpoint files; refusing to export extra content')
        _tokenizer_names(hashes)
        copies = {name: (policy / name, sha) for name, sha in hashes.items()}
        generated = {}
    # Rewrite only small export copies; checkpoint bytes remain untouched.
    for name in ('config.json', 'tokenizer_config.json', 'generation_config.json'):
        if name in copies:
            generated[name] = _encoded(_clean_config(_object(copies[name][0])))
            del copies[name]
    public = dict(info)
    generated['training_metadata.json'] = _encoded(public)
    generated['README.md'] = _card(info, description).encode('utf-8')
    for payload in generated.values():
        _safe_text(payload)
    expected = {name: sha for name, (_, sha) in copies.items()}
    expected.update({name: hashlib.sha256(payload).hexdigest() for name, payload in generated.items()})
    public['files'] = dict(sorted(expected.items()))
    folder = destination / f'epoch-{info["epoch"]:03d}'
    if folder.exists() or folder.is_symlink():
        checkpoints._directory(folder)
        if _object(folder / 'export_manifest.json') != public:
            raise ValueError('Existing export differs; use a fresh export destination')
        if {path.name for path in folder.iterdir()} != set(expected) | {'export_manifest.json'}:
            raise ValueError('Existing export has unexpected or missing files')
        for name, sha in expected.items():
            if checkpoints.digest(checkpoints._regular(folder / name)) != sha:
                raise ValueError('Existing export payload hash mismatch')
            if Path(name).suffix in ('.json', '.md', '.txt', '.jinja', '.jinja2'):
                _safe_text((folder / name).read_bytes())
        return dict(public, folder=str(folder.resolve()))
    stage = Path(tempfile.mkdtemp(prefix=f'.epoch-{info["epoch"]:03d}-', dir=destination))
    try:
        for name, (source, sha) in copies.items():
            _transfer(source, stage / name, sha)
        for name, payload in generated.items():
            with (stage / name).open('xb') as stream:
                stream.write(payload)
        with (stage / 'export_manifest.json').open('xb') as stream:
            stream.write(_encoded(public))
        for path in stage.iterdir():
            with path.open('rb') as stream:
                os.fsync(stream.fileno())
        checkpoints._fsync_directory(stage)
        stage.rename(folder)
        checkpoints._fsync_directory(destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return dict(public, folder=str(folder.resolve()))


def prepare_exports(run_dir, destination, epochs=(2, 3, 4), description=''):
    """Export selected epochs of a completed run, including verified ancestors.

    Requests past the completed schedule are ignored (e.g. a two-epoch pilot
    exports epoch two). An empty selection fails. Identical retries validate and
    reuse existing exports; changed or incomplete exports are never overwritten.
    """
    if not isinstance(description, str):
        raise ValueError('Export description must be text')
    _safe_text(description.encode('utf-8'))
    run, started, contract = _run(run_dir)
    done = _object(run / 'done.json')
    per_epoch = _updates_per_epoch(contract)
    total_steps = per_epoch * contract['epochs']
    if (done.get('complete') is not True or done.get('steps') != total_steps
            or done.get('epochs') != contract['epochs']
            or done.get('finetune_mode') != contract['finetune_mode']):
        raise ValueError('Upload requires a successfully completed training schedule')
    if (not isinstance(epochs, (tuple, list)) or not epochs
            or any(type(epoch) is not int or epoch < 1 for epoch in epochs)
            or len(set(epochs)) != len(epochs)):
        raise ValueError('Export epochs must be distinct positive integers')
    selected = sorted(epoch for epoch in epochs if epoch <= contract['epochs'])
    if not selected:
        raise ValueError('No requested export epoch exists in the completed training schedule')
    ancestry = _ancestry(run, started, contract)
    # Also validate final completion even when exporting an earlier epoch.
    _committed(run, total_steps, contract)
    source = _base_source(started, contract)
    destination = Path(destination).expanduser()
    if destination.is_symlink():
        raise ValueError('Export destination must not be a symlink')
    destination = destination.resolve()
    if destination in ancestry or destination == source or any(
            part.startswith(('state_', 'adapter_', 'model_'))
            for base in [*ancestry, source] if destination.is_relative_to(base)
            for part in destination.relative_to(base).parts):
        raise ValueError('Export destination must not overlap source checkpoint directories')
    destination.mkdir(parents=True, exist_ok=True)
    outputs = []
    for epoch in selected:
        step = epoch * per_epoch
        matches = [parent for parent in ancestry if (parent / f'state_{step:03d}').exists()]
        if not matches:
            raise ValueError(f'Requested epoch {epoch} has no retained committed checkpoint')
        policy, hashes = _committed(matches[0], step, contract)
        # A copied resume boundary may exist twice; both must name identical bytes.
        for parent in matches[1:]:
            _, other = _committed(parent, step, contract)
            if other != hashes:
                raise ValueError('Resume ancestry contains conflicting epoch policies')
        info = {key: contract[key] for key in ('model', 'arm', 'finetune_mode', 'seed', 'K',
                 'alpha', 'training_questions', 'batch_size')}
        info.update(epoch=epoch, step=step, total_epochs=contract['epochs'],
                    dataset_sha256=contract['data_sha256'],
                    run_fingerprint=checkpoints.digest(run / 'started.json'),
                    base_model=BASE_MODELS[contract['model']][0],
                    base_revision=BASE_MODELS[contract['model']][1])
        outputs.append(_export_one(destination, info, policy, hashes, source, contract, description))
    return outputs
