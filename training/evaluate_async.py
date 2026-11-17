"""Explicit one-GPU evaluation of a local Qwen full checkpoint or LoRA adapter.

This command never trains or makes judge API calls. It uses the same native step
parser and signed-integer answer graders as training. Text distinctness is only
a lexical diagnostic; this is not a general-purpose math benchmark evaluator.
Run --help without importing torch, transformers, PEFT, or vLLM.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
import uuid

from sampler_worker import atomic_json, file_sha256, full_checkpoint_identity, require_finetune_mode
from pilot_grading.integer_reference import _ground_truth

# Keep all runtime dependencies inside training/ so run.py's frozen source
# snapshot remains self-contained. This normalization matches data preparation.
from runtime_config import canonical_question as canonical, THINKING_MODE


def question_hash(text):
    return hashlib.sha256(canonical(text).encode('utf-8')).hexdigest()

CONTEXT_LIMIT = 8192
MODEL_REPOS = {
    'qwen4b': 'Qwen/Qwen3-4B-Instruct-2507',
    'qwen17b': 'Qwen/Qwen3-1.7B',
}


def _read_json(path):
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream)


def _write_once(path, payload):
    """Create a final evidence file exclusively; never overwrite another run."""
    path = Path(path)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def _question_id(value):
    if isinstance(value, bool) or not isinstance(value, (str, int)) or not str(value).strip():
        raise ValueError('Question IDs must be nonempty strings or integers')
    return str(value).strip()


def load_data(path, expected_sha256=None, limit=None):
    """Validate every row before applying an optional deterministic prefix limit."""
    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None:
        if not re.fullmatch(r'[a-f0-9]{64}', expected_sha256) or actual_sha256 != expected_sha256:
            raise ValueError('Dataset SHA256 mismatch')
    rows = json.loads(raw)
    if not isinstance(rows, list) or not rows:
        raise ValueError('Dataset must be a nonempty JSON array')
    seen_ids, seen_questions = set(), set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not {'id', 'question', 'ground_truth'} <= row.keys():
            raise ValueError(f'Malformed question record at row {index}')
        identity = _question_id(row['id'])
        if identity in seen_ids:
            raise ValueError('Duplicate question ID in dataset')
        seen_ids.add(identity)
        if not isinstance(row['question'], str) or not canonical(row['question']):
            raise ValueError(f'Empty or non-string question at row {index}')
        signature = question_hash(row['question'])
        if signature in seen_questions:
            raise ValueError('Duplicate normalized question in dataset')
        seen_questions.add(signature)
        try:
            _ground_truth(row['ground_truth'])
        except ValueError as exc:
            raise ValueError(f'Unsupported reference at row {index}: signed-integer answers only') from exc
    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= len(rows):
            raise ValueError('Evaluation limit must be between one and the dataset size')
        rows = rows[:limit]
    return rows


def check_overlap(evaluation_rows, training_rows):
    """Reject identity or presentation-normalized duplicates, without answer use."""
    ids = {_question_id(row['id']) for row in training_rows}
    questions = {question_hash(row['question']) for row in training_rows}
    id_overlap = sum(_question_id(row['id']) in ids for row in evaluation_rows)
    text_overlap = sum(question_hash(row['question']) in questions for row in evaluation_rows)
    if id_overlap or text_overlap:
        raise ValueError(f'Train/evaluation overlap: {id_overlap} IDs, {text_overlap} normalized questions. '
                         'Use dataset-prefixed IDs for unrelated external benchmarks.')
    return dict(checked=True, training_questions=len(training_rows),
                evaluation_questions=len(evaluation_rows), id_overlap=0, question_overlap=0,
                semantic_near_duplicate_check=False)


def checkpoint_identity(adapter_path, *, model_identity=None, model_alias=None, model_path=None,
                        finetune_mode='lora'):
    """Check a completed immutable policy export and its recorded base identity."""
    require_finetune_mode(finetune_mode)
    original = Path(adapter_path)
    if original.is_symlink():
        raise ValueError('Evaluation checkpoint must be a pinned directory, not a symlink')
    checkpoint = original.resolve(strict=True)
    if not checkpoint.is_dir():
        raise ValueError('Checkpoint must be a directory')
    manifest_path = checkpoint.with_name(checkpoint.name + '_manifest.json')
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError('Completed sibling checkpoint manifest is required before evaluation')
    manifest = _read_json(manifest_path)
    required = {'adapter_model.safetensors', 'adapter_config.json'} if finetune_mode == 'lora' else {'config.json'}
    if not isinstance(manifest, dict) or not required <= manifest.keys():
        raise ValueError('Checkpoint manifest lacks completed weights/config')
    for name, expected in manifest.items():
        if not isinstance(name, str) or Path(name).name != name or name in {'.', '..'}:
            raise ValueError('Unsafe checkpoint manifest filename')
        if not isinstance(expected, str) or not re.fullmatch(r'[a-f0-9]{64}', expected):
            raise ValueError('Invalid checkpoint digest')
        path = checkpoint / name
        if path.is_symlink() or not path.is_file() or file_sha256(path) != expected:
            raise ValueError('Checkpoint files do not match the completed manifest')
    if {p.name for p in checkpoint.iterdir()} != set(manifest):
        raise ValueError('Checkpoint inventory differs from its manifest')
    if finetune_mode == 'full':
        # File bytes were already verified by the common manifest loop above.
        full_checkpoint_identity(str(checkpoint), verify_file_hashes=False)
    config = _read_json(checkpoint / ('adapter_config.json' if finetune_mode == 'lora' else 'config.json'))
    if not isinstance(config, dict):
        raise ValueError('Checkpoint config must be a JSON object')
    if finetune_mode == 'lora' and (config.get('peft_type'), config.get('r'), config.get('lora_alpha'), config.get('lora_dropout')) != ('LORA', 16, 32, 0):
        raise ValueError('Expected rank16, alpha32, dropout0 LoRA')
    if finetune_mode == 'lora' and any(config.get(key) for key in ('use_dora', 'use_rslora', 'modules_to_save', 'rank_pattern', 'alpha_pattern')):
        raise ValueError('Only ordinary fixed-rank LoRA adapters are supported')
    # A portable training run stores hashes of actual model files. Comparing
    # content identity permits relocation while rejecting a different model.
    metadata_path = checkpoint.parent / 'started.json'
    source_identity = None
    source_training_data_sha256 = None
    metadata_sha256 = None
    if metadata_path.is_file():
        metadata = _read_json(metadata_path)
        if not isinstance(metadata, dict):
            raise ValueError('Training provenance must be a JSON object')
        contract = metadata.get('training_contract', {})
        if not isinstance(contract, dict):
            raise ValueError('Malformed training contract')
        source_mode = contract.get('finetune_mode') or metadata.get('args', {}).get('finetune_mode')
        if source_mode is not None and source_mode != finetune_mode:
            raise ValueError('Checkpoint training mode differs from the evaluation checkpoint option')
        source_training_data_sha256 = metadata.get('data_sha256') or contract.get('data_sha256')
        source_identity = metadata.get('model_identity') or contract.get('model_identity')
        source_alias = contract.get('model') or metadata.get('args', {}).get('model')
        if source_alias in MODEL_REPOS and model_alias is not None and source_alias != model_alias:
            raise ValueError('Checkpoint training model alias differs from evaluation model')
        if source_identity is not None:
            if not isinstance(source_identity, dict) or not source_identity.get('content_sha256'):
                raise ValueError('Malformed training model identity')
            if model_identity is not None and source_identity['content_sha256'] != model_identity.get('content_sha256'):
                raise ValueError('Checkpoint base-model content identity differs from evaluation model')
        metadata_sha256 = file_sha256(metadata_path)
    declared_base = config.get('base_model_name_or_path') if finetune_mode == 'lora' else None
    if model_alias is not None and declared_base in MODEL_REPOS.values() and declared_base != MODEL_REPOS[model_alias]:
        raise ValueError('Adapter config identifies a different Qwen base model')
    # Historical adapters can contain a defunct absolute path. Do not mistake
    # path equality for file identity or claim a recovered model hash.
    return dict(path=str(checkpoint), manifest_path=str(manifest_path),
                manifest_sha256=file_sha256(manifest_path), files=manifest,
                source_model_content_sha256=source_identity.get('content_sha256') if source_identity else None,
                source_model_identity_verified=bool(source_identity and model_identity),
                source_metadata_sha256=metadata_sha256,
                source_training_data_sha256=source_training_data_sha256,
                declared_base_model=declared_base, finetune_mode=finetune_mode)


def scalar_metrics(summary, prefix='eval'):
    """Scalar metrics only; responses, prompts and answer keys stay in raw files."""
    paths = [
        ('questions',), ('responses',), ('generated_tokens',), ('mean_tokens',),
        ('median_tokens',), ('p90_tokens',), ('truncation_fraction',),
        ('max_prompt_plus_response_tokens',), ('responses_over_8192_total',),
        ('strict', 'pass1'), ('strict', 'pass8'), ('strict', 'eligible_pair_fraction'),
        ('relaxed', 'pass1'), ('relaxed', 'pass8'), ('relaxed', 'eligible_pair_fraction'),
        ('syntax', 'parse_valid_responses'), ('syntax', 'step_eligible_responses'),
        ('syntax', 'mapping_verified_responses'),
    ]
    result = {}
    for path in paths:
        value = summary
        for key in path:
            value = value.get(key) if isinstance(value, dict) else None
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            result[prefix + '/' + '/'.join(path)] = value
    return result


def _start_wandb(args, out):
    """Create an independent metrics-only run, never resume the training run."""
    if args.wandb_mode == 'disabled':
        return None
    spec = _read_json(out / 'eval_spec.json')
    checkpoint = spec.get('checkpoint') or {}
    source_args, contract = {}, {}
    if checkpoint:
        metadata_path = Path(checkpoint['path']).parent / 'started.json'
        if metadata_path.is_file():
            metadata = _read_json(metadata_path)
            source_args, contract = metadata.get('args', {}), metadata.get('training_contract', {})
    method = contract.get('arm') or source_args.get('arm') or ('base' if not checkpoint else 'unknown')
    config = dict(
        thinking_mode=THINKING_MODE,
        evaluation_only=True, model=args.model, finetune_mode=spec['finetune_mode'],
        method=method, epoch=args.epoch, training_step=args.training_step,
        evaluation_seed=args.seed, rollouts_per_question=args.rollouts,
        questions=spec['data']['evaluated_questions'], data_sha256=spec['data']['sha256'],
        train_data_sha256=spec['train_data']['sha256'], context_limit=CONTEXT_LIMIT,
        checkpoint_name=Path(checkpoint['path']).name if checkpoint else None,
        checkpoint_manifest_sha256=checkpoint.get('manifest_sha256'),
        model_content_sha256=spec['model_identity'].get('content_sha256'),
        structural_reward=0, diversity_scope=spec['diversity_scope'],
    )
    run = None
    try:
        import wandb
        # Explicit identity prevents inherited WANDB_RUN_ID/WANDB_RESUME from
        # attaching an evaluator to an active optimizer run.
        run = wandb.init(
            project=args.wandb_project, entity=args.wandb_entity, group=args.wandb_group,
            name=args.wandb_name or f'{args.model}-{spec["finetune_mode"]}-{method}-eval',
            id=uuid.uuid4().hex, resume='never', reinit=True, job_type='evaluation',
            config=config, dir=str(out), mode=args.wandb_mode, save_code=False,
            settings=wandb.Settings(console='off', disable_code=True, disable_git=True),
        )
        if run is None:
            raise RuntimeError('W&B did not create an evaluation run')
        if args.training_step is not None:
            run.define_metric('training_step')
            run.define_metric('eval/*', step_metric='training_step')
        atomic_json(out / 'wandb.json', dict(id=run.id, project=run.project,
                    entity=run.entity, url=run.url, mode=args.wandb_mode,
                    group=args.wandb_group, training_step=args.training_step, epoch=args.epoch))
        return run
    except Exception as exc:
        if run is not None:
            try:
                run.finish(exit_code=1)
            except Exception:
                pass
        # SDK errors can contain credential-bearing URLs. Persist only the
        # exception class and a fixed instruction, never the original message.
        raise RuntimeError(f'W&B evaluation initialization failed ({type(exc).__name__}); '
                           'check WANDB_API_KEY and project/entity access, or use '
                           '--wandb-mode offline/disabled.') from None


def _finish_wandb(run, summary, args):
    if run is None:
        return
    try:
        metrics = scalar_metrics(summary)
        if args.training_step is not None:
            metrics['training_step'] = args.training_step
        if args.epoch is not None:
            metrics['epoch'] = args.epoch
        run.log(metrics)
        run.finish()
    except Exception as exc:
        raise RuntimeError(f'W&B evaluation logging failed ({type(exc).__name__}); '
                           'evaluation results remain in the local output directory.') from None


def _default_model_info(args):
    from runtime_config import resolve_model_path, model_identity
    path = resolve_model_path(args.model_path, model_label=args.model)
    identity = model_identity(path)
    manifest_path = path / 'model_manifest.json'
    if manifest_path.is_file():
        manifest = _read_json(manifest_path)
        if manifest.get('alias') is not None and manifest['alias'] != args.model:
            raise ValueError('Downloaded model alias differs from --model')
        if manifest.get('repo_id') is not None and manifest['repo_id'] != MODEL_REPOS[args.model]:
            raise ValueError('Downloaded model repository differs from --model')
    return path, identity


def prepare_run(args, *, model_info_fn=None):
    """Read and validate all declared input files before starting any sampler."""
    # Check the entire held-out file, including rows beyond --limit.
    data_sha256 = args.data_sha256 or file_sha256(args.data)
    train_sha256 = args.train_data_sha256 or file_sha256(args.train_data)
    all_rows = load_data(args.data, data_sha256)
    training_rows = load_data(args.train_data, train_sha256)
    overlap = check_overlap(all_rows, training_rows)
    if args.limit is not None and args.limit > len(all_rows):
        raise ValueError('Evaluation limit exceeds dataset size')
    rows = all_rows[:args.limit] if args.limit is not None else all_rows
    if args.seed + len(rows) - 1 > 2**63 - 1:
        raise ValueError('Evaluation per-question seeds exceed the supported range')
    model_path, model = (model_info_fn or _default_model_info)(args)
    full_path = getattr(args, 'checkpoint', None)
    selected_path = full_path or args.adapter
    if full_path and args.adapter:
        raise ValueError('Choose either --checkpoint or --adapter')
    checkpoint = checkpoint_identity(selected_path, model_identity=model,
                                     model_alias=args.model, model_path=model_path,
                                     finetune_mode='full' if full_path else 'lora') if selected_path else None
    if checkpoint and checkpoint['source_training_data_sha256'] is not None and checkpoint['source_training_data_sha256'] != train_sha256:
        raise ValueError('--train-data differs from the checkpoint training dataset hash')
    devices = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if len(devices) != 1 or not devices[0].strip():
        raise ValueError('Evaluation requires exactly one allocated GPU in CUDA_VISIBLE_DEVICES')
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    spec = dict(
        version=2, thinking_mode=THINKING_MODE, evaluation_only=True, started=time.time(), model=args.model,
        model_path=str(model_path), model_identity=model, checkpoint=checkpoint,
        finetune_mode=checkpoint['finetune_mode'] if checkpoint else 'base',
        data=dict(path=str(Path(args.data).resolve()), sha256=data_sha256,
                  total_questions=len(all_rows), evaluated_questions=len(rows), prefix_limit=args.limit),
        train_data=dict(path=str(Path(args.train_data).resolve()), sha256=train_sha256),
        overlap=overlap, evaluation_seed=args.seed, sampler_engine_seed=args.seed % 2**32,
        python_hash_seed=args.seed % 2**32, rollouts_per_question=args.rollouts,
        prompt_batch=args.batch_size, context_limit=CONTEXT_LIMIT, temperature=1.0,
        sampler_max_num_seqs=args.sampler_max_num_seqs, sampler_eager=args.sampler_eager,
        generation_timeout_seconds=args.generation_timeout,
        allocated_devices=devices, grader='existing strict and relaxed signed-integer answer graders',
        diversity_scope='lexical final-explanation diagnostics only; no verified reasoning diversity',
        structural_reward=0, slurm_job_id=os.environ.get('SLURM_JOB_ID'),
        training_step=args.training_step, epoch=args.epoch,
        wandb=dict(project=args.wandb_project, entity=args.wandb_entity,
                   group=args.wandb_group, mode=args.wandb_mode, name=args.wandb_name),
        code_sha256={p.relative_to(Path(__file__).parent).as_posix(): file_sha256(p)
                     for p in sorted(Path(__file__).parent.rglob('*.py'))},
    )
    _write_once(out / 'eval_spec.json', spec)
    return rows, out, model_path, model, checkpoint, devices[0]


def _verify_checkpoint(identity, model, args):
    if identity is not None and checkpoint_identity(
            identity['path'], model_identity=model, model_alias=args.model,
            model_path=args.model_path, finetune_mode=identity.get('finetune_mode', 'lora')) != identity:
        raise ValueError('Pinned checkpoint changed during evaluation')


def _validate_generated(generated, request, rollouts):
    if not isinstance(generated, dict) or generated.get('ok') is not True:
        raise ValueError('Sampler did not return a successful response')
    if generated.get('request_id') != request['request_id'] or generated.get('context_limit') != CONTEXT_LIMIT:
        raise ValueError('Sampler response identity/context differs from request')
    rows = generated.get('rows')
    if not isinstance(rows, list) or len(rows) != len(request['prompts']):
        raise ValueError('Sampler question budget changed')
    for row, prompt in zip(rows, request['prompts']):
        if type(row.get('id')) is not type(prompt['id']) or row.get('id') != prompt['id']:
            raise ValueError('Sampler question identities/order changed')
        if row.get('prompt_token_ids') != prompt['prompt_token_ids']:
            raise ValueError('Sampler changed original prompt tokens')
        samples = row.get('samples')
        if not isinstance(samples, list) or len(samples) != rollouts or [s.get('sample_index') for s in samples] != list(range(rollouts)):
            raise ValueError('Sampler rollout count/order changed')
        for sample in samples:
            ids = sample.get('token_ids')
            if not isinstance(ids, list) or not ids or any(isinstance(t, bool) or not isinstance(t, int) or t < 0 for t in ids):
                raise ValueError('Sampler returned invalid response tokens')
            if len(prompt['prompt_token_ids']) + len(ids) > CONTEXT_LIMIT:
                raise ValueError('Sampler exceeded the 8K total context')
            if not isinstance(sample.get('text'), str):
                raise ValueError('Sampler returned invalid decoded text')


def score_batches(tok, rows, client, out, identity, model, args,
                  *, request_fn, annotate_fn):
    all_groups = []
    for start in range(0, len(rows), args.batch_size):
        _verify_checkpoint(identity, model, args)
        batch = rows[start:start + args.batch_size]
        lora = identity is not None and identity.get('finetune_mode', 'lora') == 'lora'
        request = request_fn(tok, batch, f'eval_{start:06d}',
                             identity['path'] if lora else None, 1 if lora else 0,
                             args.seed + start, cap=CONTEXT_LIMIT, n=args.rollouts)
        if identity is not None and not lora:
            request.update(model_checkpoint=identity['path'], model_version=1)
        generated = client.generate(request, timeout=args.generation_timeout)
        # Keep expensive raw evidence even if a later validation fails.
        _write_once(out / f'sampled_{start:06d}.json', generated)
        _validate_generated(generated, request, args.rollouts)
        groups = annotate_fn(tok, batch, generated)
        if len(groups) != len(batch) or [g['id'] for g in groups] != [r['id'] for r in batch]:
            raise ValueError('Scorer question identities/order changed')
        saved = []
        for group, raw in zip(groups, generated['rows']):
            records = group['records']
            if group['prompt_token_ids'] != raw['prompt_token_ids']:
                raise ValueError('Scorer changed original prompt tokens')
            if len(records) != args.rollouts or [r['sample_index'] for r in records] != list(range(args.rollouts)):
                raise ValueError('Scorer rollout count/order changed')
            for record, sample in zip(records, raw['samples']):
                if record['token_ids'] != sample['token_ids'] or record['text'] != sample['text']:
                    raise ValueError('Scorer changed original response tokens/text')
            saved.append({key: value for key, value in group.items()
                          if key not in {'advantages', 'features'}})
        _write_once(out / f'groups_{start:06d}.json', saved)
        all_groups.extend(saved)
        atomic_json(out / 'progress.json', dict(
            completed_questions=len(all_groups), expected_questions=len(rows),
            generated_tokens=sum(len(r['token_ids']) for g in all_groups for r in g['records']),
            updated=time.time()))
    _verify_checkpoint(identity, model, args)
    return all_groups


def _runtime_components():
    from transformers import AutoTokenizer
    from trainer import request, annotate_groups
    from sampler_worker import SamplerClient
    from evaluate import summarize
    return AutoTokenizer.from_pretrained, SamplerClient.launch, request, annotate_groups, summarize


def run_evaluation(args, *, model_info_fn=None, components_fn=None):
    rows, out, model_path, model, identity, device = prepare_run(args, model_info_fn=model_info_fn)
    client = None
    wandb_run = None
    began = time.monotonic()
    try:
        wandb_run = _start_wandb(args, out)
        loader, launcher, request_fn, annotate_fn, summarize_fn = (components_fn or _runtime_components)()
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        tok = loader(str(model_path), local_files_only=True, use_fast=True)
        finetune_mode = identity['finetune_mode'] if identity else 'full'
        # Full evaluation loads the requested weights directly; there is no
        # unnecessary initial base-engine load or per-batch worker restart.
        sampler_model_path = identity['path'] if identity and finetune_mode == 'full' else str(model_path)
        client = launcher(sampler_model_path, out / 'ipc', sampler_device=device,
                          max_num_seqs=args.sampler_max_num_seqs, eager=args.sampler_eager,
                          engine_seed=args.seed % 2**32, finetune_mode=finetune_mode,
                          tokenizer_path=str(model_path),
                          model_version=1 if identity and finetune_mode == 'full' else 0)
        groups = score_batches(tok, rows, client, out, identity, model, args,
                               request_fn=request_fn, annotate_fn=annotate_fn)
        summary = summarize_fn(groups)
        if summary['questions'] != len(rows) or summary['responses'] != len(rows) * args.rollouts:
            raise ValueError('Wrong final evaluation sample budget')
        if args.rollouts < 8 and any(summary[kind]['pass8'] is not None for kind in ('strict', 'relaxed')):
            raise ValueError('Pass@8 must be unavailable with fewer than eight rollouts')
        _write_once(out / 'summary.json', summary)
        client.close()
        client = None
        _finish_wandb(wandb_run, summary, args)
        wandb_run = None
        _write_once(out / 'done.json', dict(
            complete=True, ended=time.time(), wall_seconds=time.monotonic() - began,
            questions=len(rows), responses=len(rows) * args.rollouts,
            eval_spec_sha256=file_sha256(out / 'eval_spec.json'),
            summary_sha256=file_sha256(out / 'summary.json'),
            checkpoint=identity, evaluation_only=True,
        ))
        print(json.dumps(dict(complete=True, output=str(out), metrics=scalar_metrics(summary))), flush=True)
        return summary
    except BaseException as exc:
        _write_once(out / 'failed.json', dict(complete=False, error_type=type(exc).__name__,
                    error=str(exc), ended=time.time()))
        raise
    finally:
        if wandb_run is not None:
            try:
                wandb_run.finish(exit_code=1)
            except Exception as exc:
                atomic_json(out / 'wandb_cleanup_failed.json', dict(error_type=type(exc).__name__))
        if client is not None:
            try:
                client.close()
            except Exception as exc:
                _write_once(out / 'sampler_cleanup_failed.json', dict(error_type=type(exc).__name__))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.set_defaults(thinking_mode=THINKING_MODE)
    p.add_argument('--model', required=True, choices=list(MODEL_REPOS))
    p.add_argument('--model-path', required=True, help='Local downloaded model directory')
    policy = p.add_mutually_exclusive_group()
    policy.add_argument('--adapter', help='Pinned LoRA adapter directory with its sibling manifest')
    policy.add_argument('--checkpoint', help='Pinned full-model directory with its sibling manifest; omit both options for base evaluation')
    p.add_argument('--data', required=True, help='Held-out JSON array: id, question, integer ground_truth')
    p.add_argument('--data-sha256', help='Optional expected hash; actual hash is always recorded')
    p.add_argument('--train-data', required=True, help='Training JSON for mandatory overlap checking')
    p.add_argument('--train-data-sha256', help='Optional expected training-data hash')
    p.add_argument('--out', required=True, help='Fresh output directory; existing directories are rejected')
    p.add_argument('--limit', type=int, help='Evaluate the first N rows; still check overlap on the entire file')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--rollouts', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=16, help='Questions per request, independent of K')
    p.add_argument('--sampler-max-num-seqs', type=int, choices=[16, 32, 64], default=16)
    p.add_argument('--generation-timeout', type=int, default=3600, help='Maximum seconds for each sampler request')
    p.add_argument('--sampler-eager', action='store_true', help='Disable vLLM compilation/decode graphs')
    p.add_argument('--training-step', type=int, help='Optimizer step of this pinned checkpoint, for W&B comparison')
    p.add_argument('--epoch', type=int, help='Completed epoch of this pinned checkpoint')
    p.add_argument('--wandb-project', default=os.environ.get('WANDB_PROJECT', 'stride-iclr2027'))
    p.add_argument('--wandb-entity', default=os.environ.get('WANDB_ENTITY') or None)
    p.add_argument('--wandb-group', default=os.environ.get('WANDB_RUN_GROUP', 'stride-comparison'))
    p.add_argument('--wandb-name', help='Optional display name; evaluation always gets a fresh run ID')
    p.add_argument('--wandb-mode', choices=['online', 'offline', 'disabled'],
                   default=os.environ.get('WANDB_MODE', 'online'))
    args = p.parse_args(argv)
    if not 0 <= args.seed <= 2**63 - 1:
        p.error('--seed must be in [0, 2**63-1]')
    if not 1 <= args.rollouts <= 64:
        p.error('--rollouts must be in [1, 64]')
    if args.generation_timeout < 1:
        p.error('--generation-timeout must be positive')
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        p.error('--batch-size and --limit must be positive')
    if args.training_step is not None and args.training_step < 0:
        p.error('--training-step must be nonnegative')
    if args.epoch is not None and args.epoch < 1:
        p.error('--epoch must be positive')
    if args.wandb_mode not in {'online', 'offline', 'disabled'}:
        p.error('WANDB_MODE must be online, offline or disabled')
    if args.wandb_mode != 'disabled' and not args.wandb_project.strip():
        p.error('--wandb-project must be nonempty when W&B is enabled')
    for value in (args.data_sha256, args.train_data_sha256):
        if value is not None and not re.fullmatch(r'[a-f0-9]{64}', value):
            p.error('SHA256 values must be 64 lowercase hexadecimal characters')
    return args


def main(argv=None):
    return run_evaluation(parse_args(argv))


if __name__ == '__main__':
    main()
