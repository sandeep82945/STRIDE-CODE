"""Portable training configuration and content identity; no GPU imports."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import unicodedata


MODEL_LABELS = ('qwen4b', 'qwen17b')
ARMS = ('local_positive', 'single_step', 'token_uniform', 'grpo')
FINETUNE_MODES = ('full', 'lora')
DEFAULT_LEARNING_RATES = {'full': 1e-6, 'lora': 1e-4}
CONTEXT_LIMIT = 8192
THINKING_MODE = False  # Both replacement model families use nonthinking prompts.


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def add_training_arguments(parser, *, resumable=False):
    parser.set_defaults(thinking_mode=THINKING_MODE)
    parser.add_argument('mode', choices=('train',) if resumable else ('smoke', 'train'))
    parser.add_argument('--model', default='qwen4b', choices=MODEL_LABELS, help='Experiment label; never selects or downloads weights')
    parser.add_argument('--model-path', required=True, help='Prepared local Hugging Face model directory')
    parser.add_argument('--arm', default='local_positive', choices=ARMS)
    parser.add_argument('--finetune-mode', choices=FINETUNE_MODES, default='full',
                        help='full: train all policy parameters with FSDP; lora: train adapters with DDP')
    if resumable:
        parser.add_argument('--resume-from', help='Verified state_NNN directory in a prior attempt')
        parser.add_argument('--allow-epoch-extension', action='store_true',
                            help='Allow a verified resume with a strictly larger epoch target; every other scientific setting must match')
        parser.add_argument('--keep-all-checkpoints', action='store_true',
                            help='Retain every committed policy and optimizer/RNG checkpoint, including step zero')
        parser.add_argument('--resume-state-retention', choices=('all', 'latest-epoch'), default='all',
                            help='With all LoRA adapters retained, keep all resume states or only latest completed epoch plus latest update')
        parser.add_argument('--learner-microbatch-size', type=int, choices=(1, 2), default=1,
                            help='LoRA learner responses per forward: 1 legacy order; 2 length-sorted padded pairs')
    parser.add_argument('--out', required=True, help='Empty output attempt directory')
    parser.add_argument('--data', required=True, help='Prepared JSON list of training questions')
    parser.add_argument('--data-sha256', required=True, help='Expected SHA256 from data preparation')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--rollouts', type=int, default=8, help='Responses per question (K), including smoke')
    parser.add_argument('--alpha', type=float, default=1., help='Constant diversity coefficient from the first update')
    parser.add_argument('--novelty-reference-scope', choices=('correct_only', 'all_rollouts'), default='correct_only',
                        help='Rollouts whose eligible steps define the STRIDE novelty reference set')
    parser.add_argument('--learning-rate', type=float, default=None,
                        help='AdamW learning rate; defaults to 1e-6 for full and 1e-4 for LoRA (not tuned)')
    parser.add_argument('--warmup-steps', type=int, default=0,
                        help='Linear LR warmup in optimizer updates, then constant target LR')
    parser.add_argument('--kl-coefficient', type=float, default=0.,
                        help='Original GRPO sampled k3 loss coefficient against fixed initial base (LoRA only)')
    parser.add_argument('--wandb-project', default=os.environ.get('WANDB_PROJECT', 'stride-iclr2027'))
    parser.add_argument('--wandb-entity', default=os.environ.get('WANDB_ENTITY') or None)
    parser.add_argument('--wandb-group', default=os.environ.get('WANDB_RUN_GROUP', 'stride-comparison'))
    parser.add_argument('--wandb-mode', choices=('online', 'offline', 'disabled'), default=None)
    parser.add_argument('--validation-mode', choices=('slurm', 'off'), default=None,
                        help='Default: submit separate one-GPU validation jobs after each training epoch')
    parser.add_argument('--validation-data', default=os.environ.get('VALIDATION_DATA', 'data/val.json'))
    parser.add_argument('--validation-every-epochs', type=int, default=1)
    parser.add_argument('--validation-rollouts', type=int, default=8)
    parser.add_argument('--validation-partition', default=None)
    parser.add_argument('--validation-account', default=None)
    parser.add_argument('--validation-qos', default=None)
    parser.add_argument('--smoke-cap', type=int, default=128, help='Smoke-only response cap; prompt plus response remains <=8192')
    parser.add_argument('--sampler-eager', action='store_true')
    parser.add_argument('--sampler-max-num-seqs', type=int, default=32)
    parser.add_argument('--generation-timeout', type=int, default=3600)


def validate_options(args, *, world_size=4, resumable=False):
    if getattr(args, "thinking_mode", THINKING_MODE) is not THINKING_MODE:
        raise ValueError("This experiment requires thinking_mode=False")
    args.thinking_mode = THINKING_MODE
    if args.wandb_mode is None:
        args.wandb_mode = os.environ.get('WANDB_MODE', 'disabled' if args.mode == 'smoke' else 'online')
    if args.validation_mode is None:
        args.validation_mode = os.environ.get('VALIDATION_MODE', 'off' if args.mode == 'smoke' else 'slurm')
    if args.wandb_mode not in ('online', 'offline', 'disabled'):
        raise ValueError('W&B mode must be online, offline, or disabled')
    if args.validation_mode not in ('slurm', 'off'):
        raise ValueError('Validation mode must be slurm or off')
    if args.mode == 'smoke':
        args.validation_mode = 'off'
    if args.validation_every_epochs < 1 or not 1 <= args.validation_rollouts <= 64:
        raise ValueError('Validation cadence must be positive; rollouts must be 1..64')
    if args.validation_mode != 'off' and not args.validation_data:
        raise ValueError('Enabled validation requires --validation-data')
    if args.finetune_mode not in FINETUNE_MODES:
        raise ValueError('Choose full or lora fine-tuning')
    microbatch = getattr(args, 'learner_microbatch_size', 1)
    if microbatch not in (1, 2) or (microbatch != 1 and (not resumable or args.finetune_mode != 'lora')):
        raise ValueError('Learner microbatch two requires resumable LoRA training')
    if args.learning_rate is None:
        args.learning_rate = DEFAULT_LEARNING_RATES[args.finetune_mode]
    if args.mode not in (('train',) if resumable else ('smoke', 'train')):
        raise ValueError('Only direct training and the separate smoke check are supported')
    if any(hasattr(args, name) for name in ('adapter', 'format_coefficient')):
        raise ValueError('External adapter initialization and format rewards have been removed; use --resume-from for a verified training state')
    if args.batch_size <= 0 or args.batch_size % world_size or args.epochs < 1:
        raise ValueError('Batch must be positive and divisible by four; epochs must be positive')
    if not 2 <= args.rollouts <= 64:
        raise ValueError('GRPO and STRIDE require 2..64 rollouts per question (sampler limit)')
    if not 0 <= args.seed < 2**32:
        raise ValueError('Seed must be an integer in [0, 2**32) for NumPy and the sampler schedule')
    if not math.isfinite(args.alpha) or args.alpha < 0:
        raise ValueError('Alpha must be finite and nonnegative')
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError('Learning rate must be finite and positive')
    if type(args.warmup_steps) is not int or args.warmup_steps < 0:
        raise ValueError('Warmup steps must be a nonnegative integer')
    if not math.isfinite(args.kl_coefficient) or args.kl_coefficient < 0:
        raise ValueError('KL coefficient must be finite and nonnegative')
    if args.kl_coefficient and args.finetune_mode != 'lora':
        raise ValueError('Fixed-base KL currently requires LoRA training')
    if args.sampler_max_num_seqs not in (16, 32, 64):
        raise ValueError('Sampler sequence limit must be 16, 32, or 64')
    if not 1 <= args.smoke_cap <= CONTEXT_LIMIT or args.generation_timeout < 1:
        raise ValueError('Invalid response cap or generation timeout')
    if re.fullmatch(r'[0-9a-fA-F]{64}', args.data_sha256) is None:
        raise ValueError('--data-sha256 must contain a complete SHA256 digest')
    args.data_sha256 = args.data_sha256.lower()
    resume = bool(getattr(args, 'resume_from', None))
    if getattr(args, 'allow_epoch_extension', False) and (not resumable or not resume):
        raise ValueError('Epoch extension requires a verified --resume-from checkpoint')
    retention = getattr(args, 'resume_state_retention', 'all')
    if retention not in ('all', 'latest-epoch'):
        raise ValueError('Unknown resume-state retention policy')
    if retention == 'latest-epoch' and (not resumable or args.finetune_mode != 'lora'
                                      or not getattr(args, 'keep_all_checkpoints', False)):
        raise ValueError('Latest-epoch resume-state retention requires LoRA and --keep-all-checkpoints')
    if args.mode == 'smoke' and resume:
        raise ValueError('Smoke does not support resume')
    if not resumable and resume:
        raise ValueError('Resume requires trainer_resumable.py')
    if args.wandb_mode != 'disabled' and not args.wandb_project:
        raise ValueError('Enable W&B with an explicit --wandb-project')


def resolve_model_path(value, model_label=None):
    path = Path(value).expanduser().resolve()
    if not path.is_dir() or not (path / 'config.json').is_file():
        raise ValueError(f'Expected a local model directory with config.json: {path}')
    weights = list(path.glob('*.safetensors')) + list(path.glob('pytorch_model*.bin'))
    if not weights:
        raise ValueError(f'No local model weights found: {path}; run model preparation first')
    manifest = path / 'model_manifest.json'
    if model_label and manifest.is_file():
        alias = json.loads(manifest.read_text()).get('alias')
        if alias and alias != model_label:
            raise ValueError('Model label disagrees with the downloaded model manifest')
    return path


def model_identity(model_path):
    """Hash actual model, tokenizer, and template files, not an assumed alias revision.

    Only rank zero should call this for multi-GPU startup. No content-hash cache
    is trusted, so replacing a weight file cannot silently pass a resume check.
    """
    path = resolve_model_path(model_path)
    extensions = {'.json', '.safetensors', '.bin', '.model', '.tiktoken', '.txt', '.jinja', '.jinja2'}
    files = {}
    for file in sorted(path.rglob('*')):
        relative = file.relative_to(path)
        if (not file.is_file() or any(part.startswith('.') for part in relative.parts)
                or file.name == 'model_manifest.json' or file.suffix not in extensions):
            continue
        files[relative.as_posix()] = digest(file)
    encoded = json.dumps(files, sort_keys=True, separators=(',', ':')).encode()
    manifest = path / 'model_manifest.json'
    return dict(content_sha256=hashlib.sha256(encoded).hexdigest(), files=files,
                preparation_manifest_sha256=digest(manifest) if manifest.is_file() else None)


def validate_adapter_origin(adapter, identity):
    """Reject an evaluation adapter trained on another base model or an unknown run."""
    started_path = Path(adapter).resolve().parent / 'started.json'
    if not started_path.is_file():
        raise ValueError('Adapter needs its originating started.json; copy the run provenance with its manifest')
    started = json.loads(started_path.read_text())
    previous = started.get('model_identity', started.get('training_contract', {}).get('model_identity', {}))
    if previous.get('content_sha256') != identity['content_sha256']:
        raise ValueError('Adapter was produced from a different or unverified base model')


def validate_rows(rows):
    from pilot_grading.integer_reference import _ground_truth
    if not isinstance(rows, list) or not rows:
        raise ValueError('Expected a nonempty JSON list of training rows')
    seen = set()
    questions = set()
    for row in rows:
        if not isinstance(row, dict) or not {'id', 'question', 'ground_truth'} <= row.keys():
            raise ValueError('Missing id, question, or ground_truth in a training row')
        if type(row['id']) not in (int, str):
            raise ValueError('Question IDs must be strings or integers')
        identity = str(row['id']).strip()
        if not identity or identity in seen or not isinstance(row['question'], str) or not row['question'].strip():
            raise ValueError('Duplicate IDs or empty question')
        _ground_truth(row['ground_truth'])
        question = canonical_question(row['question'])
        if not question or question in questions:
            raise ValueError('Duplicate normalized training question')
        seen.add(identity)
        questions.add(question)


def canonical_question(text):
    """Same presentation normalization as prepare_data.canonical (not semantics)."""
    value = unicodedata.normalize('NFKC', text).casefold()
    value = re.sub(r'\\(?:left|right|displaystyle)\b', '', value)
    value = re.sub(r'\\(?:quad|qquad)\b|\\[,!;: ]', '', value)
    value = value.replace(r'\dfrac', r'\frac').replace(r'\tfrac', r'\frac')
    for delimiter in (r'\(', r'\)', r'\[', r'\]', '$'):
        value = value.replace(delimiter, '')
    return ''.join(value.split())


def load_training_data(args):
    path = Path(args.data).expanduser().resolve()
    if digest(path) != args.data_sha256:
        raise ValueError('Training data SHA256 differs from the preparation manifest')
    rows = json.loads(path.read_text())
    validate_rows(rows)
    epoch_update_count(len(rows), args.batch_size)
    max_sampler_seed = 20260980 + args.seed * 1000000 + schedule_steps(args, len(rows)) * args.batch_size
    if max_sampler_seed > 2**63 - 1:
        raise ValueError('Training schedule would exceed the sampler seed range')
    return path, rows


def epoch_update_count(training_questions, batch_size, world_size=4):
    """Keep every real question; a tail must leave at least one group per rank."""
    if batch_size < world_size or batch_size % world_size or training_questions < batch_size:
        raise ValueError('Training size must contain a full global batch, divisible across learners')
    remainder = training_questions % batch_size
    if 0 < remainder < world_size:
        raise ValueError('Partial batch must contain at least one real question per learner')
    return (training_questions + batch_size - 1) // batch_size


def epoch_progress(completed_steps, training_questions, batch_size):
    """Epochs measured in actual questions, including a shorter final update."""
    if completed_steps < 0:
        raise ValueError('Negative schedule position')
    per_epoch = epoch_update_count(training_questions, batch_size)
    epochs, within = divmod(completed_steps, per_epoch)
    return epochs + min(within * batch_size, training_questions) / training_questions


def schedule_steps(args, training_questions):
    per_epoch = epoch_update_count(training_questions, args.batch_size)
    if args.mode not in ('smoke', 'train'):
        raise ValueError('Unknown training mode')
    return 1 if args.mode == 'smoke' else args.epochs * per_epoch


def scientific_contract(args, *, world_size, training_questions, identity):
    """Pure, serializable scientific settings used by strict checkpoint resume."""
    return dict(version=7, thinking_mode=THINKING_MODE, batch_schedule='epoch_shuffle_keep_partial',
                collective_padding='zero_loss_no_generated_samples',
                updates_per_epoch=epoch_update_count(training_questions, args.batch_size, world_size),
                model=args.model, model_identity=identity, arm=args.arm, mode=args.mode,
                finetune_mode=args.finetune_mode,
                distributed_backend='fsdp_full_shard' if args.finetune_mode == 'full' else 'ddp',
                policy_precision='fp32_master_bf16_compute' if args.finetune_mode == 'full' else 'bf16_base_fp32_lora',
                reference_encoder='frozen_initial_base',
                lora=dict(rank=16, alpha=32, dropout=0., targets=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                                                               'gate_proj', 'up_proj', 'down_proj']) if args.finetune_mode == 'lora' else None,
                data_sha256=args.data_sha256, training_questions=training_questions,
                batch_size=args.batch_size, epochs=args.epochs, warmup_steps=args.warmup_steps,
                learning_rate_schedule='linear_warmup_then_constant',
                kl_coefficient=args.kl_coefficient,
                kl_reference='frozen_initial_base' if args.kl_coefficient else None,
                kl_estimator='original_grpo_k3' if args.kl_coefficient else None,
                kl_normalization='global_generated_tokens' if args.kl_coefficient else None,
                learner_microbatch_size=getattr(args, 'learner_microbatch_size', 1),
                learner_sequence_packing=('length_sorted_left_pad' if getattr(args, 'learner_microbatch_size', 1) == 2
                                          else 'original_order_single'),
                seed=args.seed, world_size=world_size, K=args.rollouts, alpha=args.alpha,
                novelty_reference_scope=args.novelty_reference_scope,
                context_limit=CONTEXT_LIMIT, epoch_offset=0,
                format_coefficient=0., diversity_alpha_schedule='constant',
                sampler_eager=args.sampler_eager, sampler_max_num_seqs=args.sampler_max_num_seqs,
                sampler_seed_base=20260980, sampler_seed_stride=1000000,
                sampler_engine_seed=args.seed, python_hash_seed=args.seed,
                optimizer=dict(name='AdamW', lr=args.learning_rate if args.learning_rate is not None else DEFAULT_LEARNING_RATES[args.finetune_mode], weight_decay=0., betas=[.9, .999], eps=1e-8,
                               foreach=False),
                initialization='base_model_full' if args.finetune_mode == 'full' else 'base_model_fresh_lora')
