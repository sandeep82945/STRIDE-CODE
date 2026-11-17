#!/usr/bin/env python3
"""Independent stdlib gate for the 2048-question, four-epoch LoRA runs.

Run with host Python -I -S; a native process exit code is insufficient evidence.
Final verification reads every rollout, all 129 retained adapters and original
commit receipts, plus the final full optimizer/RNG state (about 17 GB per arm).
It deliberately performs this once at completion, not after every update.
Tensor payloads are authenticated but not deserialized; successful integrated
smoke supplies the actual PEFT/vLLM load-and-update evidence.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import re
import sys

TRAINING = Path(__file__).resolve().parents[1] / 'training'
ARMS = {'local_positive', 'single_step', 'grpo', 'token_uniform'}
MODEL_KEY = 'qwen17b'
ALPHA = 1.0
EXPECTED_TRAIN_SHA256 = 'd319351d80df073e947576189f41c23b549584ecca4bad660a2b752427f429cc'
QUESTIONS, BATCH, EPOCHS, WORLD, K, STEPS = 2048, 64, 4, 4, 8, 128
LORA_TARGETS = {'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    path = Path(path)
    need(path.is_file() and not path.is_symlink(), 'Missing or unsafe evidence file: ' + path.name)
    def invalid(_):
        raise ValueError('Nonfinite JSON evidence: ' + path.name)
    return json.loads(path.read_text(), parse_constant=invalid)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def checkpoint_module():
    spec = importlib.util.spec_from_file_location('camd_full_checkpoint_audit', TRAINING / 'resumable_checkpoint.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def common(output, kind):
    started, done = read(output / 'started.json'), read(output / 'done.json')
    args = started['args']
    expected = dict(mode=kind, model=MODEL_KEY, finetune_mode='lora', seed=42,
                    batch_size=BATCH, epochs=EPOCHS, alpha=ALPHA, learning_rate=2e-5, kl_coefficient=0., warmup_steps=10,
                    novelty_reference_scope='all_rollouts', validation_mode='off')
    need(all(args.get(k) == v for k, v in expected.items()) and args.get('arm') in ARMS,
         'Recorded settings differ from the stabilized LoRA family experiment')
    need(args.get('thinking_mode') is False, 'Explicit nonthinking arguments are required')
    need(done.get('complete') is True and done.get('finetune_mode') == 'lora'
         and not (output / 'failed.json').exists(), 'Run did not finish successfully')
    need(started.get('world_size') == WORLD and started.get('context_limit') == 8192,
         'Unexpected learner count or context limit')
    hashes = {p.relative_to(TRAINING).as_posix(): sha(p) for p in TRAINING.rglob('*.py')}
    need(started.get('code_sha256') == hashes, 'Output source hashes differ from this handoff')
    data = Path(args['data'])
    rows = read(data)
    need(isinstance(rows, list) and len(rows) == QUESTIONS, 'Full run requires exactly 2048 questions')
    need(sha(data) == args.get('data_sha256') == started.get('data_sha256') == EXPECTED_TRAIN_SHA256,
         'Dataset hash differs from the frozen 2048-question training split')
    ids = [str(row['id']).strip() for row in rows]
    need(all(ids) and len(set(ids)) == QUESTIONS, 'Training question IDs are missing or duplicated')
    return started, done, args, ids, hashes


def policy_config(path):
    config = read(path / 'adapter_config.json')
    need(config.get('peft_type') == 'LORA' and config.get('task_type') == 'CAUSAL_LM'
         and config.get('r') == 16 and config.get('lora_alpha') == 32
         and config.get('lora_dropout') == 0 and set(config.get('target_modules') or ()) == LORA_TARGETS,
         'Adapter configuration differs from the required reloadable LoRA architecture')


def smoke(output):
    _, done, args, _, _ = common(output, 'smoke')
    need(done.get('steps') == 1 and args.get('smoke_cap') == 128 and args.get('rollouts') == 2
         and args.get('sampler_eager') is True, 'Unexpected integrated smoke budget')
    update = read(output / 'update_001.json')
    need(finite(update.get('actor_behavior_abs_logp_mean')) and 0 <= update['actor_behavior_abs_logp_mean'] < .05,
         'Smoke actor/behavior log probabilities disagree')
    checks = [read(output / f'rank_{rank}/smoke_checks.json') for rank in range(WORLD)]
    for item in checks:
        need(finite(item.get('adapter_changed_logps')) and item['adapter_changed_logps'] > 1e-5
             and item.get('fixed_reference_maxdiff') == 0
             and finite(item.get('fresh_adapter_abs_logp_mean')) and 0 <= item['fresh_adapter_abs_logp_mean'] < .05
             and item.get('all_rank_trainable_weights_equal') is True, 'A rank failed the integrated smoke checks')
    fingerprints = {item.get('trainable_sha256') for item in checks}
    need(len(fingerprints) == 1 and re.fullmatch('[0-9a-f]{64}', next(iter(fingerprints)) or ''),
         'Learner weights differ or have invalid fingerprints')
    checkpoint_module().verify_policy(output / 'adapter_001', 'lora')
    policy_config(output / 'adapter_001')
    need(Path(done['final_adapter']).resolve() == output / 'adapter_001', 'Smoke final adapter path mismatch')
    return dict(arm=args['arm'], steps=1, cap=128, rollouts=2, sampler_eager=True,
                synthetic_update_only=True, integrated_adapter_reload_checked=True)


def group_ids(path):
    values = read(path)
    need(isinstance(values, list) and len(values) == BATCH // WORLD, 'Wrong per-rank question count')
    ids, tokens = [], 0
    for group in values:
        records = group['records']
        need(len(records) == K and [r['sample_index'] for r in records] == list(range(K)),
             'Missing, repeated or reordered generated responses')
        need(all(isinstance(r['token_ids'], list) and r['token_ids'] for r in records), 'Empty generated response')
        ids.append(str(group['id']).strip())
        tokens += sum(len(r['token_ids']) for r in records)
    return ids, tokens


def train(output):
    started, done, args, ids, hashes = common(output, 'train')
    contract = started['training_contract']
    expected = dict(version=7, mode='train', model=MODEL_KEY, finetune_mode='lora', arm=args['arm'],
        batch_schedule='epoch_shuffle_keep_partial', collective_padding='zero_loss_no_generated_samples',
        updates_per_epoch=32, training_questions=QUESTIONS, batch_size=BATCH, epochs=EPOCHS, world_size=WORLD,
        K=K, seed=42, alpha=ALPHA, context_limit=8192, format_coefficient=0., warmup_steps=10, kl_coefficient=0.,
        novelty_reference_scope='all_rollouts', learning_rate_schedule='linear_warmup_then_constant',
        kl_reference=None, kl_estimator=None, kl_normalization=None)
    need(all(contract.get(k) == v for k, v in expected.items()) and args.get('rollouts') == K,
         'Training contract does not describe the complete 2048-question four-epoch experiment')
    need(contract.get('thinking_mode') is False, 'Explicit nonthinking training contract is required')
    need(contract.get('data_sha256') == started['data_sha256'] and contract.get('code_sha256')
         and all(hashes.get(k) == v for k, v in contract['code_sha256'].items()), 'Training contract provenance mismatch')
    need(args.get('keep_all_checkpoints') is True
         and started.get('checkpoint_retention') == done.get('checkpoint_retention') == 'all',
         'Full experiment must retain every checkpoint')
    need(args.get('resume_state_retention') == started.get('resume_state_retention')
         == done.get('resume_state_retention') == 'latest-epoch', 'Resume states must retain only latest epoch/update')
    need(done.get('steps') == STEPS and done.get('epochs') == EPOCHS and done.get('start_step') == 0
         and started.get('start_step') == 0 and started.get('resume_from') is None,
         'Incomplete or unexpected resumed schedule; resumed attempts need ancestry-aware verification')
    for prefix in ('update', 'checkpoint'):
        need({p.name for p in output.glob(prefix + '_*.json')} == {f'{prefix}_{i:03d}.json' for i in range(1, STEPS + 1)},
             'Missing or unexpected ' + prefix + ' records')
    need({p.name for p in output.glob('state_*')} == {f'state_{STEPS:03d}'},
         'Missing or unexpected retained optimizer/RNG checkpoints')
    need({p.name for p in output.glob('adapter_*') if p.is_dir()} == {f'adapter_{i:03d}' for i in range(STEPS + 1)},
         'Missing or unexpected retained adapters')
    receipts = output / 'checkpoint_receipts'
    need(receipts.is_dir() and not receipts.is_symlink()
         and {p.name for p in receipts.iterdir()} == {f'state_{i:03d}' for i in range(STEPS + 1)},
         'Missing or unexpected checkpoint commit receipts')
    for rank in range(WORLD):
        need({p.name for p in (output / f'rank_{rank}').glob('rollouts_*.json')}
             == {f'rollouts_{i:03d}.json' for i in range(STEPS)}, 'Missing or unexpected rank rollout records')
    active, zero, total_tokens = 0, 0, 0
    for epoch in range(EPOCHS):
        order = list(ids)
        random.Random(42 + epoch).shuffle(order)
        for within in range(32):
            step = epoch * 32 + within + 1
            update = read(output / f'update_{step:03d}.json')
            need(update.get('step') == step and update.get('mode') == 'train' and update.get('arm') == args['arm']
                 and update.get('finetune_mode') == 'lora', 'Optimizer update identity mismatch')
            expected_lr = 2e-5 * min(step / 10, 1.)
            need(finite(update.get('learning_rate')) and abs(update['learning_rate'] - expected_lr) < 1e-12,
                 'Logged learning rate differs from the warmup schedule')
            need(update.get('kl_coefficient') == 0 and update.get('kl_loss') == 0,
                 'KL must remain disabled for this experiment')
            norm, totals = update.get('gradient_norm'), update['totals']
            need(finite(norm) and norm >= 0 and finite(update.get('loss')), 'Nonfinite loss or invalid gradient norm')
            need(update.get('rollout_count') == totals.get('responses') == BATCH * K and totals.get('groups') == BATCH,
                 'Generated responses do not match the 64-question/K8 budget')
            need(update.get('collective_padding_forwards') == 0, 'Unexpected collective padding budget')
            signal = totals.get('nonzero_advantage_tokens')
            need(type(signal) is int and signal >= 0, 'Invalid advantage signal count')
            active += norm > 0 and signal > 0
            zero += norm == 0
            actual, response_tokens = [], 0
            for rank in range(WORLD):
                rank_ids, rank_tokens = group_ids(output / f'rank_{rank}/rollouts_{step-1:03d}.json')
                actual.extend(rank_ids)
                response_tokens += rank_tokens
            need(actual == order[within * BATCH:(within + 1) * BATCH], 'Dropped, repeated or wrong-epoch training questions')
            need(update.get('response_tokens') == totals.get('response_tokens') == response_tokens
                 and signal <= response_tokens, 'Training token counts disagree with recorded rollouts')
            saved = read(output / f'checkpoint_{step:03d}.json')
            prior_keep = {step - 1, (step - 1) // 32 * 32}
            current_keep = {step, step // 32 * 32}
            need(saved.get('completed_step') == step and saved.get('checkpoint_retention') == 'all'
                 and saved.get('resume_state_retention') == 'latest-epoch'
                 and saved.get('pruned_state_steps') == sorted(prior_keep - current_keep)
                 and saved.get('pruned_steps') == []
                 and Path(saved.get('state_checkpoint', '')).resolve() == output / f'state_{step:03d}',
                 'Checkpoint retention metadata or path mismatch')
            total_tokens += response_tokens
    need(active > 0, 'No genuine update with nonzero gradient and advantage signal')
    cp = checkpoint_module()
    for step in range(STEPS + 1):
        cp.inspect_checkpoint_receipt(output, step, contract, WORLD)
        policy_config(output / f'adapter_{step:03d}')
    final = cp.inspect_checkpoint(output / f'state_{STEPS:03d}', contract, WORLD)
    need(Path(done['final_state_checkpoint']).resolve() == output / f'state_{STEPS:03d}'
         and Path(done['final_adapter']).resolve() == Path(final['adapter_path']) == output / f'adapter_{STEPS:03d}',
         'Final optimizer state or adapter path mismatch')
    return dict(arm=args['arm'], epochs=EPOCHS, steps=STEPS, training_questions=QUESTIONS,
                real_responses=QUESTIONS * EPOCHS * K, response_tokens=total_tokens,
                nonzero_updates=active, zero_gradient_updates=zero,
                epoch_steps=list(range(32, STEPS + 1, 32)), retained_checkpoint_pairs=1,
                retained_adapter_checkpoints=STEPS + 1, archived_commit_receipts=STEPS + 1,
                final_adapter=final['adapter_path'], final_state_checkpoint=final['state_path'],
                checkpoint_retention='all', resume_state_retention='latest-epoch', optimizer_state_hashes_verified=True,
                optimizer_tensors_deserialized=False, adapter_configuration_verified=True,
                limitations='Filesystem/provenance verification; no convergence or quality claim.')


def verify(kind, output):
    output = Path(output)
    need(output.is_dir() and not output.is_symlink(), 'Output must be a regular existing directory')
    need(kind in ('smoke', 'train'), 'Only smoke and training are supported')
    output = output.resolve()
    result = smoke(output) if kind == 'smoke' else train(output)
    return dict(passed=True, kind=kind, output=str(output), **result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=('smoke', 'train'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--report', type=Path, help='New report file; existing reports are never overwritten')
    args = parser.parse_args(argv)
    try:
        result = verify(args.kind, args.output)
    except Exception as exc:
        result = dict(passed=False, kind=args.kind, error_type=type(exc).__name__, error=str(exc))
    text = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.report:
        try:
            with args.report.open('x') as stream:
                stream.write(text)
        except OSError:
            print('Unable to create fresh verification report', file=sys.stderr)
            return 1
    print(text, end='')
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
