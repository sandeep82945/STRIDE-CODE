#!/usr/bin/env python3
"""Standard-library evidence gate for the 100-question CAMD LoRA pilot.

Run with host Python -I -S: never trust a Torch process exit status alone.
This checks recorded results and hashes, not learning quality or remote state.
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
ARMS = {'local_positive', 'token_uniform', 'single_step', 'grpo'}


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


def sources(record):
    expected = {p.relative_to(TRAINING).as_posix(): sha(p) for p in TRAINING.rglob('*.py')}
    need(record.get('code_sha256') == expected, 'Output source hashes differ from this handoff')
    return expected


def checkpoint_module():
    spec = importlib.util.spec_from_file_location('camd_checkpoint_audit', TRAINING / 'resumable_checkpoint.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # This module imports Torch only in unused load/save functions.
    return module


def data_ids(path, expected):
    need(sha(path) == expected, 'Dataset hash differs from recorded training/evaluation data')
    rows = read(path)
    need(isinstance(rows, list) and len(rows) == 100, 'Pilot requires exactly 100 questions')
    ids = [str(row['id']).strip() for row in rows]
    need(all(ids) and len(set(ids)) == 100, 'Pilot question IDs are missing or duplicated')
    return ids


def groups(path, rollouts):
    values = read(path)
    need(isinstance(values, list), 'Grouped rollout evidence must be a list')
    ids = []
    for group in values:
        records = group['records']
        need(len(records) == rollouts and [r['sample_index'] for r in records] == list(range(rollouts)),
             'Missing, repeated or reordered generated responses')
        need(all(isinstance(r['token_ids'], list) and r['token_ids'] for r in records), 'Empty generated response')
        ids.append(str(group['id']).strip())
    return ids


def common(output, kind):
    started, done = read(output / 'started.json'), read(output / 'done.json')
    args = started['args']
    expected = dict(mode=kind, model='qwen4b', finetune_mode='lora', seed=42,
                    batch_size=64, epochs=4, alpha=1., learning_rate=1e-4, validation_mode='off')
    need(all(args.get(k) == v for k, v in expected.items()) and args.get('arm') in ARMS,
         'Recorded settings differ from the Qwen4B LoRA pilot')
    need(done.get('complete') is True and done.get('finetune_mode') == 'lora', 'Run did not finish successfully')
    need(started.get('world_size') == 4 and started.get('context_limit') == 8192,
         'Unexpected learner count or context limit')
    hashes = sources(started)
    ids = data_ids(args['data'], started['data_sha256'])
    need(args.get('data_sha256') == started['data_sha256'], 'Launcher dataset identity mismatch')
    return started, done, args, ids, hashes


def smoke(output):
    _, done, args, _, _ = common(output, 'smoke')
    need(done.get('steps') == 1 and (args.get('smoke_cap'), args.get('rollouts')) in
         {(128, 2), (128, 8), (8192, 8)}, 'Unexpected smoke budget')
    update = read(output / 'update_001.json')
    need(finite(update.get('actor_behavior_abs_logp_mean')) and 0 <= update['actor_behavior_abs_logp_mean'] < .05,
         'Smoke actor/behavior log probabilities disagree')
    checks = [read(output / f'rank_{rank}/smoke_checks.json') for rank in range(4)]
    for item in checks:
        need(finite(item.get('adapter_changed_logps')) and item['adapter_changed_logps'] > 1e-5
             and item.get('fixed_reference_maxdiff') == 0
             and finite(item.get('fresh_adapter_abs_logp_mean')) and 0 <= item['fresh_adapter_abs_logp_mean'] < .05
             and item.get('all_rank_trainable_weights_equal') is True, 'A rank failed the smoke checks')
    fingerprints = {item.get('trainable_sha256') for item in checks}
    need(len(fingerprints) == 1 and re.fullmatch('[0-9a-f]{64}', next(iter(fingerprints)) or ''),
         'Learner weights differ or have invalid fingerprints')
    checkpoint_module().verify_policy(output / 'adapter_001', 'lora')
    need(Path(done['final_adapter']).resolve() == output / 'adapter_001', 'Smoke final adapter path mismatch')
    return dict(arm=args['arm'], steps=1, cap=args['smoke_cap'], rollouts=args['rollouts'],
                sampler_eager=args['sampler_eager'], synthetic_update_only=True)


def train(output):
    started, done, args, ids, hashes = common(output, 'train')
    contract = started['training_contract']
    expected = dict(version=6, mode='train', model='qwen4b', finetune_mode='lora', arm=args['arm'],
        batch_schedule='epoch_shuffle_keep_partial', collective_padding='zero_loss_no_generated_samples',
        updates_per_epoch=2, training_questions=100, batch_size=64, epochs=4, world_size=4,
        K=8, seed=42, alpha=1., context_limit=8192, format_coefficient=0., warmup_steps=0)
    need(all(contract.get(k) == v for k, v in expected.items()) and args.get('rollouts') == 8,
         'Training contract does not describe the complete 100-question pilot')
    need(contract.get('data_sha256') == started['data_sha256'] and contract.get('code_sha256')
         and all(hashes.get(k) == v for k, v in contract['code_sha256'].items()), 'Training contract provenance mismatch')
    need(done.get('steps') == 8 and done.get('epochs') == 4 and done.get('start_step') == 0
         and started.get('start_step') == 0 and started.get('resume_from') is None, 'Incomplete or unexpected resumed schedule')
    need({p.name for p in output.glob('update_*.json')} == {f'update_{i:03d}.json' for i in range(1, 9)},
         'Missing or unexpected optimizer update files')
    active, zero = 0, 0
    for epoch in range(4):
        order = list(ids)
        random.Random(42 + epoch).shuffle(order)
        for within, count in enumerate((64, 36)):
            step = epoch * 2 + within + 1
            u = read(output / f'update_{step:03d}.json')
            need(u.get('step') == step and u.get('mode') == 'train' and u.get('arm') == args['arm']
                 and u.get('finetune_mode') == 'lora', 'Optimizer update identity mismatch')
            norm, totals = u.get('gradient_norm'), u['totals']
            need(finite(norm) and norm >= 0, 'Nonfinite or negative gradient norm')
            need(u.get('rollout_count') == totals.get('responses') == count * 8 and totals.get('groups') == count,
                 'Generated responses do not match the partial-batch budget')
            need(u.get('collective_padding_forwards') == 0, 'Unexpected collective padding budget')
            need(finite(totals.get('nonzero_advantage_tokens')) and totals['nonzero_advantage_tokens'] >= 0,
                 'Invalid advantage signal count')
            active += norm > 0 and totals['nonzero_advantage_tokens'] > 0
            zero += norm == 0
            actual = [identity for rank in range(4)
                      for identity in groups(output / f'rank_{rank}/rollouts_{step-1:03d}.json', 8)]
            need(actual == order[within * 64:within * 64 + count], 'Dropped, repeated or wrong-epoch training questions')
    need(active > 0, 'No genuine update with nonzero gradient and advantage signal')
    cp = checkpoint_module()
    checkpoints = [cp.inspect_checkpoint(output / f'state_{step:03d}', contract, 4) for step in (2, 4, 6, 8)]
    final = checkpoints[-1]
    need(Path(done['final_state_checkpoint']).resolve() == output / 'state_008'
         and Path(done['final_adapter']).resolve() == Path(final['adapter_path']) == output / 'adapter_008',
         'Final optimizer state or adapter path mismatch')
    return dict(arm=args['arm'], epochs=4, steps=8, training_questions=100, real_responses=3200,
                nonzero_updates=active, zero_gradient_updates=zero, epoch_steps=[2, 4, 6, 8],
                final_adapter=final['adapter_path'], final_state_checkpoint=final['state_path'],
                optimizer_state_hashes_verified=True, optimizer_tensors_deserialized=False)


def evaluate(output):
    done, spec, summary = (read(output / name) for name in ('done.json', 'eval_spec.json', 'summary.json'))
    need(done.get('complete') is True and done.get('evaluation_only') is True and not (output / 'failed.json').exists(),
         'Evaluation did not finish successfully')
    need(done.get('eval_spec_sha256') == sha(output / 'eval_spec.json')
         and done.get('summary_sha256') == sha(output / 'summary.json'), 'Evaluation evidence hash mismatch')
    sources(spec)
    need(all(spec.get(k) == v for k, v in dict(evaluation_only=True, model='qwen4b', rollouts_per_question=8,
         context_limit=8192, evaluation_seed=42, temperature=1., structural_reward=0).items()), 'Evaluation protocol mismatch')
    need(done.get('questions') == summary.get('questions') == spec['data'].get('evaluated_questions') == 100
         and done.get('responses') == summary.get('responses') == 800, 'Evaluation sample budget mismatch')
    ids = data_ids(spec['data']['path'], spec['data']['sha256'])
    training_ids = data_ids(spec['train_data']['path'], spec['train_data']['sha256'])
    need(not set(ids) & set(training_ids) and spec['overlap'].get('checked') is True
         and spec['overlap'].get('id_overlap') == spec['overlap'].get('question_overlap') == 0,
         'Evaluation overlaps training or lacks leakage checks')
    need(spec['data'].get('prefix_limit') is None and spec.get('finetune_mode') in ('base', 'lora')
         and done.get('checkpoint') == spec.get('checkpoint'), 'Unexpected evaluation model or subset selection')
    checkpoint = spec.get('checkpoint')
    if spec['finetune_mode'] == 'base':
        need(checkpoint is None, 'Base evaluation must use the unchanged initial model')
    else:
        need(isinstance(checkpoint, dict) and checkpoint.get('finetune_mode') == 'lora'
             and Path(checkpoint.get('path', '')).name == 'adapter_008'
             and spec.get('epoch') == 4 and spec.get('training_step') == 8,
             'Pilot final evaluation must use the completed epoch-four adapter_008')
    actual = [identity for path in sorted(output.glob('groups_*.json')) for identity in groups(path, 8)]
    need(actual == ids, 'Evaluation groups do not match the held-out questions exactly')
    for kind in ('strict', 'relaxed'):
        need(all(finite(summary[kind].get(k)) and 0 <= summary[kind][k] <= 1 for k in ('pass1', 'pass8')),
             'Invalid or missing evaluation accuracy')
    return dict(questions=100, responses=800, finetune_mode=spec['finetune_mode'], epoch=spec.get('epoch'),
                training_step=spec.get('training_step'), strict=summary['strict'], relaxed=summary['relaxed'],
                limitations='Recorded grader results; this gate does not establish improvement or reasoning validity.')


def upload(output):
    raise ValueError('Hugging Face publication is disabled for the 100-question pilot')


def verify(kind, output):
    output = Path(output)
    need(output.is_dir() and not output.is_symlink(), 'Output must be a regular existing directory')
    actions = dict(smoke=smoke, train=train, eval=evaluate, upload=upload)
    need(kind in actions, 'Unknown verification kind')
    output = output.resolve()
    return dict(passed=True, kind=kind, output=str(output), **actions[kind](output))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=('smoke', 'train', 'eval', 'upload'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--report', type=Path, help='Optional new report file; never overwrites an existing report')
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
