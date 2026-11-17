#!/usr/bin/env python3
"""Compare the fixed 100-question validation pilot; standard library only.

Manifest v1: {"version":1, "baseline":"/evaluation/base", "arms":{
"local_positive":{"4":"/evaluation/local-final"},
"single_step":{...}, "token_uniform":{...}, "grpo":{...}}}.
Paths may be absolute or relative to the manifest. No checkpoint is selected.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import statistics
import sys

HERE = Path(__file__).resolve().parent
ARMS = ('local_positive', 'single_step', 'token_uniform', 'grpo')
EPOCHS = ('4',)
METRICS = ('strict/pass1', 'strict/pass8', 'relaxed/pass1', 'relaxed/pass8')
QUESTIONS, ROLLOUTS, BOOTSTRAPS, SEED = 100, 8, 2000, 42
PROTOCOL_FIELDS = ('model', 'evaluation_seed', 'sampler_engine_seed', 'python_hash_seed',
                   'rollouts_per_question', 'prompt_batch', 'context_limit', 'temperature',
                   'sampler_max_num_seqs', 'sampler_eager', 'generation_timeout_seconds',
                   'grader', 'structural_reward', 'diversity_scope')
LIMITATIONS = [
    'This is the fixed validation pilot, not a final held-out test or a benchmark-superiority claim.',
    'Intervals resample questions as paired clusters; they do not estimate variability across training seeds.',
    'The 95% percentile intervals are exploratory and are not adjusted for multiple comparisons.',
    'Pass@1 averages correctness over eight responses per question; pass@8 is any correct response among those eight.',
    'Syntax eligibility and lexical diversity do not establish valid reasoning. No checkpoint is selected here.',
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'Missing or unsafe input: ' + str(path))

    def invalid(value):
        raise ValueError('Nonfinite JSON value: ' + value)

    return json.loads(path.read_text(), parse_constant=invalid)


def module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dependencies():
    gate = module_from_path('camd_comparison_gate', HERE / 'verify_outputs.py')
    aggregation = module_from_path('camd_comparison_aggregate', HERE.parent / 'training/evaluate.py')
    return gate, aggregation


def question_id(value):
    require(type(value) in (str, int) and str(value).strip(), 'Invalid question identity')
    return type(value).__name__, str(value)


def finite_equal(left, right):
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(finite_equal(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(finite_equal(a, b) for a, b in zip(left, right))
    if type(left) in (int, float) and type(right) in (int, float):
        return math.isfinite(left) and math.isfinite(right) and math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)
    return type(left) is type(right) and left == right


def protocol(spec):
    require(all(name in spec for name in PROTOCOL_FIELDS), 'Incomplete evaluation protocol')
    identity = spec.get('model_identity')
    require(isinstance(identity, dict) and identity.get('content_sha256') and identity.get('files'),
            'Missing base model content identity')
    require(spec.get('code_sha256'), 'Missing evaluator source identity')
    result = {name: spec[name] for name in PROTOCOL_FIELDS}
    result.update(model_content_sha256=identity['content_sha256'], model_files=identity['files'],
                  data_sha256=spec['data']['sha256'], training_data_sha256=spec['train_data']['sha256'],
                  source_sha256=spec['code_sha256'])
    return result


def checkpoint_provenance(spec, arm, epoch, gate):
    checkpoint = spec.get('checkpoint')
    if arm == 'base':
        require(checkpoint is None and spec.get('finetune_mode') == 'base'
                and spec.get('epoch') is None and spec.get('training_step') is None,
                'Baseline must be the unchanged base, without a training checkpoint')
        return None
    require(isinstance(checkpoint, dict) and spec.get('finetune_mode') == 'lora'
            and checkpoint.get('finetune_mode') == 'lora', 'Expected a LoRA checkpoint evaluation')
    require(spec.get('epoch') == epoch and spec.get('training_step') == epoch * 2,
            'Evaluation epoch/update does not match the manifest slot')
    folder = Path(checkpoint['path'])
    require(folder.name == f'adapter_{epoch*2:03d}', 'Checkpoint name does not match its epoch')
    source_path = folder.parent / 'started.json'
    require(checkpoint.get('source_metadata_sha256') == sha(source_path), 'Changed checkpoint training provenance')
    source = read(source_path)
    contract = source.get('training_contract', {})
    expected = dict(version=6, mode='train', model='qwen4b', finetune_mode='lora', arm=arm,
                    seed=42, epochs=4, K=8, batch_size=64, training_questions=100, world_size=4,
                    updates_per_epoch=2, batch_schedule='epoch_shuffle_keep_partial',
                    collective_padding='zero_loss_no_generated_samples')
    require(all(contract.get(key) == value for key, value in expected.items()),
            'Checkpoint method or training configuration differs from the manifest')
    require(source.get('code_sha256') == spec['code_sha256'], 'Training/evaluation source identities differ')
    require(contract.get('code_sha256') and all(spec['code_sha256'].get(key) == value
            for key, value in contract['code_sha256'].items()), 'Training contract code differs from evaluation')
    model_hash = spec['model_identity']['content_sha256']
    require(checkpoint.get('source_model_identity_verified') is True
            and checkpoint.get('source_model_content_sha256') == model_hash
            and source.get('model_identity', {}).get('content_sha256') == model_hash
            and contract.get('model_identity', {}).get('content_sha256') == model_hash,
            'Checkpoint and evaluation base models differ')
    train_hash = spec['train_data']['sha256']
    require(checkpoint.get('source_training_data_sha256') == source.get('data_sha256')
            == contract.get('data_sha256') == train_hash, 'Checkpoint training datasets differ')
    verified = gate.checkpoint_module().verify_policy(folder, 'lora')
    require(verified['files'] == checkpoint.get('files')
            and verified['manifest_sha256'] == checkpoint.get('manifest_sha256'),
            'Evaluation checkpoint weight identity changed')
    comparable = dict(contract)
    comparable.pop('arm')
    return comparable


def load_evaluation(path, arm, epoch, gate, aggregation):
    path = Path(path)
    verified = gate.verify('eval', path)
    require(verified.get('passed') is True, 'Evaluation output gate did not pass')
    path = path.resolve()
    spec, summary = read(path/'eval_spec.json'), read(path/'summary.json')
    contract = checkpoint_provenance(spec, arm, epoch, gate)
    batch_size = spec.get('prompt_batch')
    require(type(batch_size) is int and batch_size > 0, 'Invalid evaluation prompt batch')
    expected_files = [path/f'groups_{start:06d}.json' for start in range(0, QUESTIONS, batch_size)]
    require(set(path.glob('groups_*.json')) == set(expected_files), 'Unexpected or missing grouped result files')
    groups, group_hashes = [], {}
    for start, file in zip(range(0, QUESTIONS, batch_size), expected_files):
        batch = read(file)
        require(isinstance(batch, list) and len(batch) == min(batch_size, QUESTIONS-start),
                'Grouped result batch size changed')
        groups.extend(batch)
        group_hashes[file.name] = sha(file)
    ids = [question_id(group['id']) for group in groups]
    data_ids = [question_id(row['id']) for row in read(spec['data']['path'])]
    require(ids == data_ids and len(set(ids)) == QUESTIONS, 'Question identities/types/order differ from data')
    recomputed = aggregation.summarize(groups)
    require(finite_equal(recomputed, summary), 'Recorded summary differs from raw grouped responses')
    rows, prompts = [], []
    for group in groups:
        records = group['records']
        require(len(records) == ROLLOUTS, 'Unexpected response count')
        prompts.append(group['prompt_token_ids'])
        row = {'id': group['id']}
        for kind in ('strict', 'relaxed'):
            flags = [aggregation._correct(record, kind) for record in records]
            row[kind+'/pass1'] = sum(flags) / ROLLOUTS
            row[kind+'/pass8'] = float(any(flags))
        rows.append(row)
    return dict(arm=arm, epoch=epoch, output=str(path), protocol=protocol(spec), contract=contract,
                ids=ids, prompts=prompts, per_question=rows, summary=summary,
                evidence=dict(eval_spec_sha256=sha(path/'eval_spec.json'), summary_sha256=sha(path/'summary.json'),
                              done_sha256=sha(path/'done.json'), groups_sha256=group_hashes),
                diagnostics=dict(mean_tokens=summary['mean_tokens'], median_tokens=summary['median_tokens'],
                    p90_tokens=summary['p90_tokens'], truncation_fraction=summary['truncation_fraction'],
                    parse_valid_fraction=summary['syntax']['parse_valid_responses']/summary['responses'],
                    step_eligible_fraction=summary['syntax']['step_eligible_responses']/summary['responses'],
                    mapping_verified_fraction=summary['syntax']['mapping_verified_responses']/summary['responses']))


def bootstrap_weights(n=QUESTIONS, replicates=BOOTSTRAPS, seed=SEED):
    rng = random.Random(seed)
    return [tuple(Counter(rng.randrange(n) for _ in range(n)).items()) for _ in range(replicates)]


def quantile(ordered, probability):
    position = (len(ordered)-1) * probability
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high]-ordered[low]) * (position-low)


def paired_contrast(treatment, reference, weights):
    left, right = treatment['per_question'], reference['per_question']
    require([question_id(row['id']) for row in left] == [question_id(row['id']) for row in right],
            'Paired comparison question identities differ')
    n = len(left)
    results = {}
    for metric in METRICS:
        deltas = [a[metric]-b[metric] for a, b in zip(left, right)]
        samples = sorted(sum(deltas[index]*count for index, count in sample)/n for sample in weights)
        lower, upper = quantile(samples, .025), quantile(samples, .975)
        results[metric] = dict(treatment=statistics.fmean(row[metric] for row in left),
            reference=statistics.fmean(row[metric] for row in right), delta=statistics.fmean(deltas),
            ci95=[lower, upper], ci_excludes_zero=lower > 0 or upper < 0,
            direction='positive_validation_delta' if lower > 0 else 'negative_validation_delta' if upper < 0
                      else 'uncertain_validation_delta')
    return dict(arm=treatment['arm'], epoch=treatment['epoch'], reference=reference['arm'],
                reference_epoch=reference['epoch'], metrics=results)


def manifest_entries(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    manifest = read(manifest_path)
    require(set(manifest) == {'version', 'baseline', 'arms'} and manifest['version'] == 1,
            'Expected comparison manifest version 1 with baseline and arms')
    require(isinstance(manifest['arms'], dict) and set(manifest['arms']) == set(ARMS),
            'Manifest must include exactly the four comparison arms')

    def resolve(value):
        require(isinstance(value, str) and value.strip(), 'Evaluation output path is missing')
        path = Path(value)
        if not path.is_absolute():
            path = manifest_path.parent / path
        require(not path.is_symlink(), 'Evaluation output cannot be a symlink')
        return path.resolve(strict=True)

    entries = [('base', None, resolve(manifest['baseline']))]
    for arm in ARMS:
        epochs = manifest['arms'][arm]
        require(isinstance(epochs, dict) and set(epochs) == set(EPOCHS), 'Each arm must supply only its final epoch 4 evaluation')
        entries.extend((arm, int(epoch), resolve(epochs[epoch])) for epoch in EPOCHS)
    require(len({entry[2] for entry in entries}) == 5, 'An evaluation output was reused in different manifest slots')
    return entries


def public_evaluation(evaluation):
    return {key: evaluation[key] for key in ('arm', 'epoch', 'output', 'per_question', 'summary',
                                              'diagnostics', 'evidence')}


def compare(manifest_path):
    gate, aggregation = dependencies()
    entries = manifest_entries(manifest_path)
    evaluations = [load_evaluation(path, arm, epoch, gate, aggregation) for arm, epoch, path in entries]
    baseline = evaluations[0]
    training_contract = evaluations[1]['contract']
    for evaluation in evaluations[1:]:
        require(evaluation['protocol'] == baseline['protocol'], 'Mixed data, model, source or evaluation configuration')
        require(evaluation['ids'] == baseline['ids'] and evaluation['prompts'] == baseline['prompts'],
                'Paired questions or rendered prompt token IDs differ')
        require(evaluation['contract'] == training_contract, 'Training settings differ across comparison arms/epochs')
    weights = bootstrap_weights()
    grpo = {evaluation['epoch']: evaluation for evaluation in evaluations if evaluation['arm'] == 'grpo'}
    contrasts = [paired_contrast(evaluation, reference, weights) for evaluation in evaluations[1:]
                 for reference in (baseline, grpo[evaluation['epoch']])]
    return dict(version=1, complete=True, scope='validation_pilot', checkpoint_selected=False,
                manifest_sha256=sha(manifest_path), questions=QUESTIONS, rollouts_per_question=ROLLOUTS,
                protocol=baseline['protocol'], training_contract=training_contract,
                bootstrap=dict(unit='paired_question', replicates=BOOTSTRAPS, seed=SEED,
                    confidence=.95, interval='percentile_linear_quantile', multiplicity_adjusted=False,
                    shared_resamples_across_contrasts=True),
                evaluations=[public_evaluation(value) for value in evaluations], contrasts=contrasts,
                limitations=LIMITATIONS)


def percentage(value):
    return f'{100*value:.1f}%'


def delta_text(result):
    low, high = result['ci95']
    return f"{100*result['delta']:+.1f} [{100*low:+.1f}, {100*high:+.1f}]"


def markdown(result):
    baseline = result['evaluations'][0]
    rows = ['# STRIDE 100-question validation pilot', '',
            'Four LoRA methods trained for four epochs, evaluated once at the final checkpoint on the same 100 validation questions with eight responses per question. '
            'No checkpoint is selected and the final test split is not used.', '',
            f"Base: strict pass@1 {percentage(baseline['summary']['strict']['pass1'])}; "
            f"relaxed pass@1 {percentage(baseline['summary']['relaxed']['pass1'])}; "
            f"relaxed pass@8 {percentage(baseline['summary']['relaxed']['pass8'])}.", '',
            'Deltas below are relaxed pass@1 percentage points with paired question bootstrap 95% intervals. '
            'All four strict/relaxed pass@1/pass@8 contrasts and original summaries are in `comparison.json`.', '',
            '| Method | Epoch | Strict P@1 | Relaxed P@1 | Relaxed P@8 | Δ base [95% CI] | Δ GRPO [95% CI] | Mean tokens | Step eligible | Truncated |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    lookup = {(item['arm'], item['epoch'], item['reference']): item for item in result['contrasts']}
    for item in result['evaluations'][1:]:
        summary, diagnostics = item['summary'], item['diagnostics']
        base = lookup[item['arm'], item['epoch'], 'base']['metrics']['relaxed/pass1']
        grpo = lookup[item['arm'], item['epoch'], 'grpo']['metrics']['relaxed/pass1']
        rows.append(f"| {item['arm']} | {item['epoch']} | {percentage(summary['strict']['pass1'])} | "
                    f"{percentage(summary['relaxed']['pass1'])} | {percentage(summary['relaxed']['pass8'])} | "
                    f"{delta_text(base)} | {delta_text(grpo)} | {diagnostics['mean_tokens']:.0f} | "
                    f"{percentage(diagnostics['step_eligible_fraction'])} | {percentage(diagnostics['truncation_fraction'])} |")
    rows += ['', 'Intervals use 2,000 shared paired question resamples, seed 42. They are exploratory, '
             'are not adjusted for multiple comparisons, and do not measure variability across training seeds. '
             'Step eligibility describes syntax and token mapping, not verified reasoning.', '']
    return '\n'.join(rows)


def write_results(manifest_path, output):
    output = Path(output)
    require(not output.exists() and not output.is_symlink(), 'Use a fresh comparison output directory')
    manifest_before = sha(manifest_path)
    result = compare(manifest_path)
    require(sha(manifest_path) == manifest_before, 'Comparison manifest changed while reading results')
    # Output creation follows all verification, so invalid inputs cannot produce a success marker.
    output.mkdir(parents=True, exist_ok=False)
    with (output/'comparison.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    with (output/'comparison.md').open('x') as stream:
        stream.write(markdown(result))
    with (output/'done.json').open('x') as stream:
        json.dump(dict(complete=True, scope='validation_pilot', manifest_sha256=manifest_before,
                       comparison_sha256=sha(output/'comparison.json'), markdown_sha256=sha(output/'comparison.md')),
                  stream, indent=2, allow_nan=False)
        stream.write('\n')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = write_results(args.manifest, args.output)
    except Exception as exc:
        print(json.dumps(dict(complete=False, error_type=type(exc).__name__, error=str(exc))), file=sys.stderr)
        return 1
    print(json.dumps(dict(complete=True, output=str(args.output.resolve()), evaluations=len(result['evaluations']),
                          contrasts=len(result['contrasts']), scope=result['scope'])))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
