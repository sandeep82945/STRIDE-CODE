"""Aggregate-only evaluation of the pilot's grouped, already-scored records.

Answer correctness comes from the existing frozen graders. Text distinctness
is a lexical diagnostic of final explanations, never proof or strategy diversity.
"""
from __future__ import annotations

from collections import Counter
import math
import re
import statistics


LENGTH_REASONS = {'length', 'max_tokens', 'max_length', 'max_new_tokens'}
FINAL_HEADING = re.compile(r'(?im)^[ \t]*(?:#{1,6}[ \t]*)?(?:\*\*)?Final[ \t]+Answer(?:\*\*)?[ \t]*(?::|$)')


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _correct(record, kind):
    score = record['score']
    field = f'{kind}_answer_correct'
    direct = score.get(field)
    nested = score.get(kind, {}).get('answer_correct')
    if direct is None and nested is None:
        raise ValueError(f'Missing {kind} correctness')
    if direct is not None and nested is not None and bool(direct) != bool(nested):
        raise ValueError(f'Inconsistent {kind} correctness fields')
    return bool(direct if direct is not None else nested)


def normalized_explanation(record):
    """Return target-blind, answer-free final explanation text or None.

    Preserve numbers, mathematical expressions, punctuation and step labels.
    Only collapse whitespace and casefold. Native thinking is outside this
    diagnostic's scope, so reasoning-style differences are not conflated with
    whether a model exposes a thinking channel.
    """
    score = record['score']
    text = score.get('response', record.get('text'))
    region = score.get('region', {})
    if not isinstance(text, str) or not region.get('valid') or not region.get('final'):
        return None
    left, right = region['final']
    if not (isinstance(left, int) and isinstance(right, int) and 0 <= left <= right <= len(text)):
        raise ValueError('Invalid final-region offsets')
    body = text[left:right]
    candidate = score.get('relaxed', {}).get('final_answer_start')
    if not isinstance(candidate, int) or not 0 <= candidate <= len(body):
        candidate = None
    headings = [m for m in FINAL_HEADING.finditer(body) if candidate is None or m.start() <= candidate]
    if headings:
        boundary = headings[-1].start()
    elif candidate is not None:
        boundary = candidate
    else:
        # A strict-correct result can lack relaxed extraction offsets. Its
        # terminal boxing command is a target-blind fallback boundary.
        boxes = list(re.finditer(r'(?<!\\)\\boxed[ \t]*\{', body))
        if not boxes:
            return None
        boundary = boxes[-1].start()
    normalized = ' '.join(body[:boundary].casefold().split())
    return normalized or None


def _quantile(values, probability):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * probability
    low = math.floor(position)
    high = math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def _pass_at_k(n, c, k):
    if n < k:
        return None
    return 1.0 if n - c < k else 1.0 - math.comb(n - c, k) / math.comb(n, k)


def _grader_summary(groups, kind):
    counts, eligible, lexical_available, lexical_distinct = [], [], [], []
    group_sizes = []
    reasons = Counter()
    for group in groups:
        records = group['records']
        correct = [record for record in records if _correct(record, kind)]
        counts.append(len(correct))
        eligible.append(sum(bool(r['score']['step_eligible']) for r in correct))
        group_sizes.append(len(records))
        texts = [text for r in correct if (text := normalized_explanation(r)) is not None]
        lexical_available.append(len(texts))
        lexical_distinct.append(len(set(texts)))
        reasons.update(str(r['score'].get(kind, {}).get('reason', 'unavailable')) for r in records)
    questions, responses = len(groups), sum(group_sizes)
    pass8 = [_pass_at_k(n, c, 8) for n, c in zip(group_sizes, counts)]
    lexical_fractions = [d / n for d, n in zip(lexical_distinct, lexical_available) if n]
    return dict(
        correct_answers=sum(counts),
        pass1=statistics.fmean(c / n for c, n in zip(counts, group_sizes)) if questions else None,
        pooled_correct_fraction=_ratio(sum(counts), responses),
        pass8=statistics.fmean(pass8) if questions and all(p is not None for p in pass8) else None,
        observed_any_correct_fraction=_ratio(sum(c > 0 for c in counts), questions),
        questions_with_correct=sum(c > 0 for c in counts),
        at_least_two_correct=sum(c >= 2 for c in counts),
        correct_with_usable_steps=sum(eligible),
        at_least_two_with_declared_steps=sum(c >= 2 for c in eligible),
        eligible_pair_fraction=_ratio(sum(c >= 2 for c in eligible), questions),
        correct_count_histogram=dict(sorted(Counter(map(str, counts)).items(), key=lambda x: int(x[0]))),
        eligible_count_histogram=dict(sorted(Counter(map(str, eligible)).items(), key=lambda x: int(x[0]))),
        answer_reason_counts=dict(sorted(reasons.items())),
        lexical_final_explanation=dict(
            correct_responses_with_nonempty_explanation=sum(lexical_available),
            correct_responses_without_extractable_explanation=sum(counts) - sum(lexical_available),
            distinct_normalized_texts_sum_over_questions=sum(lexical_distinct),
            distinct_fraction_pooled=_ratio(sum(lexical_distinct), sum(lexical_available)),
            mean_distinct_fraction_among_nonempty_groups=statistics.fmean(lexical_fractions) if lexical_fractions else None,
            mean_distinct_texts_per_question=_ratio(sum(lexical_distinct), questions),
            questions_with_two_distinct_correct_texts=sum(c >= 2 for c in lexical_distinct),
        ),
    )


def summarize(groups):
    """Accept trainer.annotate_groups output; return only aggregate statistics."""
    groups = list(groups)
    identities = [(type(g['id']).__name__, g['id']) for g in groups]
    if len(set(identities)) != len(identities):
        raise ValueError('Duplicate evaluation question id')
    for group in groups:
        if not group.get('records'):
            raise ValueError('Evaluation group has no responses')
        indices = [r['sample_index'] for r in group['records']]
        if sorted(indices) != list(range(len(indices))):
            raise ValueError('Missing or duplicate sample index')
    records = [r for g in groups for r in g['records']]
    lengths = []
    for record in records:
        length = len(record['token_ids'])
        for declared in [record.get('generated_tokens'), record['score'].get('generated_tokens')]:
            if declared is not None and declared != length:
                raise ValueError('Token length metadata disagrees with original token ids')
        if 'text' in record and 'response' in record['score'] and record['text'] != record['score']['response']:
            raise ValueError('Sampler and scorer decoded text differ')
        lengths.append(length)
    scores = [r['score'] for r in records]
    truncated = sum(bool(s.get('truncated', r.get('finish_reason') in LENGTH_REASONS)) for r, s in zip(records, scores))
    categories = Counter(code for s in scores for code in set(s.get('diagnostic_categories', [])))
    issues = Counter(code for s in scores for code in set(s.get('parse_issues', [])))
    mapping_issues = Counter(code for s in scores for code in set(s.get('mapping_issues', [])))
    size_counts = Counter(len(g['records']) for g in groups)
    total_context = [len(g['prompt_token_ids']) + len(r['token_ids']) for g in groups for r in g['records']]
    return dict(
        questions=len(groups), responses=len(records),
        rollouts_per_question=next(iter(size_counts)) if len(size_counts) == 1 else None,
        rollout_count_histogram={str(k): v for k, v in sorted(size_counts.items())},
        strict=_grader_summary(groups, 'strict'), relaxed=_grader_summary(groups, 'relaxed'),
        truncations=truncated, truncation_fraction=_ratio(truncated, len(records)),
        generated_tokens=sum(lengths), mean_tokens=statistics.fmean(lengths) if lengths else None,
        median_tokens=_quantile(lengths, .5), p90_tokens=_quantile(lengths, .9),
        p99_tokens=_quantile(lengths, .99), max_tokens=max(lengths, default=None),
        max_prompt_plus_response_tokens=max(total_context, default=None),
        responses_over_8192_total=sum(n > 8192 for n in total_context),
        finish_reason_counts=dict(sorted(Counter(str(r.get('finish_reason')) for r in records).items())),
        stop_reason_counts=dict(sorted(Counter(str(r.get('stop_reason')) for r in records).items())),
        syntax=dict(
            parse_status_counts=dict(sorted(Counter(str(s.get('parse_status', 'unavailable')) for s in scores).items())),
            parse_valid_responses=sum(bool(s.get('parse_valid')) for s in scores),
            step_eligible_responses=sum(bool(s['step_eligible']) for s in scores),
            step_count_histogram=dict(sorted(Counter(str(s.get('step_count', 0)) for s in scores).items(), key=lambda x: int(x[0]))),
            mapping_verified_responses=sum(bool(s.get('mapping_verified')) for s in scores),
            mapping_training_safe_responses=sum(bool(s.get('mapping_training_safe')) for s in scores),
            closed_thinking_responses=sum(bool(s.get('thinking_closed')) for s in scores),
            final_region_reason_counts=dict(sorted(Counter(str(s.get('region', {}).get('reason', 'unavailable')) for s in scores).items())),
            diagnostic_category_counts=dict(sorted(categories.items())),
            parse_issue_response_counts=dict(sorted(issues.items())),
            mapping_issue_response_counts=dict(sorted(mapping_issues.items())),
        ),
        definitions=dict(
            pass1='Mean fraction correct per question; equal to pooled response accuracy when K is fixed.',
            pass8='Mean unbiased 1-C(n-c,8)/C(n,8); with exactly8 draws this is the observed fraction of questions with any correct response. Null if any group has fewer than8 draws.',
            eligible_pair='At least two answer-correct responses with the existing target-independent step_eligible gate; syntax eligibility does not verify the reasoning.',
            lexical_text='Completed final explanation before the terminal answer, casefolded with whitespace collapsed. Numbers, mathematical expressions, punctuation and step labels remain. Native thinking is excluded.',
            lexical_distinct_fraction='Distinct normalized final-explanation strings divided by nonempty extracted strings among answer-correct responses, with uniqueness computed within each question.',
        ),
        limitations=[
            'Lexical text distinctness is not verified reasoning diversity, URT, strategy diversity, or proof validity.',
            'This small sampled comparison does not establish algorithm superiority.',
            'Truncated and malformed responses retain the existing graders and eligibility gates; evaluation does not repair or regrade them.',
        ],
    )
