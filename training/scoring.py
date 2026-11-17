"""Native full-response syntax validation, independent accuracy, original IDs.

Only the completed final region is parsed. The strict full parser remains a
separate diagnostic; a validated native presentation adapter supports harmless
Markdown and uncredited preambles. Thinking controls never receive diversity
credit. No structural reward is created. Syntax is not a mathematical proof.
"""
from dataclasses import replace
import re

from native_step_parser import ParseResult, parse_response, map_original_token_spans
from pilot_grading.relaxed_grader import verify_relaxed_answer, verify_strict_answer
from pilot_grading.integer_reference import _LENGTH_FINISHES
from native_contract import parse_native_final


def native_regions(text, prompt, eos_text=None):
    end = len(text.rstrip())
    if eos_text and text[:end].endswith(eos_text):
        end = len(text[:end-len(eos_text)].rstrip())
    body = text[:end]
    opened = prompt.rstrip().endswith('<think>')
    start = 0
    first = re.match(r'^\s*<think>\s*', body)
    if first:
        if opened:
            return {'valid': False, 'reason': 'duplicate_thinking_open'}
        opened, start = True, first.end()
    if opened:
        if body.count('</think>') != 1 or '<think>' in body[start:]:
            return {'valid': False, 'reason': 'unfinished_or_malformed_thinking'}
        close = body.index('</think>')
        if close < start:
            return {'valid': False, 'reason': 'malformed_thinking_order'}
        return {'valid': True, 'thinking': (start, close),
                'final': (close + len('</think>'), end), 'reason': 'closed_thinking'}
    if '<think>' in body or '</think>' in body:
        return {'valid': False, 'reason': 'unexpected_thinking_tag'}
    return {'valid': True, 'thinking': None, 'final': (0, end), 'reason': 'plain_response'}


def _shift_parse(parsed, original, offset):
    steps = [replace(s, start=s.start+offset, end=s.end+offset,
                     marker_start=s.marker_start+offset, marker_end=s.marker_end+offset)
             for s in parsed.steps]
    answer = parsed.final_answer
    if answer is not None:
        answer = replace(answer, **{name: getattr(answer, name)+offset for name in
                                    ('start', 'end', 'box_start', 'box_end', 'content_start', 'content_end')})
    issues = [replace(i, start=None if i.start is None else i.start+offset,
                      end=None if i.end is None else i.end+offset) for i in parsed.issues]
    # Keep the actual result status and every parser issue; do not manufacture
    # a canonical ParseResult just to satisfy the token mapper.
    result = ParseResult(original, parsed.status, steps, answer, issues)
    if any(s.text != original[s.start:s.end] for s in steps):
        raise AssertionError('Shifted step differs from original response text')
    return result


def _mapping_safety(mapping, parsed, ids, tokenizer, region):
    errors = [i.code for i in mapping.issues if i.severity == 'error']
    if mapping.training_safe:
        return True, False
    if not mapping.verified or not region.get('thinking') or set(errors) != {'interior_special_token_id'}:
        return False, False
    # The generic mapper cannot know that validated native thinking controls
    # are outside the final region. Permit only that specific global warning
    # after inspecting every original special ID and original byte offset.
    special = set(tokenizer.all_special_ids)
    last = max((i for i, token in enumerate(ids) if token not in special), default=-1)
    bodies = [(len(parsed.source[:s.start].encode()), len(parsed.source[:s.end].encode())) for s in parsed.steps]
    final_start_byte = len(parsed.source[:region['final'][0]].encode())
    for i, token in enumerate(ids[:last]):
        if token not in special:
            continue
        if tokenizer.convert_ids_to_tokens(token) not in ('<think>', '</think>'):
            return False, False
        a, b = mapping.token_byte_offsets[i]
        if b > final_start_byte or any(a < d and b > c for c, d in bodies):
            return False, False
    return True, True


def annotate(tokenizer, prompt_ids, response_ids, ground_truth, finish_reason):
    """Annotate original token IDs. ``step_eligible`` is target-independent;
    ``stride_eligible`` additionally requires relaxed answer correctness.
    Returned ``spans`` remain available for incorrect but structurally eligible
    responses, allowing the driver to extract features without target gating.
    """
    ids = [int(x) for x in response_ids]
    prompt = tokenizer.decode(prompt_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    text = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    region = native_regions(text, prompt, tokenizer.eos_token)
    truncated = finish_reason in _LENGTH_FINISHES
    result = dict(response=text, generated_tokens=len(ids), finish_reason=finish_reason,
                  answer_correct=False, strict_answer_correct=False, relaxed_answer_correct=False,
                  step_eligible=False, stride_eligible=False, spans=[], steps=[], step_char_spans=[],
                  parse_valid=False, parse_status='invalid', parse_issues=[], mapping_issues=[],
                  full_parse_valid=False, full_parse_status='invalid', full_parse_issues=[],
                  native_parse_valid=False, native_parse_issues=[], parse_path='none',
                  mapping_verified=False, mapping_training_safe=False, raw_mapping_training_safe=False,
                  validated_native_controls=False, thinking_closed=bool(region.get('thinking')),
                  truncated=truncated, region=region, step_count=0,
                  diagnostic_categories=[])
    if not region['valid']:
        failure = {'answer_correct': False, 'reason': region['reason']}
        return {**result, 'strict': failure, 'relaxed': failure,
                'diagnostic_categories': [region['reason']] + (['truncated'] if truncated else [])}
    left, right = region['final']
    final = text[left:right]
    strict = verify_strict_answer(final, ground_truth, finish_reason, 'integer')
    relaxed = verify_relaxed_answer(final, ground_truth, finish_reason, 'integer')
    full_parsed = _shift_parse(parse_response(final, finish_reason=finish_reason), text, left)
    native_parsed = _shift_parse(parse_native_final(final), text, left)
    # Even canonical-looking text must satisfy the hardened native gates
    # (nonempty content, no decimal step labels, no fenced fake boundaries).
    parsed = full_parsed if native_parsed.valid and full_parsed.valid else native_parsed
    parse_path = 'strict_full' if parsed is full_parsed else 'native_presentation' if parsed.valid else 'none'
    mapping = map_original_token_spans(tokenizer, ids, parsed, skip_special_tokens=False,
                                      allow_recoverable=True)
    safe, control_allowance = _mapping_safety(mapping, parsed, ids, tokenizer, region)
    step_eligible = bool(not truncated and parsed.valid and len(parsed.steps) >= 2 and mapping.verified and safe)
    categories = []
    if truncated: categories.append('truncated')
    if not parsed.valid: categories.append('invalid_full_response_format')
    elif not parsed.canonical: categories.append('recoverable_full_response_format')
    if len(parsed.steps) < 2: categories.append('fewer_than_two_steps')
    if not mapping.verified: categories.append('unverified_token_mapping')
    elif not safe: categories.append('unsafe_token_mapping')
    if relaxed['answer_correct'] and not step_eligible: categories.append('correct_but_step_ineligible')
    if control_allowance: categories.append('validated_native_thinking_controls')
    result.update(strict=strict, relaxed=relaxed, answer_correct=bool(relaxed['answer_correct']),
                  strict_answer_correct=bool(strict['answer_correct']),
                  relaxed_answer_correct=bool(relaxed['answer_correct']), step_eligible=step_eligible,
                  stride_eligible=bool(relaxed['answer_correct'] and step_eligible),
                  parse_valid=parsed.valid, parse_status=parsed.status,
                  parse_issues=[i.code for i in parsed.issues],
                  full_parse_valid=full_parsed.valid, full_parse_status=full_parsed.status,
                  full_parse_issues=[i.code for i in full_parsed.issues],
                  native_parse_valid=native_parsed.valid,
                  native_parse_issues=[i.code for i in native_parsed.issues],parse_path=parse_path,
                  mapping_issues=[i.code for i in mapping.issues],
                  mapping_verified=mapping.verified, mapping_training_safe=safe,
                  raw_mapping_training_safe=mapping.training_safe,
                  validated_native_controls=control_allowance, step_count=len(parsed.steps),
                  diagnostic_categories=categories)
    if step_eligible:
        result.update(spans=[[s.token_start, s.token_end] for s in mapping.spans],
                      steps=[s.text for s in parsed.steps],
                      step_char_spans=[[s.start, s.end] for s in parsed.steps])
    return result


def score_record(record, row, tokenizer):
    return annotate(tokenizer, record['prompt_ids'], record['response_ids'],
                    row['ground_truth'], record['finish_reason'])
