"""Adversarial syntax/grading/aggregation tests; no semantic or GPU claims.

The ByteLevel fixture controls original token boundaries, including split UTF-8
characters, without relying on retokenization. Set STRIDE_TEST_TOKENIZER_PATH to
an existing pinned Qwen tokenizer snapshot to exercise the actual tokenizer too.
"""
from __future__ import annotations

from fractions import Fraction
import itertools
import os
import random
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training'))
import evaluate
import native_contract as native
import native_declared_steps as declared
import native_step_parser as parser
from pilot_grading import integer_reference as integer
from pilot_grading import math_reference as legacy
from pilot_grading import relaxed_grader as relaxed
import scoring

CANONICAL = 'Step 1: Compute 2 + 2 = 4.\nStep 2: The result is 4.\nFinal Answer: \\boxed{4}'


class ByteLevelFixture:
    """An exact byte tokenizer with configurable merged tokens and controls."""
    eos_token = '<|im_end|>'

    def __init__(self):
        self.decoder = parser._byte_decoder()
        self.encoder = {v: k for k, v in self.decoder.items()}
        self.vocab = {i: self.encoder[i] for i in range(256)}
        self.payload = {i: bytes([i]) for i in range(256)}
        self.all_special_ids = [256, 257, 258, 259]
        self.vocab.update({256: self.eos_token, 257: '<think>', 258: '</think>', 259: '<|bad|>'})
        self.payload.update({k: self.vocab[k].encode() for k in self.all_special_ids})
        self.added = {v: k for k, v in self.vocab.items() if k in self.all_special_ids}

    def encode(self, text, add_special_tokens=False):
        return list(text.encode())

    def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        return b''.join(self.payload[i] for i in ids if not (skip_special_tokens and i in self.all_special_ids)).decode('utf-8', errors='replace')

    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, int):
            return self.vocab[ids]
        return [self.vocab[i] for i in ids]

    def get_added_vocab(self):
        return self.added

    def merge(self, payload, *, added=False):
        token_id = max(self.vocab) + 1
        self.vocab[token_id] = payload.decode() if added else ''.join(self.encoder[b] for b in payload)
        self.payload[token_id] = payload
        if added:
            self.added[self.vocab[token_id]] = token_id
        return token_id


def record(text=CANONICAL, correct=True, eligible=True, sample_index=0, length=4):
    final = relaxed.extract_final_answer(text)
    score = dict(response=text, generated_tokens=length,
                 strict_answer_correct=correct, relaxed_answer_correct=correct,
                 strict={'answer_correct': correct, 'reason': 'fixture'},
                 relaxed={**final, 'answer_correct': correct, 'reason': 'fixture'},
                 step_eligible=eligible, parse_valid=True, parse_status='canonical',
                 step_count=2, mapping_verified=True, mapping_training_safe=True,
                 region={'valid': True, 'final': (0, len(text)), 'reason': 'plain_response'})
    return dict(sample_index=sample_index, text=text, score=score, token_ids=list(range(length)),
                generated_tokens=length, finish_reason='stop')


def group(identity, flags):
    return dict(id=identity, prompt_token_ids=[1, 2],
                records=[record(correct=c, sample_index=i) for i, c in enumerate(flags)])


class StrictParserTests(unittest.TestCase):
    def test_span_preservation_and_serialization(self):
        response = CANONICAL.replace('2 + 2', 'α + 中文 + 2')
        parsed = parser.parse_response(response)
        self.assertTrue(parsed.canonical)
        self.assertTrue(parsed.valid)
        self.assertEqual(parsed.to_dict()['source'], response)
        self.assertEqual(parsed.to_dict()['steps'][0]['number'], 1)
        for step in parsed.steps:
            self.assertEqual(step.text, response[step.start:step.end])
            self.assertEqual(response[step.marker_start:step.marker_end], f'Step {step.number}:')
        answer = parsed.final_answer
        self.assertEqual(answer.text, response[answer.content_start:answer.content_end])
        self.assertEqual(response[answer.box_start:answer.box_end], r'\boxed{4}')

    def test_recoverable_presentation_is_explicit_opt_in(self):
        for replacement in ('  Step 1:', '### Step 1:', '**Step 1:**', 'step 1.', 'Step 1:\t'):
            with self.subTest(replacement=replacement):
                text = CANONICAL.replace('Step 1:', replacement, 1)
                parsed = parser.parse_response(text)
                self.assertEqual(parsed.status, 'recoverable')
                self.assertEqual(parser.parse_reasoning_steps(text), [])
                self.assertEqual(len(parser.parse_reasoning_steps(text, allow_recoverable=True)), 2)

    def test_newline_forms_preserve_source(self):
        for newline, status in (('\n', 'canonical'), ('\r\n', 'canonical'), ('\r', 'recoverable'), ('\u2028', 'recoverable'), ('\u2029', 'recoverable'), ('\x85', 'recoverable')):
            with self.subTest(newline=repr(newline)):
                source = CANONICAL.replace('\n', newline)
                parsed = parser.parse_response(source)
                self.assertEqual(parsed.status, status)
                self.assertEqual([source[s.start:s.end] for s in parsed.steps], [s.text for s in parsed.steps])

    def test_invalid_markers_and_boundaries(self):
        cases = {
            'preamble': 'Intro\n' + CANONICAL,
            'nonsequential_numbering': CANONICAL.replace('Step 2:', 'Step 3:'),
            'invalid_step_number': CANONICAL.replace('Step 1:', 'Step 01:'),
            'unbalanced_marker_bold': CANONICAL.replace('Step 1:', '**Step 1:'),
            'inline_step_marker': CANONICAL.replace('\nStep 2:', ' Step 2:'),
            'malformed_marker': CANONICAL.replace('Step 2:', 'Step 2)'),
            'control_character': CANONICAL.replace('Compute', '\x00Compute'),
            'interior_special_token': CANONICAL.replace('Compute', '<|im_start|>Compute'),
            'step_after_final_answer': CANONICAL + '\nStep 3: More.',
            'multiple_final_answers': CANONICAL + '\nFinal Answer: \\boxed{4}',
            'empty_step': CANONICAL.replace('Compute 2 + 2 = 4.', '  '),
            'text_after_final_answer': CANONICAL + ' Extra text.',
        }
        for expected, text in cases.items():
            with self.subTest(expected=expected):
                parsed = parser.parse_response(text)
                self.assertFalse(parsed.valid)
                self.assertIn(expected, [i.code for i in parsed.issues])

    def test_huge_number_fails_without_int_conversion_crash(self):
        text = CANONICAL.replace('Step 1:', 'Step ' + '9' * 5000 + ':')
        parsed = parser.parse_response(text)
        self.assertFalse(parsed.valid)
        self.assertIn('step_number_overflow', [i.code for i in parsed.issues])
        self.assertEqual(declared.permissive_steps(text, 0, len(text)), [])

    def test_missing_steps_final_and_limits_keep_diagnostics(self):
        self.assertFalse(parser.parse_response('no markers').valid)
        for text in ('Final Answer: \\boxed{4}', 'Step 1: Body.'):
            self.assertFalse(parser.parse_response(text).valid)
        for kwargs, issue in (({'min_step_chars': 100}, 'configured_min_step_chars'),
                              ({'max_step_chars': 1}, 'configured_max_step_chars'),
                              ({'max_steps': 1}, 'configured_max_steps')):
            parsed = parser.parse_response(CANONICAL, **kwargs)
            self.assertFalse(parsed.valid)
            self.assertEqual(len(parsed.steps), 2)
            self.assertIn(issue, [i.code for i in parsed.issues])
        with self.assertRaises(ValueError):
            parser.parse_reasoning_steps(CANONICAL, require_math_content=True)
        with self.assertRaises(TypeError):
            parser.parse_response(None)

    def test_box_grammar_and_wrappers(self):
        nested_math = CANONICAL.replace(r'\boxed{4}', r'\boxed{\frac{8}{2}}')
        self.assertTrue(parser.parse_response(nested_math).canonical)
        for opener, closer in (('$', '$'), ('$$', '$$'), (r'\[', r'\]'), (r'\(', r'\)')):
            text = CANONICAL.replace(r'\boxed{4}', opener + r'\boxed{4}' + closer)
            self.assertEqual(parser.parse_response(text).status, 'recoverable')
            self.assertFalse(parser.parse_response(text[:-len(closer)]).valid)
        for box in (r'\boxed{}', r'\boxed{4', r'\boxed{\boxed{4}}', '4', r'\boxed{4} trailing'):
            self.assertFalse(parser.parse_response(CANONICAL.replace(r'\boxed{4}', box)).valid)
        self.assertFalse(parser.parse_response(CANONICAL.replace('Compute', r'\boxed{1} Compute')).valid)

    def test_fences_and_step_references(self):
        text = CANONICAL.replace('Compute 2 + 2 = 4.', 'Compute.\n```python\nStep 99: fake\n```')
        self.assertEqual(len(parser.parse_response(text).steps), 2)
        self.assertEqual(parser.parse_response(text).status, 'recoverable')
        self.assertFalse(parser.parse_response(text.replace('\n```\nStep 2', '\nStep 2')).valid)
        for reference, valid in (('Using Step 1: result.', True), ('from Step 9: result.', False), ('Step 1: result.', False)):
            text = CANONICAL.replace('The result is 4.', 'Obtain ' + reference)
            self.assertEqual(parser.parse_response(text).valid, valid)
        match = parser._INLINE_STEP.search('from Step 1.')
        self.assertTrue(parser._is_step_reference('from Step 1.', match, {1}))
        self.assertFalse(parser._is_step_reference('from Step 1.', match, {2}))

    def test_terminal_tokens_and_supported_length_gates(self):
        for token in ('<|im_end|>', '<|endoftext|>', '<|eot_id|>'):
            self.assertTrue(parser.parse_response(CANONICAL + '\n' + token).canonical)
        for reason in integer._LENGTH_FINISHES:
            self.assertFalse(parser.parse_response(CANONICAL, finish_reason=reason).valid)
        self.assertEqual(parser._trim_span(' abc ', 0, 5), (1, 4))
        self.assertTrue(parser._escaped(r'\{', 1))
        self.assertFalse(parser._escaped(r'\\{', 2))


class NativeAndMappingTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = ByteLevelFixture()

    def test_native_allows_uncredited_intro_and_markdown(self):
        source = 'Brief introduction.\n' + CANONICAL.replace('Step 1:', '## Step 1 —').replace('Step 2:', '**Step 2.**')
        parsed = native.parse_native_final(source)
        self.assertTrue(parsed.valid)
        self.assertEqual(parsed.status, 'recoverable')
        self.assertIn('native_uncredited_preamble', [i.code for i in parsed.issues])
        self.assertEqual([source[s.start:s.end] for s in parsed.steps], [s.text for s in parsed.steps])
        self.assertTrue(native._presentation_only(r'\[ ** $ : '))
        self.assertFalse(native._presentation_only('the answer'))

    def test_native_rejects_malformed_declared_steps(self):
        for label in ('Step 02:', 'Step 2.1:', 'Step -2:', 'Step two:', 'Step 3:', 'Step ２:'):
            source = CANONICAL.replace('Step 2:', label)
            with self.subTest(label=label):
                self.assertFalse(native.parse_native_final(source).valid)
                self.assertEqual(declared.permissive_steps(source, 0, len(source)), [])
        for body in ('!!! ...', '\u200b\u2060', '  '):
            self.assertFalse(native.parse_native_final(CANONICAL.replace('The result is 4.', body)).valid)
        self.assertFalse(declared._has_content(' -- ** '))
        self.assertTrue(declared._has_content('∑ x'))
        self.assertEqual(declared.permissive_steps(CANONICAL, -1, len(CANONICAL)), [])
        self.assertEqual(declared.permissive_steps(CANONICAL, 2, 1), [])
        self.assertEqual(declared.permissive_steps(CANONICAL, 0, len(CANONICAL) + 1), [])

    def test_native_answer_separation_and_extra_box_gate(self):
        valid = CANONICAL.replace('Final Answer: \\boxed{4}', 'Answer: 4')
        self.assertTrue(native.parse_native_final(valid).valid)
        for source in (CANONICAL.replace('Final Answer: ', 'Here is '),
                       CANONICAL.replace('Final Answer: ', 'Final Answer: After calculation '),
                       CANONICAL.replace('Compute', r'\boxed{4} Compute'),
                       CANONICAL + '\nFinal Answer: 4',
                       CANONICAL.replace('The result is 4.', 'Done. Step 3: more'),
                       CANONICAL.replace('Compute', '\x00Compute')):
            self.assertFalse(native.parse_native_final(source).valid)

    def test_native_fenced_fake_steps_and_unclosed_fences_rejected(self):
        for source in ('```\n' + CANONICAL + '\n```',
                       CANONICAL.replace('Compute', '\n```\nCompute'),
                       CANONICAL.replace('Compute', '\n```\nStep 8: fake\n```\nCompute')):
            self.assertFalse(native.parse_native_final(source).valid)
        ordinary = CANONICAL.replace('Compute', '\n```\nx = 4\n```\nCompute')
        self.assertTrue(native.parse_native_final(ordinary).valid)

    def test_mapper_never_retokenizes_and_preserves_split_unicode(self):
        text = CANONICAL.replace('Compute', '中文 α 😀 Compute')
        ids = self.tokenizer.encode(text)
        with patch.object(self.tokenizer, 'encode', side_effect=AssertionError('retokenization forbidden')):
            mapping = parser.map_original_token_spans(self.tokenizer, ids)
        self.assertTrue(mapping.verified)
        self.assertTrue(mapping.training_safe)
        self.assertEqual(len(mapping.token_byte_offsets), len(ids))
        self.assertEqual(mapping.token_byte_offsets[-1][1], len(text.encode()))
        self.assertTrue(mapping.to_dict()['verified'])
        for step, span in zip(parser.parse_response(text).steps, mapping.spans):
            self.assertEqual(self.tokenizer.decode(ids[span.token_start:span.token_end + 1]), step.text)
        self.assertEqual(len(set(parser._byte_decoder().values())), 256)

    def test_mapper_accepts_whitespace_boundary_token_but_rejects_marker_merges(self):
        original = self.tokenizer.encode(CANONICAL)
        position = CANONICAL.index('Compute')
        harmless = self.tokenizer.merge(b' Compute')
        ids = original[:position - 1] + [harmless] + original[position + len('Compute'):]
        self.assertTrue(parser.map_original_token_spans(self.tokenizer, ids).training_safe)
        crossing = self.tokenizer.merge(b': Compute')
        ids = original[:position - 2] + [crossing] + original[position + len('Compute'):]
        mapped = parser.map_original_token_spans(self.tokenizer, ids)
        self.assertTrue(mapped.verified)
        self.assertFalse(mapped.training_safe)
        self.assertIn('boundary_content_loss', [i.code for i in mapped.issues])
        self.assertTrue(mapped.spans[0].excluded_boundary_tokens)

    def test_mapper_rejects_decode_mismatch_bad_vocab_and_invalid_parse(self):
        ids = self.tokenizer.encode(CANONICAL)
        wrong = parser.parse_response(CANONICAL.replace('Compute', 'Calculate'))
        self.assertFalse(parser.map_original_token_spans(self.tokenizer, ids, wrong).verified)
        self.assertFalse(parser.map_original_token_spans(self.tokenizer, self.tokenizer.encode('no steps')).verified)
        with patch.object(self.tokenizer, 'convert_ids_to_tokens', return_value=['invalid']):
            mapped = parser.map_original_token_spans(self.tokenizer, ids)
            self.assertIn('unsupported_token_bytes', [i.code for i in mapped.issues])
        with patch.object(self.tokenizer, 'convert_ids_to_tokens', return_value=['x'] * len(ids)):
            mapped = parser.map_original_token_spans(self.tokenizer, ids)
            self.assertIn('token_bytes_decode_mismatch', [i.code for i in mapped.issues])

    def test_mapper_added_vocab_special_ids_and_empty_spans(self):
        ids = self.tokenizer.encode(CANONICAL)
        mapped = parser.map_original_token_spans(self.tokenizer, ids + [256])
        self.assertTrue(mapped.training_safe)
        self.assertEqual(mapped.token_byte_offsets[-1][0], mapped.token_byte_offsets[-1][1])
        mapped = parser.map_original_token_spans(self.tokenizer, [259] + ids)
        self.assertTrue(mapped.verified)
        self.assertFalse(mapped.training_safe)
        added = self.tokenizer.merge('Compute'.encode(), added=True)
        position = CANONICAL.index('Compute')
        changed = ids[:position] + [added] + ids[position + 7:]
        self.assertTrue(parser.map_original_token_spans(self.tokenizer, changed).training_safe)
        tiny = 'Step 1: a\nFinal Answer: \\boxed{4}'
        complete = self.tokenizer.merge(b': a')
        changed = self.tokenizer.encode('Step 1') + [complete] + self.tokenizer.encode('\nFinal Answer: \\boxed{4}')
        mapped = parser.map_original_token_spans(self.tokenizer, changed)
        self.assertFalse(mapped.training_safe)
        self.assertIn('empty_token_span', [i.code for i in mapped.issues])

    def test_interior_controls_and_noncontiguous_mapping_fail_closed(self):
        ids = self.tokenizer.encode(CANONICAL)
        position = CANONICAL.index('Compute') + 3
        internal = ids[:position] + [259] + ids[position:]
        mapped = parser.map_original_token_spans(self.tokenizer, internal)
        self.assertTrue(mapped.verified)
        self.assertFalse(mapped.training_safe)
        self.assertIn('noncontiguous_token_span', [i.code for i in mapped.issues])
        ids = [257, 259] + self.tokenizer.encode('work') + [258] + self.tokenizer.encode(CANONICAL)
        scored = scoring.annotate(self.tokenizer, [], ids, 4, 'stop')
        self.assertTrue(scored['mapping_verified'])
        self.assertFalse(scored['mapping_training_safe'])
        self.assertFalse(scored['stride_eligible'])

    def test_deterministic_mutation_fuzz_preserves_spans_and_fails_truncation(self):
        rng = random.Random(731)
        insertions = ['Step 9:', '\\boxed{5}', '<think>', '</think>', '<|bad|>',
                      '\x00', '中文', '\u2028', '***', '\\', '}', '{', '\n', 'Answer: 5']
        for _ in range(1000):
            source = CANONICAL
            for _ in range(rng.randint(1, 4)):
                at = rng.randrange(len(source) + 1)
                source = source[:at] + rng.choice(insertions) + source[at:]
            parsed = parser.parse_response(source)
            for step in parsed.steps:
                self.assertTrue(0 <= step.start <= step.end <= len(source))
                self.assertEqual(step.text, source[step.start:step.end])
            result = relaxed.extract_final_answer(source)
            if result['extraction_valid']:
                self.assertIsNotNone(relaxed.numeric_value(result['extracted_answer']))
            for target in (4, 5):
                truncated = scoring.annotate(self.tokenizer, [], self.tokenizer.encode(source), target, 'max_new_tokens')
                self.assertFalse(truncated['answer_correct'])
                self.assertFalse(truncated['stride_eligible'])

    def test_native_regions_thinking_variants(self):
        result = scoring.native_regions('<think>work</think>' + CANONICAL, '')
        self.assertTrue(result['valid'])
        self.assertEqual(result['thinking'], (7, 11))
        result = scoring.native_regions('work</think>' + CANONICAL + '<|im_end|>', 'prompt<think>', '<|im_end|>')
        self.assertTrue(result['valid'])
        self.assertEqual(result['final'][1], len('work</think>' + CANONICAL))
        for text, prompt in (('<think>work</think>' + CANONICAL, 'prompt<think>'),
                             ('<think>unfinished', ''),
                             ('<think>x</think></think>' + CANONICAL, ''),
                             ('<think>x<think>y</think>' + CANONICAL, ''),
                             ('x</think>' + CANONICAL, '')):
            self.assertFalse(scoring.native_regions(text, prompt)['valid'])

    def test_annotation_correctness_is_independent_of_step_eligibility(self):
        ids = self.tokenizer.encode(CANONICAL)
        good = scoring.annotate(self.tokenizer, [], ids, 4, 'stop')
        wrong = scoring.annotate(self.tokenizer, [], ids, 5, 'stop')
        self.assertTrue(good['stride_eligible'])
        self.assertTrue(wrong['step_eligible'])
        self.assertFalse(wrong['stride_eligible'])
        self.assertEqual(good['spans'], wrong['spans'])
        self.assertEqual(good['steps'], wrong['steps'])
        no_steps = scoring.annotate(self.tokenizer, [], self.tokenizer.encode('Answer: 4'), 4, 'stop')
        self.assertTrue(no_steps['answer_correct'])
        self.assertFalse(no_steps['step_eligible'])
        self.assertIn('correct_but_step_ineligible', no_steps['diagnostic_categories'])
        self.assertEqual(scoring.score_record({'prompt_ids': [], 'response_ids': ids, 'finish_reason': 'stop'}, {'ground_truth': 4}, self.tokenizer), good)

    def test_annotation_all_truncation_finish_reasons_fail_closed(self):
        for reason in integer._LENGTH_FINISHES:
            scored = scoring.annotate(self.tokenizer, [], self.tokenizer.encode(CANONICAL), 4, reason)
            self.assertTrue(scored['truncated'])
            self.assertFalse(scored['answer_correct'])
            self.assertFalse(scored['step_eligible'])
            self.assertFalse(scored['stride_eligible'])
            self.assertEqual(scored['spans'], [])

    def test_annotation_thinking_special_ids_never_receive_credit(self):
        thinking = self.tokenizer.encode('Private work 中文')
        ids = [257] + thinking + [258] + self.tokenizer.encode('\n' + CANONICAL) + [256]
        scored = scoring.annotate(self.tokenizer, [], ids, 4, 'stop')
        self.assertTrue(scored['stride_eligible'])
        self.assertTrue(scored['validated_native_controls'])
        self.assertFalse(scored['raw_mapping_training_safe'])
        self.assertTrue(scored['mapping_training_safe'])
        for a, b in scored['spans']:
            self.assertGreater(a, len(thinking) + 1)
            self.assertFalse(set(ids[a:b + 1]) & set(self.tokenizer.all_special_ids))
        invalid = scoring.annotate(self.tokenizer, [], [257] + self.tokenizer.encode(CANONICAL), 4, 'stop')
        self.assertFalse(invalid['answer_correct'])
        self.assertEqual(invalid['parse_path'], 'none')

    def test_gibberish_is_syntactically_eligible_not_semantically_verified(self):
        source = 'Step 1: IQueryable 星雲 toaster nonsense.\nStep 2: Purple syntax 随机 queue banana.\nFinal Answer: \\boxed{4}'
        scored = scoring.annotate(self.tokenizer, [], self.tokenizer.encode(source), 4, 'stop')
        self.assertTrue(scored['answer_correct'])
        self.assertTrue(scored['stride_eligible'])
        # This documents a known limitation; syntax/answer checks are not judges.

    @unittest.skipUnless(os.environ.get('STRIDE_TEST_TOKENIZER_PATH'), 'Set STRIDE_TEST_TOKENIZER_PATH to a pinned local Qwen tokenizer snapshot')
    def test_actual_qwen_tokenizer_unicode_thinking_and_boundaries(self):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(os.environ['STRIDE_TEST_TOKENIZER_PATH'], local_files_only=True)
        for source in (CANONICAL, CANONICAL.replace('Compute', '中文 α 😀 Compute'), '<think>Internal calculation.</think>\n' + CANONICAL):
            ids = tokenizer.encode(source, add_special_tokens=False)
            scored = scoring.annotate(tokenizer, [], ids, 4, 'stop')
            self.assertTrue(scored['answer_correct'])
            self.assertTrue(scored['mapping_verified'], scored)
            self.assertTrue(scored['stride_eligible'], scored)
            for a, b in scored['spans']:
                self.assertNotIn(tokenizer.eos_token_id, ids[a:b + 1])
                self.assertNotIn(r'\boxed', tokenizer.decode(ids[a:b + 1]))


class IntegerAndRelaxedGraderTests(unittest.TestCase):
    def test_exact_integer_normalization_table(self):
        cases = {'4': 4, '+004': 4, '4.000': 4, '1,024': 1024, '−4': -4,
                 r'\frac{8}{2}': 4, r'-\dfrac{8}{2}': -4, r'\text{4}': 4,
                 '$4$': 4, '.5': Fraction(1, 2), '-.25': Fraction(-1, 4)}
        for value, expected in cases.items():
            with self.subTest(value=value):
                self.assertEqual(integer._numeric_value(value), expected)
        for value in ('', 'NaN', '1e3', '1+3', '1,00', '٤', r'\frac{1}{0}', r'\frac{1}{\frac{2}{3}}', '9' * 513, '$4', r'\text{4}x'):
            self.assertIsNone(integer._numeric_value(value))
        self.assertIsNone(integer._numeric_value('4', depth=9))
        self.assertEqual(integer._plain_number('1,024.50'), Fraction(2049, 2))
        self.assertIsNone(integer._plain_number('1,2'))

    def test_ground_truth_types_and_bounds(self):
        self.assertEqual(integer._ground_truth(' +0004 '), '4')
        for value in (True, 4.0, None, '4.0', '1/1', '9' * 513, '٤'):
            with self.subTest(value=repr(value)[:30]), self.assertRaises(ValueError):
                integer._ground_truth(value)
        with self.assertRaises(TypeError):
            integer.verify_answer(None, 4)
        with self.assertRaises(TypeError):
            integer.verify_answer(r'\boxed{4}', 4, finish_reason=3)

    def test_braces_escape_and_wrapper_helpers(self):
        self.assertEqual(integer._braced(r'{a\{b\}c}', 0), (r'a\{b\}c', 9))
        self.assertIsNone(integer._braced('{a', 0))
        self.assertIsNone(integer._braced('x', 0))
        self.assertTrue(integer._escaped(r'\{', 1))
        self.assertFalse(integer._escaped(r'\\{', 2))
        self.assertEqual(integer._immediate_outer_wrapper('math $'), ('$', '$'))
        self.assertIsNone(integer._immediate_outer_wrapper('$x=1$'))
        self.assertEqual(integer._immediate_outer_wrapper(r'\['), (r'\[', r'\]'))
        self.assertEqual(integer._result(True, '4', 'correct', '4')['answer_correct'], True)

    def test_strict_answer_contract_adversaries(self):
        for source in (r'\boxed{4}', r'$\boxed{4}$', r'\[\boxed{4}\]', r'$x=1$ \boxed{4}', '\\boxed{4}\n</solution>', r'\boxed{4}<|im_end|>'):
            with self.subTest(source=source):
                self.assertTrue(integer.verify_answer(source, 4)['answer_correct'])
        cases = {'missing_box': 'Answer: 4', 'multiple_or_nested_boxes': r'\boxed{4} \boxed{4}',
                 'malformed_or_unbalanced_box': r'\boxed{4', 'text_after_box': r'\boxed{4} because',
                 'unmatched_math_wrapper_or_tail': r'$\boxed{4}',
                 'interior_or_unknown_special_token': r'<|bad|> \boxed{4}',
                 'misplaced_solution_tag': '<solution>\\boxed{4}',
                 'unsupported_or_noninteger_answer': r'\boxed{0.5}'}
        for reason, source in cases.items():
            result = integer.verify_answer(source, 4)
            self.assertFalse(result['answer_correct'])
            self.assertEqual(result['reason'], reason)
        wrong = integer.verify_answer(r'\boxed{5}', 4)
        self.assertTrue(wrong['answer_contract_valid'])
        self.assertFalse(wrong['answer_correct'])

    def test_relaxed_surface_exactness_and_math_unit_guards(self):
        self.assertEqual(relaxed.normalize_surface(r' ** $\displaystyle −4$ **.;'), '-4')
        self.assertEqual(relaxed.numeric_value('x = 8/2'), 4)
        self.assertIsNone(relaxed.numeric_value('4/0'))
        self.assertTrue(relaxed.equivalent('8/2', '4'))
        self.assertFalse(relaxed.equivalent('1/2', '.5', 'integer'))
        self.assertTrue(relaxed.equivalent('1/2', '.5', 'math'))
        self.assertFalse(relaxed.equivalent(r'50\%', '50', 'math'))
        self.assertFalse(relaxed.equivalent('90°', '90', 'math'))
        self.assertTrue(relaxed.equivalent(r'\sqrt2', r'\sqrt{2}', 'math'))
        for fn in (relaxed.equivalent,):
            with self.assertRaises(ValueError):
                fn('4', '4', 'scienceqa')

    def test_relaxed_terminal_declarations_and_conflicts(self):
        positives = ('Answer: 4', 'The final answer is **4**.', 'Final result = 8/2', r'\fbox{4}',
                     'Answer: 4\nAnswer: 4', r'\boxed{4}.' , '<solution>Answer: 4</solution>')
        for source in positives:
            result = relaxed.verify_relaxed_answer(source, 4)
            self.assertTrue(result['answer_correct'], (source, result))
        negatives = ('The target 4 appears here.', 'Answer: 4 or 5', 'Answer: 4\nAnswer: 5',
                     'Answer: 4\nextra text', r'\boxed{4} correction 5', r'\boxed{\boxed{4}}',
                     r'\boxed 4', '<|bad|>Answer: 4', 'Answer: approximately 4', 'Answer: 4 meters')
        for source in negatives:
            self.assertFalse(relaxed.verify_relaxed_answer(source, 4)['answer_correct'], source)
        repeated = relaxed.extract_final_answer('Answer: 4\nAnswer: 4')
        self.assertEqual(repeated['flags'], ['repeated_equivalent_answer'])
        for target in (4, 5):
            self.assertEqual(relaxed.verify_relaxed_answer('Answer: 4\nAnswer: 5', target)['reason'], 'conflicting_or_unresolved_answer_declarations')

    def test_relaxed_transport_plausibility_and_type_checks(self):
        text = '<solution>Answer: 4</solution><|im_end|>'
        cleaned = relaxed._clean_transport(text)
        self.assertEqual(cleaned.index('Answer'), text.index('Answer'))
        self.assertIsNone(relaxed._clean_transport('Answer<|im_end|>: 4'))
        self.assertTrue(relaxed._box_tail(r'\]** .!;'))
        self.assertFalse(relaxed._box_tail('correction'))
        self.assertTrue(relaxed._plausible('x+1', 'math'))
        self.assertFalse(relaxed._plausible('maybe x', 'math'))
        self.assertFalse(relaxed._plausible('length', 'math'))
        self.assertFalse(relaxed._plausible('9' * 1025, 'integer'))
        with self.assertRaises(TypeError):
            relaxed.extract_final_answer(None)
        with self.assertRaises(ValueError):
            relaxed.extract_final_answer('4', 'other')

    def test_relaxed_truncation_tracks_candidate_without_reward(self):
        for reason in integer._LENGTH_FINISHES:
            result = relaxed.verify_relaxed_answer('Answer: 4', 4, reason)
            self.assertTrue(result['candidate_answer_correct'])
            self.assertFalse(result['answer_correct'])
            self.assertFalse(result['answer_contract_valid'])
            self.assertTrue(result['generation_truncated'])
        self.assertTrue(relaxed.verify_strict_answer(r'\boxed{4}', 4)['answer_correct'])
        self.assertTrue(relaxed.verify_strict_answer(r'\boxed{\sqrt2}', r'\sqrt{2}', task='math')['answer_correct'])
        for source in (r'\boxed{4}\boxed{4}', r'\boxed{4', r'\boxed{<|bad|>}', r'\boxed{4} tail'):
            self.assertFalse(relaxed.verify_strict_answer(source, '4', task='math')['answer_correct'])
        with self.assertRaises(ValueError):
            relaxed.verify_strict_answer('4', 4, task='other')

    def test_legacy_math_helpers_without_enabling_structure_reward(self):
        self.assertEqual(legacy.last_boxed_only_string(r'a \boxed{1} b \boxed{2}'), r'\boxed{2}')
        self.assertIsNone(legacy.last_boxed_only_string('no box'))
        self.assertIsNone(legacy.last_boxed_only_string(r'\boxed{'))
        self.assertEqual(legacy.remove_boxed(r'\boxed{4}'), '4')
        self.assertEqual(legacy.remove_boxed(r'\boxed 4'), '4')
        with self.assertRaises(AssertionError):
            legacy.remove_boxed('4')
        self.assertEqual(legacy.fix_fracs(r'\frac12'), r'\frac{1}{2}')
        self.assertEqual(legacy.fix_a_slash_b('2/3'), r'\frac{2}{3}')
        self.assertEqual(legacy.fix_a_slash_b('2/03'), '2/03')
        self.assertEqual(legacy.remove_right_units(r'4\text{ meters}'), '4')
        self.assertEqual(legacy.fix_sqrt(r'\sqrt2'), r'\sqrt{2}')
        self.assertEqual(legacy.strip_string('x = .5'), r'\frac{1}{2}')
        self.assertTrue(legacy.is_equiv(r'\frac12', '0.5'))
        self.assertFalse(legacy.is_equiv(None, '0'))
        self.assertEqual(legacy._compute_score_base(r'\boxed{4}', '4'), 1.0)
        with patch.dict(os.environ, {'STRIDE_STRUCTURE_COEF': '0'}):
            self.assertEqual(legacy.compute_score(r'\boxed{4}', '4'), 1.0)
        with patch.dict(os.environ, {'STRIDE_STRUCTURE_COEF': 'invalid'}):
            self.assertEqual(legacy.compute_score(r'\boxed{5}', '4'), 0.0)


class AggregateMetricTests(unittest.TestCase):
    def test_pass_at_k_matches_enumerated_oracle(self):
        for n in range(1, 10):
            for c in range(n + 1):
                population = list(range(n))
                for k in range(1, n + 1):
                    subsets = list(itertools.combinations(population, k))
                    expected = sum(any(i < c for i in subset) for subset in subsets) / len(subsets)
                    self.assertAlmostEqual(evaluate._pass_at_k(n, c, k), expected, places=14)
        self.assertIsNone(evaluate._pass_at_k(7, 3, 8))

    def test_macro_pass1_differs_from_pooled_with_unequal_k(self):
        groups = [group('a', [True]), group('b', [False] * 8)]
        summary = evaluate.summarize(groups)
        self.assertEqual(summary['relaxed']['pass1'], .5)
        self.assertEqual(summary['relaxed']['pooled_correct_fraction'], 1 / 9)
        self.assertIsNone(summary['relaxed']['pass8'])
        self.assertEqual(summary['relaxed']['observed_any_correct_fraction'], .5)
        self.assertEqual(summary['rollout_count_histogram'], {'1': 1, '8': 1})

    def test_pass8_exact8_and_eligibility_counts(self):
        groups = [group('a', [True, True] + [False] * 6), group('b', [False] * 8)]
        groups[0]['records'][1]['score']['step_eligible'] = False
        summary = evaluate.summarize(groups)
        self.assertEqual(summary['relaxed']['pass1'], .125)
        self.assertEqual(summary['relaxed']['pass8'], .5)
        self.assertEqual(summary['relaxed']['at_least_two_correct'], 1)
        self.assertEqual(summary['relaxed']['at_least_two_with_declared_steps'], 0)
        self.assertEqual(summary['relaxed']['correct_with_usable_steps'], 1)
        self.assertEqual(summary['questions'], 2)
        self.assertEqual(summary['responses'], 16)

    def test_normalized_explanation_excludes_answers_and_thinking(self):
        first = record(CANONICAL)
        second = record(CANONICAL.replace('Compute', 'COMPUTE').replace(' ', '  ').replace(r'\boxed{4}', r'\boxed{5}'))
        self.assertEqual(evaluate.normalized_explanation(first), evaluate.normalized_explanation(second))
        text = '<think>Different invisible reasoning</think>' + CANONICAL
        thinking = record(text)
        thinking['score']['region']['final'] = (text.index('Step 1:'), len(text))
        thinking['score']['relaxed'] = relaxed.extract_final_answer(CANONICAL)
        self.assertEqual(evaluate.normalized_explanation(first), evaluate.normalized_explanation(thinking))
        no_header = record('A calculation.\n\\boxed{4}')
        no_header['score']['relaxed'] = {}
        self.assertEqual(evaluate.normalized_explanation(no_header), 'a calculation.')
        self.assertIsNone(evaluate.normalized_explanation(record('Answer: 4')))

    def test_lexical_distinctness_is_counted_within_each_question(self):
        groups = [group('a', [True, True]), group('b', [True, True])]
        groups[0]['records'][1] = record(CANONICAL.replace('2 + 2', '1 + 3'), sample_index=1)
        summary = evaluate.summarize(groups)
        lexical = summary['relaxed']['lexical_final_explanation']
        self.assertEqual(lexical['distinct_normalized_texts_sum_over_questions'], 3)
        self.assertEqual(lexical['distinct_fraction_pooled'], .75)
        self.assertEqual(lexical['questions_with_two_distinct_correct_texts'], 1)
        self.assertTrue(any('not verified reasoning diversity' in x for x in summary['limitations']))

    def test_aggregate_rejects_corrupt_identity_indices_metadata_and_correctness(self):
        duplicate = [group('a', [True]), group('a', [False])]
        with self.assertRaisesRegex(ValueError, 'Duplicate evaluation'):
            evaluate.summarize(duplicate)
        bad_cases = []
        empty = group('a', []); bad_cases.append(empty)
        missing = group('a', [True]); missing['records'][0]['sample_index'] = 2; bad_cases.append(missing)
        token_count = group('a', [True]); token_count['records'][0]['generated_tokens'] = 999; bad_cases.append(token_count)
        text_change = group('a', [True]); text_change['records'][0]['text'] = 'different'; bad_cases.append(text_change)
        correctness = group('a', [True]); correctness['records'][0]['score']['strict']['answer_correct'] = False; bad_cases.append(correctness)
        for case in bad_cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                evaluate.summarize([case])
        with self.assertRaisesRegex(ValueError, 'Missing strict correctness'):
            evaluate._correct({'score': {}}, 'strict')
        item = record(); item['score'].pop('strict_answer_correct')
        self.assertTrue(evaluate._correct(item, 'strict'))

    def test_all_truncation_reasons_in_aggregate_fallback(self):
        for reason in integer._LENGTH_FINISHES:
            item = group('a', [False])
            item['records'][0]['finish_reason'] = reason
            self.assertNotIn('truncated', item['records'][0]['score'])
            self.assertEqual(evaluate.summarize([item])['truncations'], 1)

    def test_empty_summary_quantiles_context_and_diagnostic_counts(self):
        self.assertEqual(evaluate.summarize([])['responses'], 0)
        self.assertIsNone(evaluate.summarize([])['relaxed']['pass1'])
        self.assertIsNone(evaluate._ratio(0, 0))
        self.assertEqual(evaluate._ratio(1, 2), .5)
        self.assertIsNone(evaluate._quantile([], .5))
        self.assertEqual(evaluate._quantile([1, 3, 5, 7], .5), 4)
        item = group('a', [True]); item['prompt_token_ids'] = [0] * 8190
        item['records'][0]['score']['diagnostic_categories'] = ['x', 'x']
        item['records'][0]['score']['truncated'] = True
        summary = evaluate.summarize([item])
        self.assertEqual(summary['responses_over_8192_total'], 1)
        self.assertEqual(summary['syntax']['diagnostic_category_counts']['x'], 1)
        self.assertEqual(summary['truncations'], 1)
        bad = record(); bad['score']['region']['final'] = (0, 9999)
        with self.assertRaises(ValueError):
            evaluate.normalized_explanation(bad)


if __name__ == '__main__':
    unittest.main()
