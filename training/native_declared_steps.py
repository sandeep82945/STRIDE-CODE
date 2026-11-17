"""Frozen post-hoc syntax parser; original text and offsets are preserved."""
import re
import string
import unicodedata
from native_step_parser import Step

PARSER_VERSION = 2
PREFIX = r'^[ \t]*(?:#{1,6}[ \t]*)?(?:\*\*)?'
# Detect malformed labels as well as valid markers so they cannot be silently
# absorbed into another step body. Ordinary prose such as "Step through this"
# is not a heading unless the word following Step has a heading delimiter.
STEP_LIKE = re.compile(r'(?im)' + PREFIX + r'Step[ \t]*(?:[+-]?\d[\w.]*\b|\w+(?=[ \t]*(?:\*\*)?[ \t]*(?:[:.)–—-]|$)))')
STEP_MARKER = re.compile(r'(?im)' + PREFIX + r'Step[ \t]*([1-9][0-9]*)\b[ \t]*(?:\*\*)?[ \t]*(?::|[.)](?![0-9])|[-–—])(?:\*\*)?[ \t]*')
FINAL_ANSWER = re.compile(r'(?im)' + PREFIX + r'Final[ \t]+Answer(?:\*\*)?[ \t]*(?::|$)')

def _has_content(body):
    # This rejects formatting-only bodies, without assessing mathematical
    # correctness or requiring English words. Unicode math symbols survive.
    return any(not c.isspace() and c not in string.punctuation and
               unicodedata.category(c)[0] not in 'PZC' for c in body)

def permissive_steps(text, left, right):
    """Return consecutive nonempty declared steps using original char spans.

    Caller retains the frozen thinking-closure/truncation gates and verifies
    these spans against the actual generated token IDs before eligibility.
    This function does not evaluate answers or prove reasoning validity.
    """
    if not 0 <= left <= right <= len(text):
        return []
    region = text[left:right]
    answer = FINAL_ANSWER.search(region)
    if answer:
        right = left + answer.start()
        region = text[left:right]
    markers = list(STEP_MARKER.finditer(region))
    # Malformed huge labels must fail closed rather than exceed Python's
    # integer-conversion limit and interrupt an otherwise valid rollout group.
    if any(len(x.group(1)) > 128 for x in markers):
        return []
    if len(markers) < 2 or [int(x.group(1)) for x in markers] != list(range(1, len(markers) + 1)):
        return []
    if len(list(STEP_LIKE.finditer(region))) != len(markers):
        return []
    result = []
    for i, marker in enumerate(markers):
        a = left + marker.end()
        b = left + markers[i + 1].start() if i + 1 < len(markers) else right
        while a < b and text[a].isspace(): a += 1
        while b > a and text[b - 1].isspace(): b -= 1
        if a == b or not _has_content(text[a:b]): return []
        result.append(Step(i + 1, text[a:b], a, b, left + marker.start(), left + marker.end()))
    return result
