"""Exact terminal integer-answer verification, independent of reasoning parsing.

This checks the final-answer contract, not the validity of preceding reasoning.
Only the Python standard library is used. No expressions are evaluated. Numeric
strings are bounded at 512 characters, and normalization uses exact fractions.
The caller must pass the actual generation finish reason so capped generations
cannot receive reward merely for containing a correct box.
"""

import re
from fractions import Fraction


_BOX_COMMAND = re.compile(r"\\boxed\b")
_SPECIAL = re.compile(r"<\|[^<>\r\n]*\|>")
_EOS = ("<|endoftext|>", "<|im_end|>", "<|eot_id|>")
_WRAPPERS = (("$$", "$$"), (r"\[", r"\]"), (r"\(", r"\)"), ("$", "$"))
_INTEGER = re.compile(r"[+-]?[0-9]+\Z")
_DECIMAL = re.compile(
    r"(?P<sign>[+-]?)(?P<whole>[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)"
    r"(?:\.(?P<fraction>[0-9]*))?\Z"
)
_LEADING_DECIMAL = re.compile(r"(?P<sign>[+-]?)\.(?P<fraction>[0-9]+)\Z")
_FRACTION = re.compile(r"(?P<sign>[+-]?)\\(?:dfrac|tfrac|frac)\b")
_LENGTH_FINISHES = frozenset(("length", "max_length", "max_tokens", "max_new_tokens"))
MAX_NUMERIC_CHARACTERS = 512


def _escaped(text, index):
    """Whether a backslash/brace is escaped by an odd preceding slash run."""
    count = 0
    index -= 1
    while index >= 0 and text[index] == "\\":
        count += 1
        index -= 1
    return bool(count % 2)


def _braced(text, start):
    """Return (unchanged inner content, end-exclusive) or None."""
    if start >= len(text) or text[start] != "{":
        return None
    depth = 0
    for index in range(start, len(text)):
        if _escaped(text, index):
            continue
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:index], index + 1
    return None


def _plain_number(text):
    match = _DECIMAL.fullmatch(text)
    if match:
        whole = match.group("whole").replace(",", "")
        fractional = match.group("fraction") or ""
        numerator = int(whole + fractional)
    else:
        match = _LEADING_DECIMAL.fullmatch(text)
        if not match:
            return None
        fractional = match.group("fraction")
        numerator = int(fractional)
    if match.group("sign") == "-":
        numerator = -numerator
    return Fraction(numerator, 10 ** len(fractional))


def _numeric_value(text, depth=0):
    """Parse the explicitly supported numeric grammar, never arbitrary math."""
    if depth > 8:
        return None
    text = text.strip().replace("\N{MINUS SIGN}", "-")
    if not text or len(text) > MAX_NUMERIC_CHARACTERS:
        return None
    for opening, closing in _WRAPPERS:
        if text.startswith(opening):
            if not text.endswith(closing) or len(text) < len(opening) + len(closing):
                return None
            return _numeric_value(text[len(opening):-len(closing)], depth + 1)
    if text.startswith(r"\text"):
        match = re.match(r"\\text\s*", text)
        if not match:
            return None
        inner = _braced(text, match.end())
        if inner is None or inner[1] != len(text):
            return None
        return _numeric_value(inner[0], depth + 1)
    fraction = _FRACTION.match(text)
    if fraction:
        index = fraction.end()
        while index < len(text) and text[index].isspace():
            index += 1
        numerator = _braced(text, index)
        if numerator is None:
            return None
        index = numerator[1]
        while index < len(text) and text[index].isspace():
            index += 1
        denominator = _braced(text, index)
        if denominator is None or text[denominator[1]:].strip():
            return None
        # Operands are plain decimal numbers: no nested fractions or expressions.
        top = _plain_number(numerator[0].strip())
        bottom = _plain_number(denominator[0].strip())
        if top is None or bottom is None or bottom == 0:
            return None
        result = top / bottom
        return -result if fraction.group("sign") == "-" else result
    return _plain_number(text)


def _ground_truth(value):
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError("Ground truth must be a signed integer string or integer")
    raw = str(value).strip()
    if len(raw) > MAX_NUMERIC_CHARACTERS or not _INTEGER.fullmatch(raw):
        raise ValueError("Ground truth must be a signed integer string or integer")
    return str(int(raw))


def _immediate_outer_wrapper(prefix):
    """Identify an opener directly before the box, not a prior math close."""
    for opening, closing in _WRAPPERS:
        start = len(prefix) - len(opening)
        if start < 0 or not prefix.endswith(opening) or _escaped(prefix, start):
            continue
        if opening in ("$", "$$"):
            # `$x=1$ \\boxed{42}` has prior closed math, not an outer opener.
            # Greedy tokenization keeps display-dollar pairs out of `$` parity.
            tokens = [match.group() for match in re.finditer(r"\$\$|\$", prefix)
                      if not _escaped(prefix, match.start())]
            if not tokens or tokens[-1] != opening or tokens.count(opening) % 2 == 0:
                continue
        return opening, closing
    return None


def _result(valid, normalized, reason, target):
    return {
        "answer_correct": bool(valid and normalized == target),
        "answer_contract_valid": bool(valid),
        "normalized_answer": normalized,
        "reason": reason,
    }


def verify_answer(text, ground_truth, finish_reason=None):
    """Verify one terminal box, with optional wrapper closure and terminal EOS.

    The output may contain arbitrary reasoning before the box. An immediately
    enclosing math wrapper must be paired. A closing </solution> must occupy
    its own line. Repeated/nested boxes, interior transport tokens, malformed
    numeric contents and any substantive suffix fail closed. A box with a
    well-formed but wrong integer remains contract-valid.
    """
    target = _ground_truth(ground_truth)
    if not isinstance(text, str):
        raise TypeError("Response text must be a string")
    if finish_reason is not None and not isinstance(finish_reason, str):
        raise TypeError("finish_reason must be a string or None")
    if finish_reason in _LENGTH_FINISHES:
        return _result(False, None, "generation_truncated", target)

    body = text.rstrip()
    while True:
        eos = next((token for token in _EOS if body.endswith(token)), None)
        if eos is None:
            break
        body = body[:-len(eos)].rstrip()
    if _SPECIAL.search(body):
        return _result(False, None, "interior_or_unknown_special_token", target)

    closure = re.search(r"(?:^|\n)[ \t]*</solution>[ \t]*\Z", body)
    if closure:
        body = body[:closure.start()].rstrip()
    if re.search(r"<\s*/?\s*solution\b", body, flags=re.IGNORECASE):
        return _result(False, None, "misplaced_solution_tag", target)

    boxes = [match for match in _BOX_COMMAND.finditer(body)
             if not _escaped(body, match.start())]
    if not boxes:
        return _result(False, None, "missing_box", target)
    if len(boxes) != 1:
        return _result(False, None, "multiple_or_nested_boxes", target)
    box = boxes[0]
    index = box.end()
    while index < len(body) and body[index].isspace():
        index += 1
    braced = _braced(body, index)
    if braced is None:
        return _result(False, None, "malformed_or_unbalanced_box", target)
    content, end = braced
    prefix, suffix = body[:box.start()].rstrip(), body[end:].strip()
    wrapper = _immediate_outer_wrapper(prefix)
    if wrapper is not None:
        if suffix != wrapper[1]:
            return _result(False, None, "unmatched_math_wrapper_or_tail", target)
        suffix = ""
    if suffix:
        return _result(False, None, "text_after_box", target)

    numeric = _numeric_value(content)
    if numeric is None or numeric.denominator != 1:
        return _result(False, None, "unsupported_or_noninteger_answer", target)
    normalized = str(numeric.numerator)
    return _result(True, normalized, "correct" if normalized == target else "wrong_answer", target)
