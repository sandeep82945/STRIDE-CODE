"""Conservative final-answer diagnostic, with extraction independent of target.

No proof checking is performed. Truncated candidate matches are separate from
completed matches. Conflicting boxed/explicit final declarations fail closed.
Unrecognized prose, implicit bare numbers, units and symbolic equivalences can
be false negatives. This is deliberately stricter than searching for the target
anywhere in the response or choosing whichever of multiple answers matches it.

Dependencies are unchanged local copies of the prior integer verifier and the
production benchmark MATH normalizer. No model or third-party package is used.
"""

import re
from fractions import Fraction
from . import integer_reference as integer
from .math_reference import is_equiv, strip_string

_BOX = re.compile(r"\\(?:boxed|fbox)\b")
_DECLARATION = re.compile(
    r"\b(?:(?:the|our)\s+)?(?:final\s+answer|answer|final\s+result)"
    r"\s*(?:\*\*)?\s*(?:(?:is|equals)\s*:?\s*|[:=]\s*)(?:\*\*\s+)?", re.I
)
_EOS = integer._EOS + ("<|end|>",)
_FINISH_LENGTH = integer._LENGTH_FINISHES
_WRAPPERS = (("$$", "$$"), (r"\[", r"\]"), (r"\(", r"\)"), ("$", "$"), ("**", "**"))


def normalize_surface(value):
    """Strip unambiguous presentation only; never evaluate arbitrary text."""
    value = value.strip().replace("\N{MINUS SIGN}", "-")
    for _ in range(8):
        old = value
        value = value.rstrip(".;").strip()
        for left, right in _WRAPPERS:
            if value.startswith(left) and value.endswith(right) and len(value) > len(left) + len(right):
                value = value[len(left):-len(right)].strip()
                break
        value = re.sub(r"^\\displaystyle\s*", "", value)
        if value == old:
            break
    return value


def numeric_value(value):
    value = normalize_surface(value)
    # Single variable assignment is a final-answer presentation, not a search.
    value = re.sub(r"^[A-Za-z]\s*=\s*", "", value)
    result = integer._numeric_value(value)
    if result is not None:
        return result
    if re.fullmatch(r"[+-]?\d+\s*/\s*[+-]?\d+", value):
        top, bottom = value.split("/")
        if int(bottom) != 0:
            return Fraction(int(top), int(bottom))
    return None


def equivalent(left, right, task="integer"):
    """Exact numeric comparison; otherwise existing normalized MATH matching."""
    if task not in ("integer", "math"):
        raise ValueError("task must be integer or math")
    left, right = normalize_surface(str(left)), normalize_surface(str(right))
    a, b = numeric_value(left), numeric_value(right)
    if a is not None and b is not None:
        return bool(a == b and (task == "math" or a.denominator == 1))
    if task == "integer":
        return False
    # The legacy benchmark strips percentages and degree markers. Do not let
    # that produce an apparent equivalence where an explicit unit differs.
    if any(marker in left or marker in right for marker in ("%", r"\circ", "°")):
        return left == right
    try:
        return bool(is_equiv(left, right))
    except Exception:
        return left == right


def _plausible(value, task):
    value = normalize_surface(value)
    if not value or len(value) > 1024 or "\n\n" in value:
        return False
    if task == "integer":
        n = numeric_value(value)
        return n is not None and n.denominator == 1
    if re.search(r"\b(?:or|because|but|maybe|approximately|not|either)\b", value, re.I):
        return False
    # Multiword natural-language explanations are not mathematical answers.
    without_commands = re.sub(r"\\[A-Za-z]+", "", value)
    if re.search(r"[A-Za-z]{3,}", without_commands):
        return False
    return not bool(re.search(r"[<>\n]", value))


def _clean_transport(text):
    body = text.rstrip()
    for _ in range(10):
        old = body
        for token in _EOS:
            if body.endswith(token):
                body = body[:-len(token)].rstrip()
                break
        body = re.sub(r"\s*</solution>\s*$", "", body, flags=re.I).rstrip()
        if body == old:
            break
    # Preserve all character offsets into the original response text.
    body = re.sub(r"^\s*<solution>\s*", lambda m: " " * len(m.group()), body, flags=re.I)
    if integer._SPECIAL.search(body) or re.search(r"</?solution\b", body, re.I):
        return None
    return body


def _box_tail(tail):
    # Mild punctuation and math/Markdown wrapper closures are harmless. No
    # prose after a box is silently ignored, including a later correction.
    return not re.sub(r"\\\]|\\\)|\$|\*\*|[\s.!;]", "", tail)


def extract_final_answer(text, task="integer"):
    """Return a target-blind terminal candidate and explicit ambiguity flags."""
    if task not in ("integer", "math"):
        raise ValueError("task must be integer or math")
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    result = dict(extraction_valid=False, extracted_answer=None, source=None, final_answer_start=None,
                  reason="no_explicit_terminal_answer", flags=[], declarations=0)
    body = _clean_transport(text)
    if body is None:
        return {**result, "reason": "interior_transport_token"}
    candidates = []
    spans = []
    for match in _BOX.finditer(body):
        if integer._escaped(body, match.start()):
            continue
        pos = match.end()
        while pos < len(body) and body[pos].isspace():
            pos += 1
        boxed = integer._braced(body, pos)
        if boxed is None:
            return {**result, "reason": "malformed_box", "flags": ["malformed_box"]}
        value, end = boxed
        if _BOX.search(value):
            return {**result, "reason": "nested_box", "flags": ["nested_box"]}
        candidates.append(dict(value=normalize_surface(value), source="boxed",
                               terminal=_box_tail(body[end:]), start=match.start()))
        spans.append((match.start(), end))
    declarations = list(_DECLARATION.finditer(body))
    for index, match in enumerate(declarations):
        if any(start <= match.start() < end for start, end in spans):
            continue
        # Keep the complete final declaration, never just its first numeric
        # substring. Earlier declarations end at their line or the next one.
        end = len(body)
        if index + 1 < len(declarations):
            end = declarations[index + 1].start()
        newline = body.find("\n", match.end())
        if newline >= 0 and body[match.end():newline].strip():
            end = min(end, newline)
        value = body[match.end():end].strip()
        if _BOX.search(value):
            continue  # The box was already recorded, including its full tail.
        terminal = not body[end:].strip()
        candidates.append(dict(value=normalize_surface(value), source="explicit_unboxed",
                               terminal=terminal, start=match.start()))
    result["declarations"] = len(candidates)
    terminals = [c for c in candidates if c["terminal"]]
    if not terminals:
        if candidates:
            result["reason"] = "substantive_text_after_answer"
        return result
    candidate = max(terminals, key=lambda c: c["start"])
    result.update(extracted_answer=candidate["value"], source=candidate["source"],
                  final_answer_start=candidate["start"])
    if not _plausible(candidate["value"], task):
        return {**result, "reason": "unsupported_final_expression"}
    if any(not _plausible(c["value"], task) or not equivalent(c["value"], candidate["value"], task)
           for c in candidates):
        return {**result, "reason": "conflicting_or_unresolved_answer_declarations",
                "flags": ["conflicting_or_unresolved_answer_declarations"]}
    flags = ["repeated_equivalent_answer"] if len(candidates) > 1 else []
    return {**result, "extraction_valid": True, "reason": "extracted", "flags": flags}


def verify_relaxed_answer(text, ground_truth, finish_reason=None, task="integer"):
    result = extract_final_answer(text, task)
    truncated = finish_reason in _FINISH_LENGTH
    candidate_correct = bool(result["extraction_valid"] and
                             equivalent(result["extracted_answer"], ground_truth, task))
    result.update(candidate_answer_correct=candidate_correct,
                  answer_correct=bool(candidate_correct and not truncated),
                  answer_contract_valid=bool(result["extraction_valid"] and not truncated),
                  generation_truncated=truncated,
                  normalized_answer=result["extracted_answer"])
    if truncated:
        result["reason"] = "generation_truncated"
    elif result["extraction_valid"]:
        result["reason"] = "correct" if candidate_correct else "wrong_or_unrecognized_equivalence"
    return result


def verify_strict_answer(text, ground_truth, finish_reason=None, task="integer"):
    """Preserve the previous strict boxed contract for matched-text scoring."""
    if task == "integer":
        return integer.verify_answer(text, ground_truth, finish_reason)
    if task != "math":
        raise ValueError("task must be integer or math")
    fail = dict(answer_correct=False, answer_contract_valid=False,
                normalized_answer=None, reason="missing_multiple_or_malformed_box")
    boxes = [m for m in integer._BOX_COMMAND.finditer(text) if not integer._escaped(text, m.start())]
    if len(boxes) != 1:
        return fail
    pos = boxes[0].end()
    while pos < len(text) and text[pos].isspace():
        pos += 1
    boxed = integer._braced(text, pos)
    if boxed is None or integer._SPECIAL.search(boxed[0]):
        return fail
    content, end = boxed
    structural = integer.verify_answer(text[:pos] + "{0}" + text[end:], 0, finish_reason)
    if not structural["answer_contract_valid"]:
        return {**structural, "answer_correct": False}
    a, b = integer._numeric_value(content), integer._numeric_value(str(ground_truth))
    correct = a == b if a is not None and b is not None else is_equiv(content, str(ground_truth))
    try:
        normalized = strip_string(content)
    except (ValueError, AssertionError, TypeError, IndexError):
        normalized = content.strip()
    return dict(answer_correct=bool(correct), answer_contract_valid=True,
                normalized_answer=normalized,
                reason="correct" if correct else "wrong_or_unrecognized_equivalence")
