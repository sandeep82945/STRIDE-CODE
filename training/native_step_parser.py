"""Strict, span-preserving reasoning parser and original-ID token mapper.

No model/runtime dependencies. Character spans are Python [start, end) indices
into the unmodified input. Token spans are original-response-relative inclusive
indices. Parsing validates structure, never the mathematical truth of a step.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any, Optional, Sequence


@dataclass(frozen=True)
class Issue:
    code: str
    severity: str
    message: str
    start: Optional[int] = None
    end: Optional[int] = None


@dataclass(frozen=True)
class Step:
    number: int
    text: str
    start: int
    end: int
    marker_start: int
    marker_end: int


@dataclass(frozen=True)
class FinalAnswer:
    text: str
    start: int
    end: int
    box_start: int
    box_end: int
    content_start: int
    content_end: int


@dataclass
class ParseResult:
    source: str
    status: str
    steps: list[Step] = field(default_factory=list)
    final_answer: Optional[FinalAnswer] = None
    issues: list[Issue] = field(default_factory=list)

    @property
    def canonical(self) -> bool:
        return self.status == "canonical"

    @property
    def valid(self) -> bool:
        return self.status in ("canonical", "recoverable")

    def to_dict(self) -> dict:
        result = asdict(self)
        result.update(valid=self.valid, canonical=self.canonical)
        return result


# Recovery is deliberately limited: indentation, heading prefixes, case,
# punctuation '.', or a bold MARKER such as '**Step 1:**'. No paragraph fallback.
_MARKER = re.compile(
    r"^(?P<prefix>[ \t]*(?:\#{1,6}[ \t]+)?)(?P<bold>\*\*)?"
    r"(?P<label>Step[ \t]+(?P<number>[0-9]+)[ \t]*[:.]|Final[ \t]+Answer[ \t]*:)"
    r"(?P<bold_end>\*\*)?", re.IGNORECASE,
)
_MARKER_LIKE = re.compile(r"^[ \t]*(?:\#{1,6}[ \t]+)?(?:\*\*)?(?:Step(?:\s|[0-9])|Final[ \t]+Answer)", re.IGNORECASE)
_INLINE_STEP = re.compile(r"\bStep[ \t]+(?P<number>[0-9]+)[ \t]*[:.]", re.IGNORECASE)
# A numbered mention is not necessarily a delimiter: "from Step 3." and
# "in Step 3:" refer to existing reasoning. Keep the exemption narrow and
# observable: require a grammatical introducer plus an already declared step.
# Coordinated references such as "from Step 2 and Step 3:" are supported.
_STEP_REFERENCE_PREFIX = re.compile(
    r"\b(?:from|in|to|of|using|see|per|by|at)[ \t]+"
    r"(?:Step[ \t]+[0-9]+[ \t]*(?:,[ \t]*(?:(?:and|or)[ \t]+)?|(?:and|or)[ \t]+))*\Z",
    re.IGNORECASE,
)
_FENCE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})(.*)$")
_BOX_COMMAND = re.compile(r"\\boxed\b")
_SPECIAL_TOKEN = re.compile(r"<\|[^<>\n]+\|>")
_TERMINAL_SPECIAL_SUFFIX = re.compile(r"(?:\s*(?:<\|im_end\|>|<\|endoftext\|>|<\|eot_id\|>))+\s*\Z")


def _is_step_reference(line: str, match: re.Match, seen_numbers: set[int]) -> bool:
    """Recognize a narrow lexical backward reference, not a new boundary.

    This does not prove the mathematical relevance of a reference. Unknown
    targets, unintroduced mentions, and fused headings still fail closed.
    No characters, steps, or token boundaries are inserted or removed.
    """
    raw_number = match.group("number")
    return (
        len(raw_number) <= 128
        and int(raw_number) in seen_numbers
        and _STEP_REFERENCE_PREFIX.search(line[:match.start()]) is not None
    )


def _inspect_inline_markers(line: str, offset: int, start: int, seen_numbers: set[int], issues: list[Issue]) -> None:
    for inline in _INLINE_STEP.finditer(line, start):
        if _is_step_reference(line, inline, seen_numbers):
            issues.append(Issue("step_reference", "info", "A lexical reference to an already declared step is retained inside its original text span.", offset + inline.start(), offset + inline.end()))
        else:
            issues.append(Issue("inline_step_marker", "error", "Each step marker must begin its own line.", offset + inline.start(), offset + inline.end()))


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _escaped(text: str, position: int) -> bool:
    slashes = 0
    position -= 1
    while position >= 0 and text[position] == "\\":
        slashes += 1
        position -= 1
    return slashes % 2 == 1


def parse_response(
    text: str,
    *,
    finish_reason: Optional[str] = None,
    min_step_chars: Optional[int] = None,
    max_steps: Optional[int] = None,
    max_step_chars: Optional[int] = None,
) -> ParseResult:
    """Validate a complete response and preserve every recognized step.

    Canonical: sequential line-start 'Step 1:' markers, at least one nonempty
    step, then 'Final Answer: \\boxed{...}' and whitespace only. Multiline body
    and nested/multiline boxed content are supported. Fenced code is retained
    inside a step and its apparent markers are ignored, with recoverable status.
    Unclosed fences, numbering errors, missing final boxes, prose outside the
    contract, or known generation truncation are invalid. Optional size limits
    invalidate the result; they never delete/truncate diagnostic steps.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a str")
    result = ParseResult(source=text, status="invalid")
    issues = result.issues
    # Terminal transport tokens are independent of answer-format validity.
    # Keep source unchanged and use its original coordinates up to this suffix.
    terminal = _TERMINAL_SPECIAL_SUFFIX.search(text)
    text_end = terminal.start() if terminal else len(text)
    if terminal:
        issues.append(Issue("trailing_special_token", "info", "Recognized terminal end-token text is excluded from reasoning and final-answer spans.", terminal.start(), terminal.end()))
    for special in _SPECIAL_TOKEN.finditer(text[:text_end]):
        issues.append(Issue("interior_special_token", "error", "Special token text occurs before the terminal response boundary.", special.start(), special.end()))
    for control in re.finditer(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", text):
        issues.append(Issue("control_character", "error", "Unexpected control character in generated text.", control.start(), control.end()))
    for separator in re.finditer(r"\r(?!\n)|[\x85\u2028\u2029]", text):
        issues.append(Issue("noncanonical_line_separator", "warning", "Use newline (LF or CRLF) to separate lines.", separator.start(), separator.end()))
    markers: list[tuple[str, Optional[int], int, int]] = []
    fence: Optional[tuple[str, int, int]] = None
    seen_step_numbers: set[int] = set()
    offset = 0
    for line in text[:text_end].splitlines(keepends=True):
        body = line.rstrip("\r\n")
        fm = _FENCE.match(body)
        if fence:
            if fm and fm.group(1)[0] == fence[0] and len(fm.group(1)) >= fence[1] and not fm.group(2).strip():
                fence = None
            offset += len(line)
            continue
        if fm:
            fence = (fm.group(1)[0], len(fm.group(1)), offset)
            issues.append(Issue("code_fence", "warning", "Fenced code retained; markers inside it are not step boundaries.", offset, offset + len(body)))
            offset += len(line)
            continue
        m = _MARKER.match(body)
        if m:
            raw_number = m.group("number")
            if raw_number is not None and len(raw_number) > 128:
                # A 129-digit step number cannot represent sequential numbering
                # in a realizable response. Reject without unbounded int parsing.
                number = -1
                issues.append(Issue("step_number_overflow", "error", "Step number exceeds 128 digits; diagnostic number is -1.", offset, offset + m.end()))
            else:
                number = int(raw_number) if raw_number is not None else None
            kind = "step" if raw_number is not None else "final"
            markers.append((kind, number, offset, offset + m.end()))
            if kind == "step" and number > 0:
                seen_step_numbers.add(number)
            canonical_marker = f"Step {number}:" if kind == "step" else "Final Answer:"
            if body[:m.end()] != canonical_marker:
                issues.append(Issue("noncanonical_marker", "warning", "Use the exact unindented, unbolded marker: " + canonical_marker, offset, offset + m.end()))
            if bool(m.group("bold")) != bool(m.group("bold_end")):
                issues.append(Issue("unbalanced_marker_bold", "error", "Bold marker delimiters do not match.", offset, offset + m.end()))
            if raw_number is not None and (number < 1 or raw_number != str(number)):
                issues.append(Issue("invalid_step_number", "error", "Step numbers must be positive ASCII integers without leading zeroes.", offset, offset + m.end()))
            if body[m.end():] and not body[m.end():].startswith(" "):
                issues.append(Issue("marker_spacing", "warning", "Put one space after the marker or begin content on the following line.", offset + m.end(), offset + len(body)))
            _inspect_inline_markers(body, offset, m.end(), seen_step_numbers, issues)
        elif _MARKER_LIKE.match(body):
            issues.append(Issue("malformed_marker", "error", "A line resembles a step/final marker but does not follow the contract.", offset, offset + len(body)))
        else:
            _inspect_inline_markers(body, offset, 0, seen_step_numbers, issues)
        offset += len(line)
    if fence:
        issues.append(Issue("unclosed_code_fence", "error", "The response ends inside a code fence.", fence[2], len(text)))
    if finish_reason in {"length", "max_tokens", "max_length", "max_new_tokens"}:
        issues.append(Issue("generation_truncated", "error", "The generation reached its response-length limit.", len(text), len(text)))
    if not markers:
        issues.append(Issue("missing_steps", "error", "No explicit step markers were found."))
        issues.append(Issue("missing_final_answer", "error", "A separate final-answer marker is required."))
        return result
    if text[:markers[0][2]].strip():
        issues.append(Issue("preamble", "error", "Text before the first step is not allowed.", 0, markers[0][2]))
    expected = 1
    final_markers = []
    seen_final = False
    for index, (kind, number, start, end) in enumerate(markers):
        segment_end = markers[index + 1][2] if index + 1 < len(markers) else text_end
        if kind == "final":
            final_markers.append((start, end))
            seen_final = True
            continue
        if seen_final:
            issues.append(Issue("step_after_final_answer", "error", "Reasoning cannot continue after the final answer.", start, end))
        if number != expected:
            issues.append(Issue("nonsequential_numbering", "error", f"Expected Step {expected}, found Step {number}.", start, end))
        expected += 1
        content_start, content_end = _trim_span(text, end, segment_end)
        step = Step(number, text[content_start:content_end], content_start, content_end, start, end)
        result.steps.append(step)
        if not step.text:
            issues.append(Issue("empty_step", "error", "Each step must have nonempty content.", content_start, content_end))
        if min_step_chars is not None and len(step.text) < min_step_chars:
            issues.append(Issue("configured_min_step_chars", "error", "Step is below the explicitly configured character minimum.", content_start, content_end))
        if max_step_chars is not None and len(step.text) > max_step_chars:
            issues.append(Issue("configured_max_step_chars", "error", "Step exceeds the explicitly configured character maximum.", content_start, content_end))
    if not result.steps:
        issues.append(Issue("missing_steps", "error", "At least one reasoning step is required."))
    if max_steps is not None and len(result.steps) > max_steps:
        issues.append(Issue("configured_max_steps", "error", "Response exceeds the explicitly configured step-count maximum."))
    if not final_markers:
        issues.append(Issue("missing_final_answer", "error", "A separate final-answer marker is required."))
    elif len(final_markers) > 1:
        issues.append(Issue("multiple_final_answers", "error", "Exactly one final-answer marker is required.", final_markers[1][0], len(text)))
    else:
        final_start, label_end = final_markers[0]
        box_start, _ = _trim_span(text, label_end, text_end)
        wrapper = ""
        if text.startswith("$$", box_start):
            wrapper, box_start = "$$", box_start + 2
        elif text.startswith("$", box_start):
            wrapper, box_start = "$", box_start + 1
        elif text.startswith(r"\[", box_start):
            wrapper, box_start = r"\]", box_start + 2
        elif text.startswith(r"\(", box_start):
            wrapper, box_start = r"\)", box_start + 2
        box_start, _ = _trim_span(text, box_start, text_end)
        if wrapper:
            issues.append(Issue("final_math_wrapper", "warning", "Omit the optional math wrapper around the final box.", label_end, box_start))
        if not text.startswith(r"\boxed{", box_start):
            issues.append(Issue("missing_final_box", "error", "Final Answer must be followed by \\boxed{...}.", box_start, len(text)))
        else:
            content_start = box_start + len(r"\boxed{")
            depth = 1
            cursor = content_start
            while cursor < text_end and depth:
                char = text[cursor]
                if char in "{}" and not _escaped(text, cursor):
                    depth += 1 if char == "{" else -1
                cursor += 1
            if depth:
                issues.append(Issue("unclosed_final_box", "error", "The final box has unmatched braces.", box_start, len(text)))
            else:
                content_end = cursor - 1
                answer = text[content_start:content_end]
                if not answer.strip():
                    issues.append(Issue("empty_final_box", "error", "The final box is empty.", content_start, content_end))
                suffix_start = cursor
                if wrapper:
                    while suffix_start < text_end and text[suffix_start].isspace():
                        suffix_start += 1
                    if text.startswith(wrapper, suffix_start):
                        suffix_start += len(wrapper)
                    else:
                        issues.append(Issue("unclosed_final_math_wrapper", "error", "The final math wrapper has no matching close.", suffix_start, len(text)))
                suffix = text[suffix_start:text_end]
                if suffix.strip():
                    issues.append(Issue("text_after_final_answer", "error", "Only whitespace may follow the final boxed answer.", suffix_start, text_end))
                result.final_answer = FinalAnswer(answer, final_start, cursor, box_start, cursor, content_start, content_end)
    # The prompt reserves \boxed exclusively for one final answer. Nested
    # mathematical braces are fine; an additional boxing command is not.
    for box_command in _BOX_COMMAND.finditer(text[:text_end]):
        if _escaped(text, box_command.start()):
            continue
        final = result.final_answer
        if final is not None and box_command.start() == final.box_start:
            continue
        if final is not None and final.box_start < box_command.start() < final.box_end:
            issues.append(Issue("nested_final_box", "error", "Use one final boxing command; nested mathematical braces are allowed, nested \\boxed commands are not.", box_command.start(), box_command.end()))
        elif final is not None or not final_markers or box_command.start() < final_markers[0][0] or any(step.start <= box_command.start() < step.end for step in result.steps):
            issues.append(Issue("boxed_outside_final_answer", "error", "The \\boxed command is reserved for the separate final answer and must not occur in reasoning or other output.", box_command.start(), box_command.end()))
    result.status = "invalid" if any(i.severity == "error" for i in issues) else "recoverable" if any(i.severity == "warning" for i in issues) else "canonical"
    return result


def parse_reasoning_steps(
    response_str: str,
    min_step_chars: Optional[int] = None,
    max_steps: Optional[int] = None,
    require_math_content: bool = False,
    max_step_chars: Optional[int] = None,
    *,
    allow_recoverable: bool = False,
) -> list[str]:
    """Legacy-shaped API; [] on invalid/noncanonical output, never a fallback.

    Defaults deliberately remove the old silent 10-step/1000-character caps.
    Set limits explicitly if desired; an exceeded limit rejects the entire
    response. Use parse_response() for diagnostic steps and failure details.
    A lexical math filter cannot establish meaningful reasoning and is removed.
    """
    if require_math_content:
        raise ValueError("require_math_content=True is unsupported: lexical patterns cannot validate reasoning; use a separate semantic check")
    parsed = parse_response(response_str, min_step_chars=min_step_chars, max_steps=max_steps, max_step_chars=max_step_chars)
    return [s.text for s in parsed.steps] if parsed.canonical or (allow_recoverable and parsed.valid) else []


@dataclass(frozen=True)
class TokenStepSpan:
    number: int
    token_start: Optional[int]
    token_end: Optional[int]
    overlap_token_indices: list[int]
    excluded_boundary_tokens: list[int]
    char_start: int
    char_end: int


@dataclass
class TokenMappingResult:
    text: str
    method: str
    verified: bool
    training_safe: bool
    spans: list[TokenStepSpan] = field(default_factory=list)
    token_byte_offsets: list[tuple[int, int]] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _byte_decoder() -> dict[str, int]:
    # GPT-2/Qwen ByteLevel's reversible byte-to-Unicode alphabet.
    byte_values = list(range(ord("!"), ord("~") + 1)) + list(range(161, 173)) + list(range(174, 256))
    codepoints = byte_values[:]
    extra = 0
    for value in range(256):
        if value not in byte_values:
            byte_values.append(value)
            codepoints.append(256 + extra)
            extra += 1
    return {chr(codepoint): value for value, codepoint in zip(byte_values, codepoints)}


def map_original_token_spans(
    tokenizer: Any,
    generated_ids: Sequence[int],
    parsed: Optional[ParseResult] = None,
    *,
    skip_special_tokens: bool = True,
    allow_recoverable: bool = False,
) -> TokenMappingResult:
    """Map original Qwen/GPT-style ByteLevel IDs, with exact decode validation.

    No encode()/retokenization call is made. Every vocabulary token is inverted
    to its original bytes; the concatenation MUST equal decode(original_ids)
    byte-for-byte. Non-ByteLevel tokenizers fail closed. Special IDs keep their
    original indices even when skipped. Unicode characters split across IDs
    work because all comparisons happen in UTF-8 byte coordinates.

    A body span may retain a crossing token only when all its exterior bytes
    are whitespace. Otherwise that token is excluded and training_safe=False;
    it is impossible to isolate part of a token for token-level credit. The
    final answer and marker text are never knowingly included. All overlap
    indices remain available for audit. Do not consume spans for training
    unless both verified and training_safe are True.
    """
    ids = [int(x) for x in generated_ids]
    text = tokenizer.decode(ids, skip_special_tokens=skip_special_tokens, clean_up_tokenization_spaces=False)
    result = TokenMappingResult(text, "validated_original_id_bytelevel", False, False)
    if parsed is None:
        parsed = parse_response(text)
    if parsed.source != text:
        result.issues.append(Issue("source_decode_mismatch", "error", "Parser input does not exactly equal the decoded original generated IDs."))
        return result
    if not (parsed.canonical or (allow_recoverable and parsed.valid)):
        result.issues.append(Issue("invalid_parse_for_mapping", "error", "Mapping requires a canonical parse (or explicit acceptance of recoverable format)."))
        return result
    try:
        vocab_tokens = tokenizer.convert_ids_to_tokens(ids)
        if len(vocab_tokens) != len(ids):
            raise ValueError("convert_ids_to_tokens changed the token count")
        special_ids = set(getattr(tokenizer, "all_special_ids", []))
        added_vocab = tokenizer.get_added_vocab() if hasattr(tokenizer, "get_added_vocab") else {}
        added_ids = set(added_vocab.values())
        decoder = _byte_decoder()
        chunks = []
        cursor = 0
        for token_id, token in zip(ids, vocab_tokens):
            if token_id in special_ids:
                chunk = b"" if skip_special_tokens else token.encode("utf-8")
            elif token_id in added_ids:
                chunk = token.encode("utf-8")
            else:
                chunk = bytes(decoder[char] for char in token)
            chunks.append(chunk)
            result.token_byte_offsets.append((cursor, cursor + len(chunk)))
            cursor += len(chunk)
        source_bytes = b"".join(chunks)
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        result.issues.append(Issue("unsupported_token_bytes", "error", "Cannot recover original ByteLevel token bytes: " + str(error)))
        return result
    if source_bytes != text.encode("utf-8"):
        result.issues.append(Issue("token_bytes_decode_mismatch", "error", "Recovered original bytes differ from decoded text; no approximate or re-tokenized mapping is permitted."))
        return result
    result.verified = True
    last_non_special = max((i for i, token_id in enumerate(ids) if token_id not in special_ids), default=-1)
    if any(token_id in special_ids for token_id in ids[:last_non_special]):
        result.issues.append(Issue("interior_special_token_id", "error", "Original generated IDs contain a special token before the response ends; skipping it cannot establish a safe training span."))
    char_to_byte = [0]
    for char in text:
        char_to_byte.append(char_to_byte[-1] + len(char.encode("utf-8")))
    for step in parsed.steps:
        start, end = char_to_byte[step.start], char_to_byte[step.end]
        overlap = [i for i, (left, right) in enumerate(result.token_byte_offsets) if left < end and right > start and left != right]
        accepted = []
        excluded = []
        for i in overlap:
            left, right = result.token_byte_offsets[i]
            exterior = source_bytes[left:min(start, right)] if left < start else b""
            exterior += source_bytes[max(end, left):right] if right > end else b""
            # Only ASCII whitespace can occur outside a canonical marker/body
            # boundary. Reject unknown/multibyte exterior bytes conservatively.
            if exterior.strip(b" \t\r\n\v\f"):
                excluded.append(i)
            else:
                accepted.append(i)
        if excluded:
            result.issues.append(Issue("boundary_content_loss", "error", f"Step {step.number} shares {len(excluded)} token(s) with non-whitespace outside its content; excluded token indices {excluded}.", step.start, step.end))
        if not accepted:
            result.issues.append(Issue("empty_token_span", "error", f"Step {step.number} has no safely assignable original tokens.", step.start, step.end))
        if accepted and accepted != list(range(accepted[0], accepted[-1] + 1)):
            result.issues.append(Issue("noncontiguous_token_span", "error", f"Step {step.number} cannot be represented by one contiguous token span.", step.start, step.end))
        result.spans.append(TokenStepSpan(step.number, accepted[0] if accepted else None, accepted[-1] if accepted else None, overlap, excluded, step.start, step.end))
    result.training_safe = not any(i.severity == "error" for i in result.issues)
    return result
