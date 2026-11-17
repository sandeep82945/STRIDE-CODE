"""Native presentation adapter with original spans and explicit recovery status.

Accepts uncredited introductory prose, declared Markdown steps and one terminal
answer region. This validates syntax, not the truth or originality of reasoning.
The unchanged strict parser result is retained independently by the caller.
"""
import re
from native_declared_steps import permissive_steps, FINAL_ANSWER, STEP_LIKE
from native_step_parser import ParseResult, Issue, _FENCE, _INLINE_STEP, _is_step_reference
from pilot_grading.relaxed_grader import extract_final_answer, _BOX
from pilot_grading.integer_reference import _escaped


def _presentation_only(text):
    return not re.sub(r'\\\[|\\\(|\$|\*\*|[\s:]', '', text)


def parse_native_final(text):
    """Validate complete final-channel syntax, without any ground truth."""
    result=ParseResult(text,'invalid',[],None,[])
    def fail(code):
        result.issues.append(Issue(code,'error','Native syntax contract failed: '+code))
        return result
    final=extract_final_answer(text,'integer')
    if not final['extraction_valid']:return fail('native_terminal_answer_invalid')
    boxes=[m for m in _BOX.finditer(text) if not _escaped(text,m.start())]
    if len(boxes)!=(1 if final['source']=='boxed' else 0):return fail('native_extra_box')
    headings=list(FINAL_ANSWER.finditer(text))
    if len(headings)>1:return fail('native_multiple_final_headings')
    candidate=final['final_answer_start']
    if headings:
        h=headings[0]
        if h.start()>candidate:return fail('native_final_heading_after_answer')
        if final['source']=='boxed' and not _presentation_only(text[h.end():candidate]):
            return fail('native_prose_between_final_heading_and_answer')
        cutoff=h.start()
    else:
        cutoff=text.rfind('\n',0,candidate)+1
        if not _presentation_only(text[cutoff:candidate]):return fail('native_answer_not_separate')
    steps=permissive_steps(text,0,cutoff)
    if len(steps)<2:return fail('native_invalid_declared_steps')
    # A fenced quoted example must not be reinterpreted as actual boundaries.
    # Ordinary fenced content inside an actual step is retained verbatim.
    fence=None
    for line in text.splitlines():
        fm=_FENCE.match(line)
        if fence:
            if fm and fm.group(1)[0]==fence[0] and len(fm.group(1))>=fence[1] and not fm.group(2).strip():fence=None
            elif STEP_LIKE.match(line) or FINAL_ANSWER.match(line):return fail('native_fenced_marker')
        elif fm:fence=(fm.group(1)[0],len(fm.group(1)))
    if fence:return fail('native_unclosed_fence')
    for step in steps:
        # Retain backward references, reject fused inline declarations.
        for line in step.text.splitlines():
            for m in _INLINE_STEP.finditer(line):
                if not _is_step_reference(line,m,set(range(1,step.number+1))):
                    return fail('native_inline_step_marker')
    if re.search(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]',text):return fail('native_control_character')
    result.steps=steps
    result.status='recoverable'
    result.issues.append(Issue('native_presentation_contract','warning','Native Markdown, introductory text and terminal-answer presentation accepted; all mapped spans remain original.'))
    if text[:steps[0].marker_start].strip():
        result.issues.append(Issue('native_uncredited_preamble','warning','Introductory content is excluded from step credit.',0,steps[0].marker_start))
    return result
