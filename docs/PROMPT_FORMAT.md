# Prompt and parser audit — 2026-09-14

This historical audit checked the shared output-format instruction in the
earlier LoRA-only trainer. A later change added default full-parameter training
and optional LoRA; that new implementation has not been run or validated. This
report does not extend the historical results to the new path. Hugging Face
upload integration remains separate and incomplete.

## Which upstream field reaches the model?

The pinned English DAPO parquet has two prompt fields. In all 14,116 source rows,
`source_prompt` wraps the question in a generic “step by step” instruction and
requests `Answer: $Answer`, with a reminder after the question. The separate
`prompt` field contains exactly the question without that wrapper.

`prepare_data.source_records()` uses `prompt` to create `question`. It does not
use `source_prompt`. All 2,816 prepared training/validation/test rows are free of
that generic wrapper. The original question text remains unchanged, including
mathematical boxes in one question; these are problem content, not answer-format
instructions, and must not be stripped.

The actual input path is:

```text
upstream prompt -> row.question -> shared INSTRUCTION + row.question
-> model's native user/assistant chat template -> original prompt token IDs
```

The reference answer is used only for grading. Changing `ground_truth` does not
change the rendered prompt. The same render function serves all four methods,
both training drivers and evaluation, with no method-specific prompt branch.

## Exact updated shared instruction

```text
Solve this mathematics problem carefully.
In your final response, present your reasoning in at least two nonempty steps. Start each step on a new line with Step 1:, Step 2:, and so on, numbered consecutively without gaps. Each step should contain one meaningful mathematical deduction; do not add empty or trivial steps just to increase the count. Include these numbered steps in the final response even if you also use a separate thinking section.
After the steps, write exactly one final-answer line: Final Answer: \boxed{answer}. Replace answer with your result. Use \boxed{...} only for this final answer, not for intermediate values. Write nothing after the final-answer line.

Problem:
[Original question, unchanged]
```

The previous instruction requested consecutive labelled sections but did not
explicitly require new-line markers, at least two nonempty steps, or visible
final-response steps when a thinking section is present. This clarification
matches the existing parser; it does not change its reward or eligibility rules.
Structural reward remains zero, and no warmup was introduced.

## What was tested

Both pinned Qwen tokenizers rendered all 2,816 rows, for 5,632 all-row prompt
checks. Every prompt round-tripped to exactly the original string, and every
request passed the sampler protocol validator. The longest updated prompt is
1,194 tokens, leaving at least 6,998 response tokens under the 8,192 total cap.
The new instruction adds 76 prompt tokens with either tokenizer.

Sixteen constructed-response cases (eight per real tokenizer) produced:

| Response format | Correct final answer retained | Step bonus eligible |
|---|---|---|
| Two consecutive visible steps | Yes | Yes |
| Inline markers on one line | Yes | No |
| Skipped step number | Yes | No |
| Empty step body | Yes | No |
| Only one step | Yes | No |
| Prose after the final answer | No | No |
| Steps only inside a thinking section, answer outside | Yes | No |
| Visible numbered steps after a closed thinking section | Yes | Yes |

The two model chat templates were left native. Qwen3-1.7B may use a thinking
section, but the current parser maps only the final response's steps. The updated
instruction explicitly requests numbered steps there. It does not disable thinking.

These tests prove prompt wiring and parser behavior for the tested examples.
They do not prove that a model will always follow the instructions or that each
extracted step is mathematically valid or a single semantic deduction. Actual
rollout compliance still needs a model-generation test; none was run on GPU.

## Provenance

The prior prompt is preserved in the audit output and previous immutable ZIPs.
Prepared dataset bytes and STRIDE maths are unchanged. Since prompt text changes
policy inputs, new comparisons must use this same prompt for every arm; strict
resume source hashes reject attempts to mix the earlier prompt with this one.
