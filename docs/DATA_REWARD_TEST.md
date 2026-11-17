# Structural reward and real dataset tests — 2026-09-14

> Historical evidence: this report predates the full/LoRA selector. Its test
> counts and runtime observations apply to the earlier LoRA-only source. The
> new default full-training path and changed launchers have not been run or
> validated. No new validation was run for the mode-switch change.

Both requested checks passed. No production code change was necessary.
The complete CPU suite passed **284 tests, no skips**, in 69.671 seconds under
coverage after the [prompt clarification](PROMPT_FORMAT.md). Four structural
regressions are in `tests/test_no_structural_reward.py`.
GPU execution and model-generated responses were not tested.

## Structural reward

| Public method | Structural/format reward | Warmup |
|---|---:|---:|
| `local_positive` | 0 | None |
| `single_step` | 0 | None |
| `token_uniform` | 0 | None |
| `grpo` | 0 | None |

Both active drivers pass `format_coefficient=0` to the shared annotation code.
Nonzero coefficients are rejected, and removed warmup/format CLI options cannot
enable them. The preserved legacy `format_credit.py` reward routine is not called
by active training; the regression test replaces it with a function that raises
if called.

The tests use the actual scorer and credit functions. They verify:

- Formatted wrong answers get no auxiliary credit. An all-wrong group has zero
  advantages and gradients; both update functions leave a fresh actor unchanged
  with a fresh AdamW optimizer. This does not assert that an optimizer carrying
  old momentum can never move after a zero-gradient batch.
- Correct answers without valid step formatting retain their correctness credit.
- Injecting format/structural diagnostic values of plus or minus one billion
  does not change any method's advantages, including an active diversity group.
- All four methods reject a nonzero `format_coefficient`.

STRIDE still requires valid step spans to assign its diversity bonus. That
eligibility rule is separate from a scalar structural reward. The smoke-only
forced update checks execution and is not a training reward or warmup stage.

## Actual dataset creation and loading

Executed from the package root:

```bash
python scripts/prepare_data.py --out data --train-count 2048 --val-count 256 --test-count 512 --seed 42
```

The pinned English DAPO source contains 14,116 rows. Removing all members of
normalized duplicate groups excludes 1,149 rows, leaving 12,967 candidates.
Sampling uses seed42 with no model-outcome or difficulty filtering.

| Split | Rows | SHA-256 |
|---|---:|---|
| Training | 2,048 | `d319351d80df073e947576189f41c23b549584ecca4bad660a2b752427f429cc` |
| Validation | 256 | `2d414c21faa8a1ebccf29f8a72f6feada213daec3457d8397160ff50340c2c8f` |
| Test | 512 | `6074c05fd6f2a85a763fe6d387ac815c45f7854e031c8572335a7891c9735327` |

Every row has exactly `id` (string), `question` (original string), and
`ground_truth` (integer). Questions are unchanged; answers agree with both
upstream answer fields. They load through the real training/evaluation loaders.
All four methods and both model labels load the identical training hash. At
batch64 and four epochs, every training question appears exactly four times
across 128 updates. Rank partitioning retains complete question groups.

All three pairwise split comparisons have zero overlapping IDs and zero
overlapping normalized questions. No within-split duplicates were found.
Recreating the splits with seed42 yields byte-identical JSON, also matching the
previous prepared data. These are DAPO-source holdouts, not external benchmark
test sets. No external benchmark exclusion files were supplied, so external
benchmark contamination has not been ruled out.

## Prompt formatting with real tokenizers

All 2,816 questions were rendered and protocol-validated with each pinned Qwen
tokenizer, for 5,632 prompt checks. Each rendered prompt round-trips exactly
through token IDs. Changing the reference answer leaves the prompt unchanged:
the renderer uses only the original question and shared instruction.

| Split | Minimum prompt tokens | Maximum prompt tokens | Minimum remaining response budget |
|---|---:|---:|---:|
| Training | 162 | 1,185 | 7,007 |
| Validation | 165 | 1,194 | 6,998 |
| Test | 164 | 1,110 | 7,082 |

These lengths matched for both tokenizers. No question exceeds the 8,192-token
**total prompt-plus-response** limit. Sampler requests carry K=8 and only prompt
IDs, question IDs and seeds; answers are not sent to the sampler.

Both native chat templates were checked against their pinned upstream files
(five files per model). The template adds the shared instruction to use
consecutive `Step N:` sections and a terminal `Final Answer: \boxed{answer}`.
These are prompt instructions, not structural reward terms. Training data need
no manually annotated step boundaries because generated responses are parsed.

An additional 144 synthetic parser cases used real tokenizer IDs: correct
formatted responses, incorrect-answer variants, and truncated variants. Correct
fixtures were mapped to two valid step spans; incorrect and truncated variants
received no STRIDE eligibility. These fixtures establish parser compatibility,
not how reliably a trained model will follow the requested format or reason.

The prepared data are kept separately from the source-only release. Reuse the
same `data/train.json` for all methods and training seeds.
