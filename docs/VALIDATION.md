# Validation of the portable handoff

## Final release with automatic Hugging Face upload — 2026-09-15

The current source passed **432 CPU tests with no failures, errors or skips**,
plus one additional cached Qwen3-1.7B tokenizer test (**433 total**).
See the root [final verification report](../VERIFICATION.md) for coverage and
limits. The smaller counts below record earlier checks, not the final total.

The upload checks cover read-only authentication preflight, automatic launch
sequencing, full/LoRA export loading on tiny random Qwen models, private/public
repository policies, remote payload hash checks, lost responses, and idempotent
retries. Hub calls use a fake service; no actual repositories were created or
uploaded. No GPU run was performed.

## Current W&B and automatic validation checks — 2026-09-14

The configuration update was checked with **117 CPU tests** covering `.env`
handling, launch arguments, training options, validation job construction,
evaluation entrypoints, metrics logging, observer startup/cleanup, and control-plane behavior. Slurm and
W&B calls were mocked. No GPU training, GPU validation job or external service
call was made. API-key handling checks use synthetic values only.

The actual prepared **2,048 training and 256 validation questions** were loaded,
hashed and checked for disjoint IDs and normalized question text. Validation
input pinning was exercised in a temporary directory with mocked `sbatch`
availability; no scheduler job was submitted. These checks establish input and
configuration behavior, not model performance or GPU memory fit.

The enabled defaults now submit held-out validation after each completed epoch
on a separate one-GPU Slurm allocation, with online W&B logging. Training does
not wait for validation completion. The complete full-training path, GPU
checkpoint reloads, cluster permissions, network connectivity and actual
inference results still require a live run.

## Historical LoRA-only audit

The remaining test counts and runtime observations below predate the
full/LoRA selector and apply to the earlier LoRA-only source. No complete
training run was performed for the new full mode; the current configuration
checks above do not establish full-training runtime correctness.

Checked locally on 2026-09-14. This covers source/data tests and real tiny-model CPU updates. GPU testing
was skipped at the user’s request; this is not an accuracy comparison.

## Executed checks

- All **284 CPU tests pass**, including a real two-process Gloo comparison of the
  distributed loss/gradient/update against serial computation; optimizer and
  per-rank RNG resume; STRIDE credit nonnegativity, support and mass; dataset
  separation; checkpoint identity checks; and portable command construction.
- New ablation checks compare single-step Cholesky credit with an independent
  Schur-complement calculation, test raw-token pooling with unequal step lengths
  and excluded gaps, and confirm bitwise preservation of local/uniform/GRPO
  outputs. Mocked reference features exercise all four annotation paths.
- The actual simple and array Slurm shell scripts were executed with a recording
  substitute for Python: each ARM selects the intended method and output path,
  with identical remaining training settings. No scheduler job was submitted.
- Non-default seed43 is verified across all four arms: unchanged frozen data,
  equal tiny-Qwen LoRA initialization, question order and per-prompt sampling
  seeds. Engine and interpreter hash seeds are explicit through the documented
  launchers. Evaluation seed inheritance, both CLI override forms and 63-bit
  sampling/32-bit initialization boundaries are covered.
- Four additional tests exercise absence of structural reward through the actual
  scorer, credit computation and both actor-update functions. See
  [structural reward and real-data tests](DATA_REWARD_TEST.md) for the newly
  created 2,048/256/512 dataset and all-row checks with both Qwen tokenizers.
- The [prompt audit](PROMPT_FORMAT.md) verifies that preparation uses the cleaned
  upstream `prompt`, excluding the conflicting `source_prompt` wrapper. The shared
  instruction now explicitly matches the existing parser's newline, numbering,
  nonempty-step, visible-final-response and one-terminal-answer requirements.
- Fresh-start tests cover all four arms with no input adapter or preparatory
  training, zero format reward, and constant alpha. Removed warmup/format options
  are rejected. Verified resume preserves the direct-start scientific contract
  and rejects older incompatible initialization and shuffle-offset settings.
- A synthetic case with seven correct answers out of eight demonstrates that
  the centered auxiliary term can reverse a correct token's total advantage at
  alpha=1. The positive variant does not reverse that token. This establishes a
  possible mechanism, not that it caused the previous experiment's accuracy.
- The original 18 source files still match the recorded pre-portability hashes.
  Ten files, including the STRIDE kernel, credit rules and scorer, remain
  byte-identical in this package. Parser, sampler and checkpoint fixes are
  documented in [provenance](PROVENANCE.md).
- Python source syntax, eight CLI help entry points, four launcher dry runs,
  frozen trainer/evaluator entry points from an unrelated directory, and all
  Slurm templates were checked locally.
- Public dependency metadata resolves to the supplied pinned lock for Python
  3.11 / Linux x86-64 / glibc 2.35. The vLLM 0.17.0 source contains the compilation,
  LoRA graph and safetensors loading options used by the sampler. These checks
  do not establish binary compatibility or successful model inference.

CPU checks used Python 3.13.3, torch 2.10.0, NumPy 2.2.6, PyArrow 21.0.0,
huggingface-hub 0.36.0, Transformers 4.57.6 and PEFT 0.18.1 on macOS ARM.
Real tiny random Qwen3/LoRA models completed four-rank Gloo updates for all four
arms, adapter reloads, and interrupted/resumed training. Fixed DDP buckets yield
bit-identical resume; normal bucket rebuilding produces bounded FP32 roundoff.
See the [extensive report](EXTENSIVE_TEST_REPORT.md) and
[per-function coverage](function_coverage.json). The recipient's documented GPU environment
uses Python 3.11 and must run its own CPU suite and GPU smoke checks.

## Real DAPO download and preparation

Executed the README preparation command with seed 42, train=2048, val=256 and
test=512 against the pinned upstream English parquet. It contained 14,116 rows;
after removing all rows belonging to duplicate normalized-question groups,
12,967 unique eligible rows remained. Selected splits have no shared IDs or
normalized questions. No external exclusion sets were supplied in this check.

| Split | Questions | SHA256 of emitted JSON |
|---|---:|---|
| Train | 2,048 | `d319351d80df073e947576189f41c23b549584ecca4bad660a2b752427f429cc` |
| Validation | 256 | `2d414c21faa8a1ebccf29f8a72f6feada213daec3457d8397160ff50340c2c8f` |
| Test | 512 | `6074c05fd6f2a85a763fe6d387ac815c45f7854e031c8572335a7891c9735327` |

These are new splits from the upstream training set. They are not four external
paper benchmarks, and this check does not establish semantic decontamination.
Dataset records are not bundled in the source release.

## Model acquisition

The Hugging Face API confirmed the two pinned model revisions and their weight,
configuration and tokenizer file inventories. Offline fixtures test interrupted
download retries, explicit resume and completed-download hash verification.
Full pretrained model weights were not downloaded or loaded in this local
validation. Both pinned Qwen tokenizers were downloaded and tested; the full
suite uses the 4B tokenizer. Tiny random Qwen3 weights were created locally.

## Required checks on the recipient's GPU node

Follow the README: install the locked environment, run `pip check` and the CPU
suite, run `check_runtime.py`, then the short and 8K smoke tests before the
comparison. No warmup is required. A successful primitive test is not a model smoke.
The direct-start/resume portability changes and public vLLM release have not undergone
end-to-end GPU training in this handoff environment.

No new Pass@1, Pass@8, convergence or semantic reasoning-diversity results are
claimed. The runner exposes four settings: local-positive STRIDE, single-step
STRIDE, uniform STRIDE and GRPO. The new single-step setting is CPU-validated,
not GPU-trained. Other planned baselines are not part of this four-arm study.
