# Extensive CPU audit — 2026-09-14

> Historical evidence: this report predates the full/LoRA selector. Its test
> counts and runtime observations apply to the earlier LoRA-only source. The
> new default full-training path and changed launchers have not been run or
> validated. No new validation was run for the mode-switch change.

This audit covers the **current custom PyTorch DDP + PEFT + vLLM handoff**.
It does not validate the older VERL integration. The four public settings are
local-positive STRIDE, single-step STRIDE, uniform STRIDE and GRPO, all using
rank-16 LoRA, no preparatory warmup, no structural reward, and constant alpha.
GPU execution was explicitly skipped at the user's request.

## Results and coverage

**284 tests passed, with no skips**, in 69.671 seconds under coverage instrumentation,
including the shared-seed update, four structural-reward regressions and the
clean-source-prompt regression. See the [exact prompt audit](PROMPT_FORMAT.md).
The separate [real-data validation](DATA_REWARD_TEST.md) created and checked
2,816 DAPO rows using both pinned Qwen tokenizers.

| Measure | Result |
|---|---:|
| Explicit source functions entered | 200 / 200 |
| Executable statements covered | 3,485 / 3,668 (95.0%) |
| Branches covered | 1,343 / 1,512 (88.8%) |
| Combined statement/branch coverage | 93.2% |

The uncovered 183 statements and 169 branches are listed in the inventory.
They include failure paths, logging options, legacy grader cases and controller
branches; they must not be described as tested. Source coverage excludes tests
and third-party libraries. No coverage exclusions were added to raise the score.

The machine-readable [function inventory](function_coverage.json) lists every
explicit Python function/method/nested function in `run.py`, `training/` and
`scripts/`, its source hash, executed statement count, missing lines and branches.
The inventory is checked against the actual Python AST, so unimported source
cannot disappear from the denominator. Generated dataclass methods, anonymous
lambdas and third-party library internals are not separate inventory entries.

**Function entry coverage is not proof of correctness.** Some functions cross
GPU, network or scheduler boundaries and are exercised with controlled
substitutes. The table below distinguishes actual computation from those mocks.

| Area | What actually ran | Limits |
|---|---|---|
| STRIDE maths | Independent float64 Schur-complement and log-determinant identities, production Cholesky comparisons, collinearity, zero vectors, masks, group gates, positive support and uniform-mass properties | Synthetic embeddings; no semantic quality claim |
| Actor and loss | Real tiny random Qwen3, PEFT adapters, selected-token log probabilities, clipped-loss finite differences, backward, AdamW, adapter save/reload | Tiny CPU model; not pretrained Qwen benchmark accuracy |
| Distributed update | Real Gloo processes, global-token loss/gradient comparison with serial computation, four-rank updates for all four arms, fixed-reference checks | CPU backend; no CUDA/NCCL |
| Full controller | Actual distributed smoke and resumable training entry points with tiny Qwen3, input parsing, credit, logging and checkpoints | CUDA setup redirected; sampler supplies deterministic synthetic text with log probabilities computed by a real current-actor copy |
| Resume | Real checkpoint/optimizer/RNG save and reload; interrupted versus continuous Qwen3 training; DDP bucket-order control | GPU RNG calls mocked; no bitwise GPU-resume claim |
| Parser and graders | Real pinned Qwen tokenizers, Unicode/token-boundary checks, malformed inputs, answer ambiguity, truncation aliases, 1,000 seeded malformed-text mutations | Valid formatting and final answers do not certify reasoning |
| Data and launchers | Real pinned DAPO preparation; offline tiny-parquet failure fixtures; source freezing/hashing; shell launchers with a recording command | Downloads use mocks in offline unit tests; no Slurm submission |
| Sampler | Real filesystem IPC and short-lived process groups, failure paths and descendant cleanup | vLLM generation, graph compilation and GPU allocation mocked |
| Shared seeds | Non-default seed43 through all four Slurm arms and array jobs; identical real tiny-Qwen LoRA initialization; matched question order and per-prompt seeds after rank partitioning; actual subprocess hash seed; evaluation 63-bit sampling/32-bit engine forwarding and CLI overrides | CPU and controlled sampler substitutes; no bitwise GPU claim |
| Evaluation | Actual CLI, input/model/checkpoint checks, overlap rejection, requests, scoring, metrics and immutable completion records | Tokenizer loading and sampler launch replaced in integrated fixture; no benchmark evaluation |
| Release | Source inventory, ZIP contents/hashes/CRC, atomic publication, corruption and race tests | No runtime/model/data artifacts shipped |

Math stress tests include 256 redundant steps and 1,000 duplicate-direction
cases. The float64 Schur/logdet comparisons meet 1e-9 tolerance. A tested FP32
versus float64 positive-credit stress case had maximum absolute error 8.82e-6
at alpha up to 3; the test tolerance scales as 1e-5 per unit alpha. These are
numerical checks over the tested matrices, not a universal error bound.

## Bugs fixed

- vLLM engine initialization and Python interpreter hashing previously relied
  on defaults. Public launchers now seed them explicitly from the experiment
  seed. Existing learner RNG, shuffle and per-question seed controls were already
  shared across arms. Use the documented `run.py` or Slurm commands; direct
  internal-driver invocation bypasses the launcher hash setup. Training contracts
  are now version4 and reject older resumes rather than mixing protocols.
- The new evaluation seed override initially exposed Bash3.2's empty-array
  behavior under `set -u`; both `--seed 43` and `--seed=43` now pass shell tests.
- Span generators were consumed during validation before reference pooling.
  Spans are now materialized once; malformed bounds and nonfinite/nonfloating
  hidden-state inputs are rejected.
- The parser and aggregate metric fallback did not consistently treat
  `max_new_tokens` as truncation. They now agree with active scoring.
- Boolean/float values could pass integer identity checks in sampler/checkpoint
  paths. IDs, step/rank counts and context declarations now require integers.
- Live sampler requests and checkpoint JSON could overwrite a competing writer.
  Exclusive atomic publication prevents replacement.
- Sampler shutdown/startup failure could leave worker descendants alive.
  Owned process groups are now terminated and reaped, with real process tests.
- Frozen-source construction could silently omit a symlinked Python module.
  It now rejects incomplete/symlinked source and verifies the content manifest.
- Release creation could leave a damaged final archive after an interrupted
  write. ZIPs are staged, validated and published atomically without overwrite.
- Preserved adapter-retention helpers matched only three-digit checkpoint
  numbers. They now handle four-digit and longer IDs and ignore unrelated names.

Shared actor helpers also infer the model device instead of hard-coding CUDA,
so actual Qwen3/PEFT computation can be tested on CPU. Production loading still
defaults to CUDA. No STRIDE kernel, credit rule or policy-loss formula changed.

## Resume roundoff investigation

The original strict bitwise assertion caught a real numerical difference after
restart. Saved adapters, optimizer tensors, all four ranks' RNG states, and the
resumed update's tokens/log probabilities/credit were exactly equal before that
update. Continuous DDP had rebuilt its gradient buckets; fresh resumed DDP had
not. The resulting collective reductions used different tensor layouts.

| Compared tensor | Maximum absolute difference with normal bucket rebuilding |
|---|---:|
| Adapter weights | 1.4901161193847656e-8 |
| Adam first moment | 1.1641532182693481e-10 |
| Adam second moment | 2.842170943040401e-14 |
| Adam step counter | 0 |

A separate **test-only** DDP setting that prevents bucket rebuilding produces
bit-for-bit equal final adapters and every optimizer tensor. Production DDP
settings were not changed. The normal-setting test requires exact checkpoint,
RNG and update-input identity and bounds the final FP32 error by
`8 * float32_epsilon * max_absolute_tensor_value`, with zero relative tolerance
and no absolute tolerance floor. The largest observed error was 2.36 such
scaled epsilons. This is evidence of bounded roundoff in this test, not a promise
that long resumed distributed runs will remain bit-identical.

## Remaining limits

The audit explicitly confirms a scientific vulnerability: gibberish with valid
step markers and an accepted final answer can still be eligible for diversity
credit. No semantic reasoning checker was added. Local-positive credit removes
negative auxiliary advantages, but does not prove higher Pass@1, Pass@K or
verified reasoning diversity than the uniform or single-step variants.

CUDA kernels, NCCL collectives, vLLM serving/graphs, full 8K GPU memory fit,
real pretrained model training and Slurm scheduling were not executed here.
Paper-scale convergence, multiple training seeds, external benchmark integration
and semantic diversity judging remain unmeasured. The suite includes dormant
reference helpers for provenance; their coverage does not enable removed
warmup/format training options.

The source evaluated in this report was LoRA-specific. The later full-training
implementation adds full-weight checkpoints, sampler refresh and a separate
frozen reference, but it is outside this report's executed coverage.

## Reproduce locally

Use a fresh environment with `requirements-cpu.txt`, then:

```bash
python scripts/run_tests.py
```

No pretrained weights are downloaded. The suite generates tiny random Qwen3
models and deletes its temporary training artifacts afterwards. To include the
optional real-tokenizer check, provide a prepared local tokenizer:

```bash
STRIDE_TEST_TOKENIZER_PATH=/absolute/path/to/qwen/tokenizer python scripts/run_tests.py
```

Both pinned tokenizer revisions were exercised during this audit: Qwen3-4B-
Instruct-2507 at `cdbee75f17c01a7cc42f958dc650907174af0554` and Qwen3-1.7B at
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`. The final complete suite used the 4B
tokenizer and skipped no tests. Without that environment variable, the optional
tokenizer test is reported as skipped.

For coverage, install `coverage==7.16.1` and use this configuration in a temporary
INI file from the extracted package's parent directory:

```ini
[run]
branch = True
parallel = True
concurrency = multiprocessing
source = stride_handoff
omit = */tests/*
```

```bash
COVERAGE_RCFILE=/absolute/path/to/audit.ini python -m coverage run stride_handoff/scripts/run_tests.py
python -m coverage combine --rcfile=/absolute/path/to/audit.ini
python -m coverage report --rcfile=/absolute/path/to/audit.ini
```

The tokenizer environment variable must also be set for the optional test.
Audit runtime: Python 3.13.3, torch 2.10.0, Transformers 4.57.6, PEFT 0.18.1,
NumPy 2.2.6, PyArrow 21.0.0, huggingface-hub 0.36.0 and coverage 7.16.1 on macOS
ARM. The documented NVIDIA deployment uses a separate Python 3.11 Linux lock.
