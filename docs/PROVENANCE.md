# Implementation provenance

This handoff derives from the local `qwen4b_scaleup_20260911` source snapshot,
which was used for the last Qwen4B custom-training jobs before CSCC access was
lost. It is not a clean upstream VERL checkout or a resumed cluster experiment.
`original_source_hashes.json` records the source identities before portability
edits. The original source archives were left unchanged.

The pinned STRIDE kernel and original five-rule reference module are byte-identical:

- `production_stride.py`: `a547f1b13423011c68b061d3ea93dffbbe97a6fe896ce709c2310a478a22ab0b`
- `credit.py`: `51d82169de7731b7e866699f58ead99c59519992a1e68e5dbbcc5c669c49bf52`

Raw novelty U is bounded in [0,1]. `local_positive` uses alpha*U. `token_uniform`
preserves each response's positive coefficient mass over its mapped step tokens.
`centered` subtracts the group's mapped-token mean and can therefore give correct
responses negative auxiliary advantage. That historical variant is retained
only in the reference module; it is not selectable in the four-setting runner.
The current positive variant does not use a softmax over log deltas.

`experiment_credit.py` exposes local_positive, single_step, token_uniform and
grpo. It reuses the unchanged original local/uniform/GRPO outputs and adds a
single pooled step per eligible response. `feature_pooling.py` preserves the
original mean-and-normalize whole/step calculations and additionally pools the
raw hidden states of the same mapped tokens for single_step. Both new modules
are part of the recorded source/resume identity. This is a new ablation, not
an already executed CSCC experiment.

Key preserved routines include token log probabilities, the clipped policy loss,
the prompt format and global-real-token loss normalization. Actor loading,
reference extraction and sampler/checkpoint handling now branch on training mode. Portability changes
make paths, K, alpha, learning rate, seed and schedule explicit; alpha is forwarded
to the same credit function. The current comparison removes preparatory warmup
and initializes each method directly from the downloaded model. Full training
is now the default; optional LoRA initializes fresh adapters.
Structural reward is disabled, alpha stays fixed, and training starts with the
seed's first data-shuffle epoch (offset0). Checkpoint
contracts now include actual model/tokenizer file content rather than trusting
a label that points to a machine-specific path. The evaluator no longer assumes
exactly 128 questions. No benchmark-specific score improvements are claimed.

Warmup stages, fresh-run input adapters and format-reward options are absent
from the public training CLI. Evaluation accepts a full checkpoint or trained adapter, and training can
resume its own complete optimizer checkpoints. The updated resume
contract records direct model initialization; incompatible older contracts
cannot silently resume. Original source archives remain available unchanged.

The earlier version-4 training contract pinned the vLLM engine seed and Python
hash seed to the experiment seed. Python, NumPy, PyTorch and fresh LoRA
initialization use that same seed, and data order uses `seed + epoch`. Rollout
seeds are `20260980 + seed * 1000000 + step * batch_size + global_prompt_index`
with no method/job offset or rank offset beyond the global question position.
Evaluation uses `seed + input_position` for sampling and the seed's low 32 bits
for the engine and Python hash seed. Preparation seed42 fixes data selection
separately from repeated training seeds such as 42 and 43. These explicit seeds
support matched comparisons; they do not guarantee bitwise GPU determinism or
identical rollouts once policies diverge. Earlier optimizer contracts are
intentionally incompatible with version4. The subsequent version-5 contract
also records full/LoRA mode, learner distribution, reference strategy and learning
rate; it rejects earlier contracts and cross-mode optimizer resumes.

## Extensive CPU audit changes (2026-09-14)

The original 18 files still match the recorded snapshot hashes. Ten remain
byte-identical in this handoff: `production_stride.py`, `credit.py`,
`format_credit.py`, `scoring.py`, `native_declared_steps.py`,
`native_contract.py`, and the four `pilot_grading` files. The parser, sampler
and checkpoint code now include the audited fixes below; they must no longer
be described as byte-identical to the cluster snapshot.

- Materialize iterable step spans once before validation/pooling; validate
  integer span endpoints and finite floating reference states.
- Recognize `max_new_tokens` as truncation consistently in parser and metrics.
- Publish sampler requests and checkpoint JSON exclusively, preventing a
  competing writer from replacing an immutable record. Reject boolean/float
  values where an actual integer ID or topology count is required.
- Launch each sampler in its own process group and reap owned descendants on
  failure or shutdown. Reject invalid GPU selections before creating IPC files.
- Validate frozen Python source manifests and reject symlinked/incomplete
  snapshots. Stage, hash-check and atomically publish source releases.
- Infer shared actor-helper tensor devices from the actor, enabling actual CPU
  Qwen3/PEFT integration tests; production actor loading still defaults to CUDA.
- Allow four-digit and longer checkpoint numbers in the preserved
  adapter-pruning routines. Active resumable training already uses the separate
  complete-pair pruning implementation.

These audited changes preserved the loss and STRIDE credit definitions in the
then LoRA-only package. The subsequent full/LoRA implementation is outside this
audit's executed coverage. See [the audit report](EXTENSIVE_TEST_REPORT.md) for actual
execution, mocked boundaries and the DDP resume roundoff control.

## Full/LoRA mode addition (not executed)

The public launcher now defaults to full-parameter FSDP training on four learner
GPUs, with four separate vLLM samplers. `--finetune-mode lora` selects the prior
rank-16 adapter approach using DDP. Full mode maintains FP32 master weights with
BF16 computation and a separate frozen reference; LoRA uses its frozen base as
the reference with adapters disabled. Full checkpoints are `model_NNN`, LoRA
checkpoints are `adapter_NNN`, and both have complete optimizer `state_NNN`
directories. Full sampler refresh reloads complete model weights.

Default learning rates are `1e-6` for full and `1e-4` for LoRA, explicitly
recorded as experimental settings, not tuned conclusions. No CPU tests,
validation, GPU jobs, throughput measurements or convergence evaluation were
run for this addition. Earlier audit numbers do not certify this new path.
Automatic Hugging Face upload was absent from that initial mode addition.
It is now integrated through the launcher and the separate CPU exporter/publisher;
see [HF_UPLOAD.md](../HF_UPLOAD.md) and the current [verification report](../VERIFICATION.md).

## Runtime

The original tested cluster environment recorded PyTorch 2.10.0+cu128,
Transformers 4.57.6, PEFT 0.18.1 and the pre-release build
`vllm 0.17.0rc1.dev168+gdc6b57846.precompiled`.

This package pins public release vLLM 0.17.0 as a **candidate** portable runtime.
Its published requirements match torch2.10.0 and transformers>=4.56,<5. The
required compilation APIs are present in the release source. Dependency
resolution/API inspection do not replace a real GPU smoke test.

Primary references:

- https://docs.vllm.ai/en/v0.17.0/getting_started/installation/gpu/
- https://github.com/vllm-project/vllm/blob/v0.17.0/requirements/cuda.txt
- https://github.com/vllm-project/vllm/blob/v0.17.0/vllm/config/compilation.py
- https://pypi.org/pypi/vllm/0.17.0/json

## Research limits

Final-answer correctness and parser validity do not prove intermediate steps are
mathematically correct. This code does not solve the gibberish/embedding reward
hacking problem by adding semantic verification. Nonnegative auxiliary credit
removes one suppressive mechanism but does not guarantee improved Pass@K.

The historical DQO-style reference calculation uses shared base-model embeddings
and binary answer rewards; it is not a reproduction of the published setup and
is not exposed by this comparison runner. Four external evaluation datasets,
additional baselines and a validated semantic diversity judge remain separate
work.
