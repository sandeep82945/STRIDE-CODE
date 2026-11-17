# Automatic Hugging Face checkpoint upload

The launcher publishes the selected epoch checkpoints **after training finishes
successfully**. This supports both full-parameter models and LoRA adapters. Each
checkpoint gets its own model repository, with loadable files at the repository
root and a generated model card explaining the model, method and training setup.

## Configure once

Fill these entries in the root `.env` file:

```dotenv
HF_USERNAME=your-user-or-organization
HF_TOKEN=your-write-token
```

Keep the existing `WANDB_API_KEY` there as well. The Hugging Face token needs
permission to create and write model repositories in the selected namespace.
`HF_USERNAME` must be the token owner's username or an organization they belong
to. The pre-training identity check cannot guarantee every fine-grained token's
permission to create a future repository; repository creation checks that access.
Never put a token in the TOML, command line, model card, or Git repository.

The root [run_settings.toml](run_settings.toml) contains:

```toml
[huggingface]
enabled = true
epochs = [2, 3, 4]
repo_prefix = "stride-iclr2027"
description = ""
retries = 3
```

Edit the existing section. `description` is optional, user-visible model-card
text; keep it suitable for publication. Epochs beyond the configured training
schedule are omitted. The launcher rejects a schedule that reaches none of the
requested upload epochs. Choose `[1]` for a one-epoch pilot.

The normal commands remain:

```bash
python run.py train --dry-run
sbatch run.sbatch
```

Dry-run prints the resolved configuration without authenticating or publishing.
Actual training checks token identity first, without creating repositories. A
failed training process never starts automatic publication. The upload process
runs on CPU after the training processes exit; under Slurm the same allocation
remains reserved until upload finishes, so include upload time in the job limit.
Epoch validation jobs remain independent and can continue running.

To turn upload off for a particular run, use `--hf-upload off` or set
`enabled = false`. Explicit `--hf-upload`, `--hf-epochs`, `--hf-repo-prefix`,
`--hf-description` and `--hf-retries` override the TOML. `HF_UPLOAD_ENABLED` in
the environment or `.env` also overrides the file; CLI takes precedence.
Smoke and evaluation stages never publish. `--no-settings` opts out of automatic
upload unless explicitly enabled through CLI/environment.

## Naming and visibility

The default visibility depends on the method and applies to both Qwen models
and both training modes:

| Method | Repository visibility |
|---|---|
| `local_positive` | Private |
| `single_step` | Private |
| `token_uniform` | Private |
| `grpo` | Public |

For a private pilot, set `HF_PRIVATE_ALL=true` in the literal `.env` file or
publisher environment. This makes every method, including GRPO, private. The
override accepts `true/false`, `on/off`, or `1/0`; an invalid value is rejected
before publication. Omitting it preserves the default table above. An existing
repository with different visibility is rejected; the uploader never changes
repository privacy. The pilot profile uses this override by default.

Version-six training contracts support a smaller final batch with no dropped
or duplicated questions. With 150 questions and nominal batch 64, each epoch
has batches of 64, 64 and 22; the four epoch checkpoints are updates 3, 6, 9 and
12. Export verifies these completed epoch boundaries. Legacy version-five
contracts still require exactly divisible batches.

Names include the configured prefix, model, method, full/LoRA mode, seed, K,
alpha, epoch and a fingerprint identifying the run/checkpoint. The fingerprint
prevents unrelated attempts from overwriting one another. Model cards record
the full base model/revision, dataset hash and question count, epoch/update,
training budgets and description. No performance claims are generated.

Every selected epoch is an independent repository, rather than a hidden
subfolder. For one finetuning mode, two models, four methods and three selected epochs make
24 repositories: 18 private STRIDE repositories and six public GRPO repositories.
Exact repository IDs, commit URLs and per-checkpoint status appear in the local
`RUN/hf_upload_status.json`.

## Files uploaded

- Full mode: complete trained safetensors weights, model configuration, tokenizer
  and generation configuration when present.
- LoRA mode: adapter weights/configuration and tokenizer. The export's adapter
  configuration points to the pinned upstream base repository/revision.
- Both: generated `README.md`, `training_metadata.json` and `export_manifest.json`.

Optimizer/RNG states, raw questions, generated responses, logs, `.env`, source
code and credentials are excluded. These are inference exports. Exact optimizer
resume still needs the original local checkpoint/state pairs.

Exports are staged under `RUN/hf_exports/epoch-NNN/`. Original checkpoints are
not rewritten or deleted. Large safetensors files are hard-linked when possible;
copy fallback is used where links are unavailable. Small configuration copies
remove machine-local path fields. Hashes and completion metadata are checked
before publication. The original base-model directory and its preparation
manifest must remain available, including tokenizer files for LoRA.

## Failure and retry

Training completion stays in `RUN/done.json`. Upload completion is separate:
require `complete: true` in `RUN/hf_upload_status.json`. The launcher returns a
nonzero exit status when upload fails, while preserving the trained checkpoints.
Retry only publication from the package root:

```bash
python upload_checkpoints.py --run-dir runs/qwen4b/full/comparison_seed42/local_positive
```

Use the same namespace, prefix, epochs and description when retrying. If the run
used a custom TOML, add `--settings /path/to/that-file.toml`. The retry launcher
reads `.env`; it does not initialize a model, allocate GPUs or rerun training.
Its `--dry-run` checks configuration only; the actual publisher performs the
checkpoint and remote checks.

Verified identical repositories are recognized without uploading again.
Transient network/server failures use bounded retries. Existing repositories
with a different payload or visibility are rejected; the uploader never flips
privacy or overwrites unrelated files. Success requires a pinned remote commit,
a matching export manifest and verified remote file hashes/sizes.

The implementation uses the official [Hugging Face upload API](https://huggingface.co/docs/huggingface_hub/guides/upload),
with exact file allowlists and a parent-commit guard. Current verification uses
CPU tests and mocked Hub responses; no real repository was created during tests.
