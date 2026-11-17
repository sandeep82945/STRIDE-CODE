# Held-out validation and final evaluation

Training now submits this evaluator automatically after each completed epoch,
using a separate Slurm allocation with **one visible GPU**. Training continues
without waiting for the job to start or finish. The default is all 256 prepared
`data/val.json` questions, eight responses per question, and the training seed.
W&B logging is enabled in an independent evaluation run in the same project and
group as training. The evaluator does not train, update the checkpoint, download
models, or call a judge model. The original CSCC monitor is not required.

The new full/LoRA and automatic scheduling paths have not been exercised on
GPUs. Current scheduling/logging checks use mocked services, not live Slurm or
W&B connections. GPU memory fit, runtime and actual model results remain to be
measured.

The current implementation checks **signed-integer final answers only**, using the unchanged training graders. It is suitable for the prepared DAPO split. Four external evaluation datasets have **not** been selected or integrated; this command must not be described as providing arbitrary MATH, Olympiad, multiple-choice, or symbolic-answer benchmark grading. An unsupported reference is rejected before inference.

## Automatic epoch validation

Copy `.env.example` to `.env` and set `WANDB_API_KEY` before launching training.
Defaults are `WANDB_PROJECT=stride-iclr2027`, `WANDB_RUN_GROUP=stride-comparison`,
`WANDB_MODE=online`, `VALIDATION_MODE=slurm`, `VALIDATION_EVERY_EPOCHS=1`, and
`VALIDATION_ROLLOUTS=8`. Override the input with `VALIDATION_DATA`; the whole file
is evaluated, not a sampled prefix. Set `VALIDATION_PARTITION`,
`VALIDATION_ACCOUNT`, and `VALIDATION_QOS` for site requirements. Blank options
use scheduler defaults, not the training job's allocation settings.

The training allocation must have `sbatch` available and permission to submit
jobs. Validation nodes must access the shared output/model/environment/source
paths and, for online logging, W&B. A completed epoch checkpoint is retained,
its manifest is checked, and exact validation/training input bytes are pinned
under `validation_jobs/input/` before the jobs run. A queued validation job does
not use or reserve one of the eight training GPUs.

Four epochs with the default split produce four jobs and `4 * 256 * 8 = 8,192`
validation responses per training run. Change frequency with
`--validation-every-epochs` or disable automatic submission with
`--validation-mode off`. Corresponding `.env` settings are documented in
[RUN_SETTINGS.md](RUN_SETTINGS.md). These checks do not select a new training
dataset or change rewards.

Job scripts, requests and submission/failure records are under
`RUN/validation_jobs/epoch_NNN/`; evaluator outputs are under
`RUN/validation/epoch_NNN/`. The parent training `done.json` means only that
training completed. It does not wait for or certify validation completion.
Inspect each evaluation's own `done.json` and scheduler logs. Submission failure
is recorded and training continues; no successful metrics are invented.

## Run a checkpoint on the final test split

Final test evaluation is still explicitly submitted using `run.py eval` on a
separate one-GPU allocation. Automatic training validation does not use
`data/test.json`.

Use the original model directory and training JSON that produced the checkpoint. The following example assumes the preparation/training outputs were placed under `work/`; change the paths to your real locations. For 2,048 questions, batch 64 and four epochs, the final checkpoint is `model_128` in full mode or `adapter_128` in LoRA mode.

```bash
python run.py eval \
  --model qwen4b \
  --model-path work/models/qwen4b \
  --checkpoint work/runs/qwen4b/full/local_positive/model_128 \
  --data work/data/test.json \
  --train-data work/data/train.json \
  --out work/evaluation/qwen4b-local-positive-final \
  --seed 42 \
  --rollouts 8 \
  --batch-size 16 \
  --sampler-max-num-seqs 16 \
  --sampler-eager
```

`CUDA_VISIBLE_DEVICES` must contain exactly the one GPU assigned to this evaluation by the scheduler. Preserve the scheduler's assignment; do not copy a physical GPU ID from another job. Use a runtime compatible with the selected model and sampler. A successful CPU fixture does not establish GPU, vLLM, or 8K inference compatibility.

For the 1.7B model, use `--model qwen17b` with its original model directory and a checkpoint trained from that model. For LoRA, use `--adapter /path/to/adapter_128` instead of `--checkpoint`. Evaluation infers mode from this argument; `--finetune-mode` is a training/smoke option and is not accepted by evaluation. To evaluate the unchanged base model, omit both checkpoint arguments; keep `--train-data` so evaluation questions are checked against the planned training split.

For a preliminary check, use a fresh output directory and `--limit 8`. This evaluates a deterministic prefix; it still checks the **entire** input evaluation file for training overlap. It is not a randomly sampled or representative final benchmark by itself. Final evaluation should omit `--limit`.

`--batch-size` is the number of questions in one sampling request. `--rollouts` is responses per question. `--sampler-max-num-seqs` bounds concurrent sequences and does not change the requested rollout count. The context limit is **8,192 prompt-plus-response tokens**; a long question reduces the response budget. Sampling temperature is 1.0, top-p is 1.0, and top-k filtering is disabled, preserving the training sampling protocol.

Use the same evaluation seed across all methods. Each question gets sampling seed `--seed + its input position`, independent of request batch boundaries and without method, job or rank offsets. The evaluation seed may use the supported 63-bit range; the vLLM engine seed and `PYTHONHASHSEED` use its low 32 bits (`seed % 2**32`). These are recorded separately from per-question sampling seeds. The Slurm evaluation wrapper uses an explicit `--seed` first, otherwise the `SEED` environment value, otherwise 42. Dataset selection remains fixed independently of training/evaluation repeats. Stochastic GPU inference is not promised to be bit-for-bit invariant under changes to batching or runtime.

The default request timeout is 3,600 seconds. Increase `--generation-timeout` if a legitimate long batch takes longer. Keep the output fresh for every retry; partial results are retained for inspection, and this evaluator does not resume an interrupted output directory.

## Input contract and leakage checks

The input is a UTF-8 JSON **array** with records of this form:

```json
[
  {"id": "benchmark-name:0001", "question": "What is 1 + 1?", "ground_truth": 2}
]
```

- `id` is a nonempty string or integer. Use dataset-prefixed IDs for unrelated external datasets; the overlap guard deliberately rejects IDs that equal training IDs, even if their question text differs. Prepared DAPO train/validation/test IDs are already disjoint.
- `question` is nonempty text. Duplicate questions are checked using the same presentation normalization as DAPO preparation, including Unicode, whitespace and selected LaTeX wrappers. This is **not** semantic near-duplicate detection.
- `ground_truth` is an integer or a signed-integer string. Floats, fractions, expressions, and option letters are not supported reference labels. The unchanged answer grader may normalize supported numeric *response* representations to an integer; that does not make it a general symbolic grader.
- All evaluation and training records are validated before launching the sampler. Overlapping IDs or normalized question text cause a failure. Question answers are not used for overlap detection.
- `--train-data` is mandatory. If the checkpoint's originating `started.json` includes its training dataset digest, it must match the supplied training file.
- Add `--data-sha256 HASH` and `--train-data-sha256 HASH` to enforce explicit expected input hashes. Otherwise, the input bytes are hashed and their actual hashes are recorded in `eval_spec.json`. For published results, preserve the prepared data manifests and report the recorded hashes.

The evaluator checks direct overlap with the supplied training file. It does not prove the base model's pretraining data were uncontaminated, and it cannot identify paraphrases or shared solution templates. Additional external benchmark exclusions must be included during dataset preparation.

## Checkpoint and model identity

Full mode uses `--checkpoint` and an immutable `model_NNN` directory containing
the exported model configuration, weights and tokenizer assets, plus its sibling
`model_NNN_manifest.json`. Keep the originating `started.json` with the checkpoint.
The original `--model-path` remains the source model used for provenance; it is
not replaced by the trained model directory. Full checkpoints contain FP32 master
weights; inference loads BF16 weights. Use only one of `--checkpoint` or `--adapter`; evaluation infers the mode from
that choice and rejects a directory of the wrong checkpoint type.

LoRA mode uses `--adapter`. An adapter must have:

- `adapter_model.safetensors` and `adapter_config.json` inside its immutable directory;
- a completed sibling file named `<adapter_directory>_manifest.json`, mapping every adapter filename to its SHA256;
- the supported rank-16, alpha-32, dropout-0 ordinary LoRA configuration.

Copy the adapter, its sibling manifest, and the originating run's `started.json` together. The evaluator checks exact adapter file inventory/hashes before loading, before every batch, and at the end. Symlink adapter directories and extra or modified files are rejected. An adapter for another known Qwen model is rejected. For new portable runs, actual model-file content hashes in `started.json` must match the supplied model directory, so moving files between machines is supported without relying on the old absolute path.

Historical adapters without model-content provenance can pass the file/LoRA checks, but `source_model_identity_verified` is false. A historical absolute `base_model_name_or_path` is recorded rather than treated as proof of model identity. The downloaded model's preparation manifest, when present, must agree with the selected model label/repository. For custom model folders without that manifest, the label alone is not architecture or model provenance verification.

## Metrics

The evaluator preserves `training/evaluate.py::summarize` and the same native parser and strict/relaxed answer graders used in training:

- **Pass@1:** mean fraction of correct responses per question. This equals pooled response accuracy when every question has the same number of rollouts.
- **Pass@8:** mean `1 - C(n-c,8) / C(n,8)`, where `n` is sampled responses and `c` is correct responses for a question. With exactly eight rollouts, this is the fraction of questions with at least one correct sample. If any question has fewer than eight responses, Pass@8 is `null` rather than an invented estimate.
- Answer correctness, generation length, truncation, native final-region handling, parser validity and mapped step eligibility are reported separately.
- **Lexical explanation distinctness:** counts distinct normalized final-explanation text among answer-correct responses, within each question. It does **not** establish different valid reasoning strategies, verified reasoning diversity, URT, or correctness of individual steps.

The evaluator does not add a structural or diversity reward, repair incomplete output, choose responses based on correctness, or silently discard malformed responses. Inspect raw examples when interpreting a high lexical-diversity score; gibberish can be lexically distinct.

## Evidence and completion

Each run requires a new `--out` directory. Its files include:

| File | Purpose |
| --- | --- |
| `eval_spec.json` | Actual input/model/checkpoint identities, sampling/engine/hash seeds, budgets, code hashes and protocol |
| `sampled_*.json` | Original sampler responses, token IDs and behavior log probabilities |
| `groups_*.json` | Scored grouped responses using the existing graders and parser |
| `progress.json` | Latest completed-question count; updated during execution |
| `summary.json` | Final aggregate answer, syntax, length and lexical metrics |
| `done.json` | Success marker with expected sample counts and summary/spec hashes |
| `failed.json` | Failure reason when execution aborts after creating the run |
| `wandb.json` | Evaluation W&B run identity and URL, plus epoch/training step when supplied |
| `ipc/worker.log` | vLLM subprocess diagnostics |

`summary.json` alone is not evidence of successful completion: require `done.json` with `complete: true` and matching hashes. Input validation can fail before creating any output. A later failure preserves raw results and does not create `done.json`.

Raw records contain question text, answer keys and generated responses. Keep
them under the run-output directory; do not add them to the code repository.
W&B receives scalar metrics and nonsecret experiment metadata, not raw rollout
files or answer keys. Automatic code uploads are disabled. Each evaluator gets
its own fresh W&B run; it never appends to the optimizer's active run. Epoch and
training-step fields link the result to its source checkpoint.

`WANDB_MODE=offline` keeps W&B logs locally without uploading;
`WANDB_MODE=disabled` disables tracking. Online mode requires `WANDB_API_KEY` in
`.env` or the exported environment; it is never a command-line argument. Use
`--wandb-mode`, `--wandb-project`, `--wandb-entity`, or `--wandb-group` for explicit
overrides. The `.env` parser does not execute or expand shell commands.

## CPU checks and runtime limits

```bash
python -m unittest discover -s tests -p test_portable_eval.py -v
```

The earlier LoRA-only CPU tests covered arbitrary row counts, unsupported
references, overlap (including unselected rows), hashes, checkpoint mutation and
model/data mismatches, fixed question seeds, exact rollout budgets, Pass@8's
sample-count requirement, immutable output directories, and failure markers.
Current configuration, scheduling and W&B tests also use mocks. These are
software checks, not GPU validation results or real model predictions. Training
commands do not run the unit-test suite; they submit the held-out evaluation
jobs described above.
