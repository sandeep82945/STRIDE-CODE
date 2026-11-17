# Data and model preparation

Run commands from the repository root after installing its Python dependencies.
Neither model weights nor dataset payloads are bundled in the code package.
Downloads use pinned public Hugging Face revisions. The scripts print aggregate
counts, artifact paths, and hashes; they do not print questions or answers.

## Download a model

```bash
python scripts/download_model.py --model qwen4b --out models/qwen4b
python scripts/download_model.py --model qwen17b --out models/qwen17b
```

| Alias | Upstream model | Pinned revision |
|---|---|---|
| `qwen4b` | `Qwen/Qwen3-4B-Instruct-2507` | `cdbee75f17c01a7cc42f958dc650907174af0554` |
| `qwen17b` | `Qwen/Qwen3-1.7B` | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |

The output directory must not exist. Each complete download includes
`model_manifest.json` with the alias, upstream repository, exact revision, and
SHA-256/size of every downloaded asset, excluding Hugging Face's `.cache`
metadata. Supply this directory to training as `--model-path models/qwen4b` or
`--model-path models/qwen17b`. The pinned snapshot is the upstream starting model,
not the warmed-up checkpoint or a recovered CSCC training checkpoint.

The script records `.download_request.json` before downloading. It makes at most
three download attempts per invocation, with short delays between attempts;
`--max-attempts` accepts a limit from one to five. If all attempts fail, the
partial directory and Hugging Face download cache are retained. Continue the
same request with:

```bash
python scripts/download_model.py --model qwen4b --out models/qwen4b --resume
```

`--resume` requires an existing `.download_request.json` with the exact same
alias, upstream repository, and pinned revision. It cannot merge another model
or revision into a partial directory. Hugging Face's local download cache handles
continuation of interrupted assets. If the model already has a complete
manifest, `--resume` verifies every recorded asset's size/hash and the asset
inventory, then returns `verified_existing_download` without downloading or
overwriting anything. Corrupted completed assets fail verification and are not
silently repaired. A directory without matching request metadata is refused;
choose a new path instead. Request/cache metadata are excluded from the manifest's
model asset inventory. Do not run simultaneous downloads into the same directory.

Do not commit model files to GitHub. Model use and redistribution remain subject
to the respective upstream terms.

## Freeze new English DAPO splits

```bash
python scripts/prepare_data.py \
  --out data \
  --train-count 2048 \
  --val-count 256 \
  --test-count 512 \
  --seed 42
```

This downloads `en/train-00000-of-00001.parquet` from
`open-r1/DAPO-Math-17k-Processed`, revision
`31dd309567e3da778038cc87d868b6097a3ccf68`, and requires its SHA-256 to equal
`40f672fe8b6dbeee953ae5acc75a38e99144e682450e07c625c87c734ec76285`.
An existing download can be supplied with
`--source-parquet /absolute/path/to/en-train.parquet`; it receives the same hash
check. There is no option to bypass this provenance check.

The new output directory contains `train.json`, `val.json`, `test.json`, and
`manifest.json`. The three data files are JSON lists with this schema:

```json
[
  {"id": "synthetic-example-id", "question": "Synthetic example: what is 2 + 3?", "ground_truth": 5}
]
```

The example above is synthetic. Actual IDs and questions come from the pinned
source without rewriting. Answers are integers, checked against both upstream
answer fields. No correctness, model response, difficulty, or answer-length
filter influences selection. Source groups with repeated normalized questions
are removed entirely; the remaining source-order records are shuffled with the
specified seed, then assigned to test, validation, and training in that order.
All three splits are disjoint by source ID and normalized question text. The
manifest records provenance, seed, counts, exclusions, verification results,
preparation-code hash, and hashes of the final split files. The output directory
must not already exist, even if a prior attempt produced identical files.

These are **fresh splits**, because the original CSCC train/evaluation payloads
were not recovered. This is not an exact reproduction of the old experiment.
Validation and test are new holdouts from the upstream DAPO training split; they
are not four external benchmark test sets. Reuse one frozen training/validation
set across all methods and training seeds. Do not regenerate a different data
split for each baseline.

## Exclude benchmark and previously used questions before splitting

Choose the external evaluation datasets first and supply their question files,
along with every available prior training/evaluation input list:

```bash
python scripts/prepare_data.py \
  --out data_decontaminated \
  --train-count 2048 --val-count 256 --test-count 512 --seed 42 \
  --exclude-json /path/to/benchmark_questions.json \
  --exclude-json /path/to/previously_used_questions.jsonl
```

`--exclude-json` is repeatable. Supported inputs include:

- A JSON list of question strings.
- JSON lists/objects or JSONL records containing `question`, `problem`, or string
  `prompt` fields. For a chat-format `prompt`, user-message content is matched.
- A `questions` or `problems` list of strings in a JSON object.
- Explicit original DAPO identifiers such as `{"source_ids": ["123", "456"]}`,
  `{"source_id": "123"}`, or `{"dataset": "dapo", "id": "123"}`.

Foreign benchmark `id` fields alone are **not** interpreted as DAPO source IDs,
because unrelated datasets commonly share numeric IDs. Original DAPO IDs can be
supplied with the explicit fields above. If a supplied file contains no supported
records, preparation fails instead of silently claiming it was checked. Dataset
specific wrappers should be converted into these schemas before use; matching an
entire instruction-wrapped prompt does not automatically match the bare problem.
Every exclusion file's hash and aggregate extracted record counts are recorded.

Normalization folds Unicode presentation, letter case, whitespace, and selected
LaTeX presentation variants. It does not prove semantic decontamination, detect
all rephrasings, or establish freedom from the model's pretraining exposure.
Only the supplied exclusion files are checked. The four external evaluation
datasets have not been chosen or downloaded by these scripts; do not describe
this DAPO holdout as an external benchmark evaluation.

## Offline checks

```bash
python -m unittest discover -s tests -p 'test_data.py' -v
python scripts/prepare_data.py --help
python scripts/download_model.py --help
```

Tests use synthetic fixtures and require no network, GPU, downloaded dataset, or
ML packages. They check answer/schema validation, deterministic split selection,
normalized duplicate removal, exclusion handling, disjointness, refusal to
overwrite frozen data, bounded download retries, matching-request continuation,
and read-only completed-model verification. They do not verify live model weight
downloads or GPU training.

The upstream data license is not inferred or replaced by this code package's
license. Review the source dataset card and upstream terms before redistributing
data. Keep downloaded inputs, weights, generated rollouts, and checkpoints out of
the source repository.
