# STRIDE

**Step-level diversity credit for reinforcement-learning post-training of reasoning models.**

STRIDE adds a bounded, non-negative novelty bonus to individual *reasoning steps* of
correct rollouts, on top of the usual outcome advantage. The bonus rewards a step for
being representationally distinct from the steps of other rollouts to the same question,
so a policy is encouraged to find genuinely different solution paths rather than to
re-derive one path with higher confidence.

This repository is the archived training implementation behind the paper, released so the
method, the exact training configuration and the data provenance can be inspected and
re-run. Please read [Scope and status](#scope-and-status) before using it — the
limitations are substantive and stated up front.

> **Citation** — fill in before release:
> ```bibtex
> @inproceedings{stride,
>   title     = {<paper title>},
>   author    = {<authors>},
>   booktitle = {<venue>},
>   year      = {2027}
> }
> ```

---

## Method

For one question, sample $K$ responses and grade their final answers. For each response
that is answer-correct *and* parses into at least two well-formed steps, extract the
frozen reference model's hidden states over that step's original token span and pool them
into a step embedding $e_{ij}$.

With $\phi_{ij} = e_{ij}/\lVert e_{ij}\rVert_2$, form the Gram matrix of all reference
steps and ridge it with $\varepsilon = 10^{-3}$:

$$K_x = \Phi_x \Phi_x^\top + \varepsilon I$$

For a target response $i$, order the kernel as (all steps of the *other* reference
rollouts, then $i$'s own preceding steps) and take a Cholesky factor $LL^\top$. The
conditional variance of step $j$ is $v_{ij} = L_{p_{ij},p_{ij}}^2$ — how much of step
$j$ is *not* explained by everything conditioned on. The novelty credit is

$$u_{ij} = \mathrm{clip}(v_{ij} - \varepsilon,\ 0,\ 1),\qquad u_{ij} \leftarrow 0 \ \text{ if } u_{ij} \le \varepsilon$$

and the auxiliary advantage $\alpha\, u_{ij}$ is added to the outcome advantage on exactly
the tokens of step $j$, and nowhere else.

Three properties matter:

- **Bounded.** $u_{ij} \in [0,1]$, so $\alpha$ is directly interpretable and the bonus
  cannot blow up when the outcome advantage collapses in an all-correct group.
- **Local-positive.** There is no step-wise centering, so a correct response never
  receives negative auxiliary credit. The historical centered variant — which *can*
  reverse a correct token's total advantage — is retained in
  [`training/credit.py`](training/credit.py) as reference code but is not selectable.
- **No rescaling.** No per-group z-score, no RMS matching, no division by step count or
  span length. Those divisions are what made the diversity term ~1/300 the scale of the
  outcome advantage in an earlier implementation; see the derivation in
  [`training/production_stride.py`](training/production_stride.py).

Structural/format reward is disabled throughout and $\alpha$ is constant from the first
update. Full equations and the exact ablation definitions are in
[`docs/ABLATIONS.md`](docs/ABLATIONS.md).

### Ablation arms

All arms share prompts, parser, answer reward and GRPO policy loss, and differ only in the
auxiliary advantage placed on a mapped token:

| Arm | Auxiliary advantage |
|---|---|
| `local_positive` | $\alpha\,u_{ij}$ on its own step $j$ — **the method** |
| `token_uniform` | each response's total bonus mass spread uniformly over its mapped step tokens |
| `single_step` | $\alpha\,u_i$ from one pooled vector per response, broadcast over its step tokens |
| `grpo` | $0$ — outcome advantage only |

---

## Scope and status

This is an **archival release of a training implementation**, not a turnkey reproduction
of the paper's measurements. Specifically:

- **No results are claimed in this repository.** It contains no logs, checkpoints, or
  Pass@1/Pass@K numbers.
- **The original experiment's data payloads were not recovered.** The preparation script
  regenerates *fresh* splits from a pinned upstream source. Re-running this code produces
  a comparable experiment, **not a bitwise rerun** of the paper's.
- **Executed:** the CPU suite (credit algebra against independent float64 Schur-complement
  identities, real tiny-Qwen LoRA/DDP updates, parser/grader, data preparation, launcher
  construction, resume) — see [`docs/EXTENSIVE_TEST_REPORT.md`](docs/EXTENSIVE_TEST_REPORT.md).
- **Not executed:** the default full-parameter FSDP path, GPU training on the pinned public
  vLLM release, cluster scheduling, and convergence at paper scale. vLLM 0.17.0 is a
  *candidate* runtime, resolved from published metadata but never GPU-smoke-tested here.
- **Known scientific limitation:** valid step markers plus an accepted final answer do not
  certify that reasoning is correct or semantically distinct. Gibberish with well-formed
  step markers can still earn diversity credit. No semantic verifier is included.

[`docs/PROVENANCE.md`](docs/PROVENANCE.md) records what changed from the original source and
why; [`docs/VALIDATION.md`](docs/VALIDATION.md) records what was and was not run.

---

## Quickstart

Python 3.11+ and a Linux GPU node. The training launch requires **one node with eight
GPUs** — four learners plus four separate vLLM inference workers.

```bash
# 1. Environment
python -m pip install -r requirements-nvidia-lock.txt   # pinned; see note below
python scripts/check_runtime.py --gpus 8                # verify versions + GPU primitives

# 2. Base model (pinned upstream revision)
python scripts/download_model.py --model qwen17b --out models/qwen3-1.7b

# 3. Fixed data splits (2,048 / 256 / 512, seed 42)
python scripts/prepare_data.py --out data \
  --train-count 2048 --val-count 256 --test-count 512 --seed 42

# 4. Verify this package and the recorded training data
python verify_archived_source.py --train-data data/train.json
```

For a CPU-only check of the algorithm and plumbing — no GPU, no weights, no network:

```bash
python -m pip install -r requirements-cpu.txt
python scripts/run_tests.py
```

> **Runtime note.** `requirements-nvidia-lock.txt` pins the candidate portable runtime
> (torch 2.10.0 / vLLM 0.17.0). The original cluster ran a vLLM pre-release build; see
> [`docs/PROVENANCE.md`](docs/PROVENANCE.md#runtime). For ROCm, use your site's matching
> ROCm PyTorch/vLLM environment and export `STRIDE_RUNTIME=rocm`.

### Pinned inputs

| Input | Identity |
|---|---|
| Qwen3-1.7B | revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |
| Qwen3-4B-Instruct-2507 | revision `cdbee75f17c01a7cc42f958dc650907174af0554` |
| Dataset | `open-r1/DAPO-Math-17k-Processed`, revision `31dd309567e3da778038cc87d868b6097a3ccf68` |
| Train split (2,048) | `d319351d80df073e947576189f41c23b549584ecca4bad660a2b752427f429cc` |
| Validation split (256) | `2d414c21faa8a1ebccf29f8a72f6feada213daec3457d8397160ff50340c2c8f` |
| Test split (512) | `6074c05fd6f2a85a763fe6d387ac815c45f7854e031c8572335a7891c9735327` |

Data preparation enforces the upstream parquet hash with no bypass option, removes
duplicate normalized questions, and guarantees the three splits are disjoint by both source
ID and normalized text. See [`docs/DATA.md`](docs/DATA.md), including how to exclude
external benchmark questions before splitting.

---

## Reproducing the archived run

**Use the dedicated launch script.** It encodes the recovered settings explicitly and runs
fully offline — no W&B, no Hugging Face upload:

```bash
bash run_archived_qwen17b.sh --dry-run   # print the exact resolved command
bash run_archived_qwen17b.sh
```

| Setting | Archived run |
|---|---|
| Model | Qwen3-1.7B, non-thinking prompts |
| Fine-tuning | LoRA rank 16, alpha 32, dropout 0 |
| Arm | `local_positive` |
| Novelty reference scope | `all_rollouts` (correct *and* incorrect eligible rollouts as references; bonus only to correct ones) |
| Learning rate | 2e-5, 10 update linear warmup, then constant |
| KL coefficient | 0 |
| Diversity coefficient $\alpha$ | 1.0, constant |
| Questions / K / epochs | 2,048 / 8 / 4 (global prompt batch 64 → 128 updates) |
| Context | 8,192 prompt + response tokens |
| Seed | 42 |
| Validation / W&B / HF upload | off / disabled / off |

Override local paths with `STRIDE_MODEL_PATH`, `STRIDE_TRAIN_DATA` or `STRIDE_RUN_OUT`;
extra arguments pass through to the driver.

> **`run_settings.toml` is a different configuration.** The generic runner
> (`sbatch run.sbatch`) defaults to Qwen3-4B, **full-parameter FSDP**, learning rate 1e-6,
> no warmup, `correct_only` references, online W&B and automatic Hugging Face upload. That
> path is a superset of the archived experiment and has **not** been executed. Do not treat
> it as the paper's configuration — it is documented separately in
> [`RUN_SETTINGS.md`](RUN_SETTINGS.md).

### Evaluating a checkpoint

Final test evaluation is a separate one-GPU job:

```bash
python run.py eval \
  --model qwen17b --model-path models/qwen3-1.7b \
  --adapter runs/<run>/adapter_128 \
  --data data/test.json --train-data data/train.json \
  --out evaluation/qwen17b-local-positive-final \
  --seed 42 --rollouts 8 --batch-size 16
```

The evaluator reports Pass@1, Pass@8, parser validity, truncation and lexical explanation
distinctness, and refuses to run if evaluation and training inputs overlap by ID or
normalized question text. It grades **signed-integer final answers only** and is suitable
for the prepared DAPO split: the four planned external evaluation datasets have not been
selected or integrated, so this is not MATH, Olympiad, multiple-choice or symbolic-answer
benchmark grading. Details and the full input contract:
[`docs/EVALUATION.md`](docs/EVALUATION.md).

---

## Repository layout

| Path | Contents |
|---|---|
| `training/` | The implementation. 28 modules, byte-identical to the archived source. |
| `training/production_stride.py` | Pinned STRIDE kernel — the exact spec, Cholesky conditional variances. |
| `training/credit.py` | Five reference credit rules on shared features; cross-checks the pinned kernel. |
| `training/experiment_credit.py` | The four selectable comparison arms. |
| `run.py` | Launcher: freezes a content-addressed source snapshot, seeds the run, allocates GPUs. |
| `scripts/` | Model download, data preparation, runtime checks, release packaging. |
| `tests/` | CPU suite — algorithm properties, real tiny-model updates, parser, data, resume. |
| `docs/` | Algorithm, data, evaluation, provenance, validation and audit reports. |
| `camd/` | Cluster helpers from the original site. **Placeholders, not runnable as shipped** — they need real paths and scheduler settings. |
| `configs/paper_split.json` | Recorded split identity for the archived run. |

---

## Verification and provenance

Every file in this package is hash-pinned. The verifier checks the package against its
manifest and separately confirms that the 28 training modules still match the original
archived source:

```bash
$ python verify_archived_source.py
Verified 151 package files
Verified 28 unchanged archived training modules
```

`RELEASE_MANIFEST.json` records every source hash, the original training-module hashes, and
which non-training files were changed for anonymization or added for this package. Personal
account identifiers and machine paths in `camd/` were replaced with placeholders; model
weights, datasets, credentials, logs and checkpoints are excluded by `.gitignore` and were
never part of the archive.

`run.py` additionally snapshots `training/` into a content-addressed cache before launching,
so edits made after a job starts cannot mix into a running experiment, and the recorded
scientific contract rejects a resume whose configuration does not match.

---

## License

Apache-2.0 — see [`LICENSE`](LICENSE).

`training/pilot_grading/math_reference.py` retains its upstream Apache-2.0 notices for
ByteDance, EleutherAI and Hugging Face (adapted from `lm-evaluation-harness` MATH
normalization); the license text is included under [`licenses/`](licenses/). See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

Model weights and dataset payloads are **not** redistributed here; their upstream terms
apply. Review the Qwen model cards and the DAPO dataset card before redistributing either.
