# STRIDE archived training source

This package preserves the training implementation from the archived
September 25, 2026 all-reference STRIDE source. All 28 Python modules under
`training/` are included byte for byte. Personal account identifiers and machine
paths in the optional cluster helpers have been replaced with placeholders.
The included source manifest records the original training-file hashes.

The default invocation below selects local-positive STRIDE with all eligible
rollouts as references and auxiliary credit only for correct eligible responses.
The generic historical `run_settings.toml` supports other configurations, so the
provided launch script uses `--no-settings` and explicit recovered settings.

## Environment and data

Use a compatible Linux GPU environment. The distributed implementation uses
eight GPUs, with four learners and four separate inference workers.

```bash
# NVIDIA environment
python -m pip install -r requirements-nvidia-lock.txt

# For ROCm, use the site's matching ROCm PyTorch/vLLM environment instead.
python scripts/download_model.py --model qwen17b --out models/qwen3-1.7b
python scripts/prepare_data.py --out data --train-count 2048 --val-count 256 --test-count 512 --seed 42
python verify_archived_source.py --train-data data/train.json
```

The recorded training-file SHA256 is
`d319351d80df073e947576189f41c23b549584ecca4bad660a2b752427f429cc`.

## Run the archived 1.7B profile

```bash
bash run_archived_qwen17b.sh --dry-run
bash run_archived_qwen17b.sh
```

The script sets Qwen3-1.7B, non-thinking mode, LoRA rank 16 / alpha 32,
2,048 training questions, four epochs, global prompt batch 64, K=8,
learning rate 2e-5, 10 learning-rate warmup updates, diversity coefficient 1,
KL coefficient 0, and seed 42. Total prompt-plus-response context is 8,192
tokens. Structural reward is disabled, and validation is off.

Set `STRIDE_MODEL_PATH`, `STRIDE_TRAIN_DATA`, or `STRIDE_RUN_OUT` to override
local paths. Additional command-line arguments override the script defaults.
For example, to enable W&B and checkpoint upload after configuring `.env`:

```bash
cp .env.example .env
# Fill your credentials locally. Never commit .env.
bash run_archived_qwen17b.sh --wandb-mode online --hf-upload on
```

The script starts a fresh run. The historical final finishing job resumed an
existing state; resume remains available through the original `--resume-from`
option when a matching state is available.

## Source verification

```bash
python verify_archived_source.py
```

`RELEASE_MANIFEST.json` lists every included source hash, the unchanged archived
training hashes, and the non-training files changed for anonymization or added
for this package. Historical documentation and tests remain included.
Machine-specific helpers under `camd/` require local paths and scheduler settings.

The source includes the original comparison arms; the supplied launch script
selects STRIDE. This archive is distinct from the separately cleaned, final-only
release.

Apache-2.0. Retained third-party notices and licenses apply.
