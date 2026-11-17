# Edit settings and run

The editable configuration is [run_settings.toml](run_settings.toml), in this
folder. Put credentials in `.env`; keep the API key out of the settings file.
This package uses the custom PyTorch FSDP/DDP + vLLM trainer.

## Prepare once

Follow [README.md](README.md) to install the environment, download the model and
prepare the fixed DAPO split. The default split has 2,048 training, 256 validation
and 512 test questions. All methods use the same files. Changing a training seed
does not create a different dataset.

```bash
cp .env.example .env
```

Fill `WANDB_API_KEY`, `HF_USERNAME` and `HF_TOKEN` in `.env`. Set the W&B project, group and optional team in
`run_settings.toml`. Existing `WANDB_*` and `VALIDATION_*` overrides in `.env`
remain supported, but override the settings file.

## Main settings

```toml
[training]
model = "qwen4b"
arm = "local_positive"
finetune_mode = "full"     # Change to "lora" for the earlier adapter settings
batch_size = 64
rollouts = 8
epochs = 4
seed = 42
alpha = 1.0
learning_rate = "auto"
```

Edit the existing entries in the file; do not add a second `[training]` section.

| Setting | Choices / meaning |
|---|---|
| `model` | `qwen4b`: Qwen3-4B-Instruct-2507; `qwen17b`: Qwen3-1.7B |
| `finetune_mode` | `full` (default), or `lora` |
| `arm` | `local_positive`, `token_uniform`, `single_step`, `grpo` |
| `batch_size` | Global questions per update; divisible by four and must divide the dataset size exactly |
| `rollouts` | Responses per question, K; default 8 |
| `epochs` | Passes through the training split; default 4 |
| `alpha` | STRIDE diversity coefficient; constant from the first update |
| `learning_rate` | `"auto"`: full `1e-6`, LoRA `1e-4`; or a positive numeric value |
| `seed` | Shared training seed across the compared methods |
| `model_path`, `data`, `out` | Model, training dataset and fresh output paths |

There is no structural reward or warmup. Prompt plus response is capped at
8,192 tokens. LoRA targets attention and MLP projections in every decoder block:
rank 16, LoRA alpha 32, dropout 0. The training `alpha` setting controls STRIDE,
not LoRA scaling. Keep training mode and budgets matched across methods.

- `local_positive`: each eligible correct step gets its positive novelty bonus.
- `token_uniform`: preserves each response's total bonus and spreads it uniformly
  over its mapped step tokens.
- `single_step`: pools the response's mapped reasoning into one step and computes
  its novelty against other eligible correct responses.
- `grpo`: correctness advantage with no diversity bonus.

## Launch

Run these commands from the extracted `stride_handoff` folder. Add your cluster's
account, partition and QoS arguments to `sbatch` as required.

```bash
# Inspect the exact resolved command; no GPU or W&B connection is made.
python run.py train --dry-run

# Submit the selected settings from run_settings.toml.
sbatch run.sbatch

# Or override one setting for this submission.
sbatch run.sbatch --finetune-mode lora

# Optional: all four methods, one eight-GPU allocation at a time.
sbatch scripts/slurm_compare.sbatch
```

The comparison array changes only the method; it retains the selected model,
full/LoRA mode, seed and budgets from the root settings. For another model, edit
`model`, or export `MODEL=qwen17b` with `sbatch --export=ALL,MODEL=qwen17b ...`.
`ARM`, `FINETUNE_MODE` and `SEED` exported overrides are also supported. Do not
pass `--arm` to a comparison array unless you intend to override its arm choice.

The root `run.sbatch` and compatibility `scripts/slurm_run.sbatch` use the same
settings. The generic `scripts/slurm_train.sbatch` accepts an explicit stage and
CLI arguments, e.g. `sbatch scripts/slurm_train.sbatch train --epochs 2`.

With 2,048 training questions, batch sizes such as 16, 32, 64 and 128 divide the
dataset exactly; memory fit must still be checked for the chosen mode.

Each job uses one node with eight GPUs: four learners and four samplers. This is
not a multi-node launcher. Changing `batch_size` does not change GPU allocation.
The two modes have different memory requirements; full FSDP/vLLM GPU execution
has not been verified on the recipient's cluster.

## Paths and precedence

Relative paths in TOML are relative to that TOML file's folder. Path templates
support `{model}`, `{finetune_mode}`, `{arm}` and `{seed}` and use the effective
settings after overrides. Explicit CLI paths remain relative to the current
working directory. The default training output is:

```text
runs/{model}/{finetune_mode}/comparison_seed{seed}/{arm}
```

Use a fresh `out` for reruns; existing attempt directories are rejected. For a
separate profile, copy the TOML and use `--settings /path/to/profile.toml`. Adjust
relative data/model paths if the profile is in a different folder. To use only
explicit CLI arguments and environment defaults, use `--no-settings`.

Precedence is **CLI > exported environment > `.env` > TOML > built-in defaults**
for the supported overrides. `.env` accepts W&B/validation and Hugging Face credential/enable settings; the
training environment names are documented in `run_settings.py`. The launcher
parses `.env` literally; do not shell-source it. A custom credentials file can
be selected with `--env-file /path/to/config.env`.

## Validation and W&B

The `[validation]` section defaults to `mode = "slurm"`, `every_epochs = 1`,
`rollouts = 8`, and `data = "data/val.json"`. Set `partition`, `account` and `qos`
there when the cluster requires them. Blank values use scheduler defaults,
not the training allocation's settings. The training node needs `sbatch` and
shared model, checkpoint, source and Python-environment paths.

Each epoch submits a separate one-GPU validation job. Training continues while
validation queues or runs. Four epochs evaluate `4 × 256 × 8 = 8,192` responses
per training run. Set `mode = "off"` to disable automatic validation. The held-out
final test file is not used by these jobs.

The `[wandb]` section defaults to online project `stride-iclr2027`, group
`stride-comparison`. Training and epoch evaluations have independent run IDs;
evaluations record checkpoint, method, epoch and training step for comparison.
Only scalar metrics and configuration are uploaded. Keys stay in the environment;
raw responses and checkpoint weights remain local. `mode = "offline"` keeps
local W&B logs; `"disabled"` turns tracking off. Smoke always skips validation
and defaults to disabled W&B unless explicitly enabled by CLI/environment.

Outputs include:

- `started.json`: actual resolved settings and provenance.
- `model_NNN/` in full mode or `adapter_NNN/` in LoRA mode, with sibling manifests.
- `state_NNN/`: optimizer and RNG state for verified resume.
- `validation_jobs/epoch_NNN/`: submission requests/status and Slurm logs.
- `validation/epoch_NNN/summary.json`: validation metrics; its own `done.json`
  confirms successful completion.
- `wandb.json`: that process's W&B run ID and URL.

The parent `done.json` confirms training completion, not asynchronous evaluation
completion. Validation reports correctness, formatting/truncation and lexical
variation; it does not establish semantic reasoning diversity. See
[evaluation details](docs/EVALUATION.md). Automatic Hugging Face publication is enabled after successful training.
Choose epochs, prefix and description in the root `[huggingface]` section.
STRIDE variants upload privately and GRPO publicly. Local checkpoints are kept.
See [HF_UPLOAD.md](HF_UPLOAD.md) for setup, contents and retrying failed uploads.

## Hugging Face upload settings

`[huggingface]` contains `enabled = true`, `epochs = [2, 3, 4]`,
`repo_prefix = "stride-iclr2027"`, optional `description`, and `retries = 3`.
Set `HF_USERNAME` and `HF_TOKEN` in `.env`. Upload runs only after successful
training; smoke/evaluation never publish. A one-epoch pilot needs `epochs = [1]`
or uploads disabled. Failed uploads can be retried with
`python upload_checkpoints.py --run-dir /path/to/completed/run` without retraining.
