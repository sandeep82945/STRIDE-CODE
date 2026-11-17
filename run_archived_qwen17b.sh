#!/usr/bin/env bash
# Portable invocation derived from the archived all-reference 1.7B launch.
# Original training modules are preserved; paths and service settings are local.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
exec "${PYTHON:-python}" run.py train --no-settings --hf-upload off \
  --model qwen17b --model-path "${STRIDE_MODEL_PATH:-models/qwen3-1.7b}" \
  --data "${STRIDE_TRAIN_DATA:-data/train.json}" \
  --out "${STRIDE_RUN_OUT:-runs/qwen17b_lora_allref_seed42}" \
  --seed 42 --arm local_positive --finetune-mode lora \
  --epochs 4 --batch-size 64 --rollouts 8 --alpha 1 \
  --novelty-reference-scope all_rollouts \
  --learning-rate 2e-5 --warmup-steps 10 --kl-coefficient 0 \
  --validation-mode off --wandb-mode disabled \
  --sampler-max-num-seqs 32 --generation-timeout 3600 \
  --keep-all-checkpoints --resume-state-retention latest-epoch \
  --learner-microbatch-size 1 "$@"
