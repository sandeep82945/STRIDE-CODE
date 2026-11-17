#!/bin/bash
# Submit from the handoff root; add site --partition/--account/--qos as required.
#SBATCH --job-name=stride
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=stride-%j.out
set -euo pipefail
cd -- "${SLURM_SUBMIT_DIR:?Submit from the handoff repository root}"
# Edit run_settings.toml in the root. Credentials go in root .env.
# CLI options override the file, e.g. sbatch run.sbatch --finetune-mode lora.
# Epoch validation requests separate GPUs; it does not wait inside training.
if [[ "${STAGE:-train}" != train ]]; then
  echo 'This runner trains directly; use run.py smoke/eval for separate checks' >&2
  exit 2
fi
exec "$PWD/.venv/bin/python" run.py train "$@"
