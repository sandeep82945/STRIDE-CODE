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
# Generic explicit stage wrapper; root settings are loaded by run.py.
# Full/LoRA selection: CLI > FINETUNE_MODE environment > run_settings.toml.
exec "$PWD/.venv/bin/python" run.py "$@"
