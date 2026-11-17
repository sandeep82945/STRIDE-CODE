#!/bin/bash
# Optional: all four methods, one eight-GPU allocation per array task.
# Default concurrency is one; --array=0-3%4 needs up to 32 training GPUs plus
# separate one-GPU validation jobs. W&B/validation settings come from .env.
#SBATCH --job-name=stride-compare
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --array=0-3%1
#SBATCH --output=stride-%A_%a.out
set -euo pipefail
arms=(local_positive single_step token_uniform grpo)
case "${SLURM_ARRAY_TASK_ID:-}" in 0|1|2|3) ;; *) echo 'Expected array task 0..3' >&2; exit 2;; esac
export ARM="${arms[$SLURM_ARRAY_TASK_ID]}"
cd -- "${SLURM_SUBMIT_DIR:?Submit from the handoff repository root}"
exec bash scripts/slurm_run.sbatch "$@"
