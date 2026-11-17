#!/bin/bash
# Supply the site's --partition, --account and --qos on the sbatch command line.
#SBATCH --job-name=stride-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=stride-eval-%j.out
set -euo pipefail
cd -- "${SLURM_SUBMIT_DIR:?Submit from the handoff repository root}"
# Explicit final-test/standalone evaluation wrapper. Automatic epoch validation
# submits its own immutable script. Both log to W&B using the resolved settings.
# run.py reads .env literally; do not shell-source it.
# Evaluation infers mode from --checkpoint (full) or --adapter (LoRA).
# Use the same exported experiment seed as the comparison array. An explicit
# evaluation --seed takes precedence; the launcher rejects duplicates/invalids.
explicit_seed=false
for arg in "$@"; do
  case "$arg" in --seed|--seed=*) explicit_seed=true; break ;; esac
done
if [[ "$explicit_seed" == true ]]; then
  exec "$PWD/.venv/bin/python" run.py eval "$@"
fi
exec "$PWD/.venv/bin/python" run.py eval --seed "${SEED:-42}" "$@"
