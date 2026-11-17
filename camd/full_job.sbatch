#!/bin/bash
#SBATCH --job-name=system-s17
#SBATCH --account=faculty-acc
#SBATCH --partition=faculty
#SBATCH --qos=gtqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=3-00:00:00
#SBATCH --no-requeue
#SBATCH --exclude=auh7-1b-gpu-188,auh7-1b-gpu-192
#SBATCH --output=/path/to/cluster-user/system/stride_allref_correctonly_20260925/logs/job-%j.out
#SBATCH --error=/path/to/cluster-user/system/stride_allref_correctonly_20260925/logs/job-%j.err
set -euo pipefail
stride_root=/path/to/cluster-user/system/stride_allref_correctonly_20260925
stride_code="$stride_root/code/stride_handoff"
stride_arm="$1"
case "$stride_arm" in local_positive|token_uniform) ;; *) exit 2 ;; esac
stride_smoke="$stride_root/checkpoints/$stride_arm-smoke-$SLURM_JOB_ID"
srun --exact --ntasks=1 --cpus-per-task=32 --gres=gpu:8 --mem=256G \
  /bin/bash "$stride_code/camd/enter_full_runtime.sh" gpu "$stride_code/camd/full_stage.py" \
  smoke --arm "$stride_arm"
/usr/bin/python3 -I -S "$stride_code/camd/full_host_gate.py" "$SLURM_JOB_ID" smoke "$stride_arm"
srun --exact --ntasks=1 --cpus-per-task=32 --gres=gpu:8 --mem=256G \
  /bin/bash "$stride_code/camd/enter_full_runtime.sh" gpu "$stride_code/camd/full_stage.py" \
  train --arm "$stride_arm" --smoke-output "$stride_smoke"
/usr/bin/python3 -I -S "$stride_code/camd/full_host_gate.py" "$SLURM_JOB_ID" train "$stride_arm"
