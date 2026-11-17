#!/bin/bash
#SBATCH --job-name=system-stride100-evaluate
#SBATCH --account=faculty-acc
#SBATCH --partition=faculty
#SBATCH --qos=xdqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=5-00:00:00
#SBATCH --exclude=auh7-1b-gpu-188
#SBATCH --output=/path/to/cluster-user/system/stride_sample100_20260915/logs/job-%j.out
#SBATCH --error=/path/to/cluster-user/system/stride_sample100_20260915/logs/job-%j.err
set -euo pipefail
/usr/bin/python3 -I -S /path/to/cluster-user/system/stride_sample100_20260915/code/stride_handoff/camd/run_evaluations.py
