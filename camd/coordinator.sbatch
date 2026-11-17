#!/bin/bash
#SBATCH --job-name=system-100-parallel-coordinator
#SBATCH --account=faculty-acc
#SBATCH --partition=faculty
#SBATCH --qos=xdqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=5-00:00:00
#SBATCH --exclude=auh7-1b-gpu-188,auh7-1b-gpu-192
#SBATCH --output=/path/to/cluster-user/system/stride_sample100_20260915/logs/job-%j.out
#SBATCH --error=/path/to/cluster-user/system/stride_sample100_20260915/logs/job-%j.err
set -euo pipefail
: "${1:?Provide the immutable parallel evaluation plan path}"
# Submit on the original evaluation node so /proc identifies only its own
# original batch controller. This coordinator does not request or use a GPU.
exec /usr/bin/python3 -I -S /path/to/cluster-user/system/stride_sample100_20260915/code/stride_handoff/camd/parallel_eval_coordinator.py --plan "$1"
