#!/bin/bash
set -euo pipefail
stride_device_mode="$1"
shift
stride_root=/path/to/cluster-user/system/stride_allref_correctonly_20260925
stride_code="$stride_root/code/stride_handoff"
stride_image=/path/to/cluster-user/kirill/images/vllm_openai_rocm_v0.20.1_official.sif
stride_python=/path/to/cluster-user/system/stride/code/.venv-stride-rocm/bin/python
: "${SLURM_JOB_ID:?Run through Slurm}" "${SLURM_STEP_ID:?Run within srun}"
for stride_key in SLURM_JOB_ID SLURM_STEP_ID SLURMD_NODENAME SLURM_JOB_GPUS SLURM_STEP_GPUS SLURM_CPUS_PER_TASK; do
  if printenv "$stride_key" >/dev/null; then
    export "APPTAINERENV_${stride_key}=${!stride_key}"
  fi
done
stride_options=()
if [[ "$stride_device_mode" == gpu || "$stride_device_mode" == gpu-parallel ]]; then
  stride_options+=(--rocm)
  export APPTAINERENV_STRIDE_RUNTIME=rocm
  for stride_key in ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL; do
    if printenv "$stride_key" >/dev/null; then
      export "APPTAINERENV_${stride_key}=${!stride_key}"
    fi
  done
elif [[ "$stride_device_mode" == cpu ]]; then
  export APPTAINERENV_ROCR_VISIBLE_DEVICES=-1
  export APPTAINERENV_HIP_VISIBLE_DEVICES=-1
  export APPTAINERENV_CUDA_VISIBLE_DEVICES=-1
else
  exit 2
fi
export APPTAINERENV_PYTHONNOUSERSITE=1
export APPTAINERENV_PYTHONDONTWRITEBYTECODE=1
export APPTAINERENV_PYTHONUNBUFFERED=1
export APPTAINERENV_OMP_NUM_THREADS=4
export APPTAINERENV_MKL_NUM_THREADS=4
export APPTAINERENV_OPENBLAS_NUM_THREADS=1
export APPTAINERENV_TOKENIZERS_PARALLELISM=false
export APPTAINERENV_HF_HOME="$stride_root/cache/huggingface"
export APPTAINERENV_HF_HUB_OFFLINE=1
export APPTAINERENV_HF_DATASETS_OFFLINE=1
export APPTAINERENV_TRANSFORMERS_OFFLINE=1
export APPTAINERENV_HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export APPTAINERENV_HF_HUB_DISABLE_TELEMETRY=1
export APPTAINERENV_WANDB_DISABLE_CODE=true
export APPTAINERENV_VLLM_NO_USAGE_STATS=1
export APPTAINERENV_VLLM_ROCM_USE_AITER=0
export APPTAINERENV_VLLM_ROCM_USE_AITER_RMSNORM=0
export APPTAINERENV_VLLM_ROCM_USE_AITER_TRITON_GEMM=0
export APPTAINERENV_VLLM_WORKER_MULTIPROC_METHOD=spawn
export APPTAINERENV_NCCL_DEBUG=WARN
export APPTAINERENV_XDG_CACHE_HOME="$stride_root/cache/xdg"
export APPTAINERENV_TRITON_CACHE_DIR="$stride_root/cache/triton/$SLURM_JOB_ID"
export APPTAINERENV_TORCHINDUCTOR_CACHE_DIR="$stride_root/cache/inductor/$SLURM_JOB_ID"
export APPTAINERENV_VLLM_CACHE_ROOT="$stride_root/cache/vllm/$SLURM_JOB_ID"
export APPTAINERENV_TMPDIR="/tmp/system-stride-full-${SLURM_JOB_ID}"
mkdir -p "$APPTAINERENV_TMPDIR"
cd "$stride_code"
if [[ "$stride_device_mode" == gpu ]]; then
  set -- "$stride_code/camd/allocated_gpu.py" -- "$@"
fi
exec apptainer exec --cleanenv "${stride_options[@]}" "$stride_image" "$stride_python" -u "$@"
