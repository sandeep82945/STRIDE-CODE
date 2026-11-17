"""Check the selected runtime and allocated GPUs before a model smoke test."""
from __future__ import annotations
import argparse
from importlib.metadata import version
import json
import os
import sys

PACKAGES = {"torch": "2.10.0", "vllm": "0.17.0", "transformers": "4.57.6",
            "peft": "0.18.1", "huggingface-hub": "0.36.0", "pyarrow": "21.0.0"}
ROCM_PACKAGES = {**PACKAGES, "vllm": "0.20.1"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=int, default=8, choices=[1, 8])
    parser.add_argument("--runtime", choices=["nvidia", "rocm"],
                        default=os.environ.get("STRIDE_RUNTIME", "nvidia"))
    args = parser.parse_args()
    report = {"python": sys.version.split()[0], "expected_gpus": args.gpus,
              "versions": {}, "checks": {}, "model_training_tested": False,
              "runtime": args.runtime}
    try:
        packages = ROCM_PACKAGES if args.runtime == "rocm" else PACKAGES
        for name, expected in packages.items():
            actual = version(name)
            report["versions"][name] = actual
            if actual.split("+", 1)[0] != expected:
                raise ValueError(f"Expected {name}=={expected}, found {actual}; use the supplied environment")
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import LoraConfig, PeftModel
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
        from vllm.config.compilation import CompilationConfig, CompilationMode, CUDAGraphMode
        report["checks"]["required_imports"] = True
        report["cuda"] = torch.version.cuda
        report["hip"] = torch.version.hip
        report["visibility"] = {name: os.environ.get(name) for name in
                                ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")}
        if args.runtime == "rocm":
            if not torch.version.hip or not torch.cuda.is_available():
                raise ValueError("Expected usable AMD ROCm GPUs for the ROCm runtime profile")
            allocated = os.environ.get("ROCR_VISIBLE_DEVICES", "").split(",")
            logical = ",".join(str(i) for i in range(args.gpus))
            if (len(allocated) != args.gpus or not all(allocated)
                    or len(set(allocated)) != args.gpus
                    or os.environ.get("HIP_VISIBLE_DEVICES") != logical
                    or os.environ.get("CUDA_VISIBLE_DEVICES") != logical):
                raise ValueError("ROCm probe requires intact ROCR allocation and normalized CUDA/HIP logical indices")
        elif torch.version.hip or not torch.cuda.is_available():
            raise ValueError("Expected NVIDIA CUDA; this package is not a validated ROCm port")
        if torch.cuda.device_count() != args.gpus:
            raise ValueError(f"Expected {args.gpus} allocated visible GPUs, got {torch.cuda.device_count()}")
        report["devices"] = []
        for index in range(args.gpus):
            device = f"cuda:{index}"
            x = torch.arange(64, dtype=torch.float32, device=device).reshape(8, 8)
            y = x.to(torch.bfloat16) @ x.to(torch.bfloat16).T
            cholesky = torch.linalg.cholesky(torch.eye(4, device=device) * 2)
            torch.cuda.synchronize(index)
            if not bool(torch.isfinite(y).all() and torch.isfinite(cholesky).all()):
                raise ValueError(f"Nonfinite BF16 or Cholesky output on {device}")
            p = torch.cuda.get_device_properties(index)
            report["devices"].append({"logical_index": index, "name": p.name,
                                      "memory_gib": round(p.total_memory / 2**30, 2),
                                      "gcn_arch": getattr(p, "gcnArchName", None)})
        report["checks"]["gpu_primitives"] = True
        report["passed"] = True
    except Exception as exc:
        report["passed"] = False
        report["error"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
