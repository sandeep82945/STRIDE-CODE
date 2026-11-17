#!/usr/bin/env python3
"""Allocated-GPU Qwen4B LoRA parity/memory/timing gate for paired learners.

No optimizer steps, model checkpoints, network calls or outside-Slurm execution.
Isolated workers release GPU memory between variants. Temporary CPU gradient
arrays support a real gradient comparison and are removed after aggregation.
"""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time

CODE = Path(__file__).resolve().parents[1]
CASES = {'unequal_8192_6144': (8192, 6144), 'full_8192_8192': (8192, 8192)}
TOLERANCES = dict(logp_mean_abs=.02, logp_max_abs=.2, gradient_relative_l2=.02,
                  gradient_cosine_min=.999, memory_fraction_max=.9, speedup_min=1.05)


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.writing')
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
    temporary.replace(path)


def require_allocation():
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_STEP_ID'):
        raise RuntimeError('This GPU probe requires an active Slurm job and srun step')


def parent_environment():
    require_allocation()
    sys.path.insert(0, str(CODE))
    import run as launcher
    # Validate the parent eight-device allocation before narrowing logical HIP.
    env = launcher.execution_environment('train', seed=42)
    original_rocr = env.get('ROCR_VISIBLE_DEVICES')
    if env.get('STRIDE_RUNTIME') != 'rocm' or not original_rocr:
        raise RuntimeError('This probe requires the validated ROCm allocation')
    env['CUDA_VISIBLE_DEVICES'] = env['HIP_VISIBLE_DEVICES'] = '0'
    env['PYTHONNOUSERSITE'] = '1'
    env['TOKENIZERS_PARALLELISM'] = 'false'
    # Keep the physical Slurm ROCR mask exactly intact. HIP selects its index 0.
    if env['ROCR_VISIBLE_DEVICES'] != original_rocr:
        raise RuntimeError('Physical allocation mask unexpectedly changed')
    return env


def sequences(tokenizer, lengths):
    prompts = [tokenizer.encode('Solve the arithmetic problem carefully. What is 1 + 1?\n', add_special_tokens=False),
               tokenizer.encode('Prove the arithmetic identity carefully. Show that 2 + 2 = 4.\n', add_special_tokens=False)]
    units = [tokenizer.encode('Step 1: Add one to one to obtain two. Step 2: Check that two minus one equals one.\n', add_special_tokens=False),
             tokenizer.encode('Step 1: Combine two equal pairs. Step 2: Count four objects and verify the sum.\n', add_special_tokens=False)]
    result = []
    for prompt, unit, length in zip(prompts, units, lengths):
        count = length - len(prompt)
        if not prompt or not unit or count < 1:
            raise ValueError('Tokenizer did not produce a valid causal test sequence')
        response = (unit * ((count + len(unit) - 1) // len(unit)))[:count]
        result.append((prompt, response))
    return result


def worker(model_path, variant, artifact_dir):
    require_allocation()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0' or os.environ.get('HIP_VISIBLE_DEVICES') != '0':
        raise RuntimeError('Probe worker must select exactly logical allocated GPU zero')
    # Import Torch only after all masks were established by the parent.
    sys.path.insert(0, str(CODE / 'training'))
    import numpy as np
    import torch
    from transformers import AutoTokenizer
    import trainer as pilot
    import batched_learner
    from stabilization import reference_logps, sampled_k3_kl
    report = dict(variant=variant, passed=False, cases={}, optimizer_steps=0,
                  model_checkpoints_written=0, seed=42,
                  objective='same signed weighted response-token log likelihood plus fixed-base sampled KL',
                  kl_coefficient=.01,
                  lora_b_initialization='deterministic normal std=0.001 to exercise nonzero A/B gradients')
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError('Probe worker does not see exactly one usable GPU')
        torch.cuda.set_device(0)
        torch.manual_seed(42)
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
        actor = pilot.load_actor(model_path, device='cuda', finetune_mode='lora')
        actor.train()
        with torch.no_grad():
            for name, parameter in actor.named_parameters():
                if parameter.requires_grad and 'lora_B' in name:
                    torch.nn.init.normal_(parameter, mean=0., std=.001)
        trainable = [(name, p) for name, p in actor.named_parameters() if p.requires_grad]
        if not trainable or not actor.is_gradient_checkpointing:
            raise RuntimeError('Probe requires real LoRA parameters and gradient checkpointing')
        properties = torch.cuda.get_device_properties(0)
        report.update(gpu_name=properties.name, gpu_total_bytes=int(properties.total_memory),
                      gpu_uuid=str(getattr(properties, 'uuid', 'unavailable')),
                      gradient_checkpointing=True, trainable_parameters=sum(p.numel() for _, p in trainable))
        for case, lengths in CASES.items():
            pairs = sequences(tokenizer, lengths)
            denominator = sum(len(response) for _, response in pairs)
            elapsed, peaks, reserved, retained_logps = [], [], [], None
            for repeat in range(3):  # Warm once, then compare two warm timings.
                actor.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(0)
                torch.cuda.synchronize(0)
                began = time.perf_counter()
                chunks = [[pair] for pair in pairs] if variant == 'legacy' else [pairs]
                captured = []
                kl_total = 0.
                for chunk in chunks:
                    references = reference_logps(actor, chunk)
                    values = ([pilot.selected_logps(actor, *chunk[0])] if variant == 'legacy'
                              else batched_learner.selected_logps(actor, chunk))
                    weighted = []
                    for value, reference in zip(values, references):
                        # Identical token weights for each sequence and variant.
                        positions = torch.arange(len(value), device=value.device)
                        weights = torch.where(positions.remainder(3) == 0, -.5, 1.).to(value.dtype)
                        kl = sampled_k3_kl(value, reference).sum()
                        kl_total += float(kl.detach())
                        weighted.append((-(value * weights).sum() + .01 * kl) / denominator)
                        captured.append(value.detach().cpu().numpy())
                    sum(weighted).backward()
                    del values, weighted, value, positions, weights, reference, references, kl
                torch.cuda.synchronize(0)
                seconds = time.perf_counter() - began
                peak = int(torch.cuda.max_memory_allocated(0))
                peak_reserved = int(torch.cuda.max_memory_reserved(0))
                if repeat:
                    elapsed.append(seconds)
                    peaks.append(peak)
                    reserved.append(peak_reserved)
                retained_logps = captured
            arrays = {f'logps_{i}': values for i, values in enumerate(retained_logps)}
            missing = [name for name, p in trainable if p.grad is None]
            if missing:
                raise RuntimeError('Some trainable LoRA parameters have no gradient')
            arrays.update({f'grad::{name}': p.grad.detach().float().cpu().numpy() for name, p in trainable})
            finite = all(bool(np.isfinite(array).all()) for array in arrays.values())
            gradient_norm = math.sqrt(sum(float(np.square(array.astype(np.float64)).sum())
                                          for name, array in arrays.items() if name.startswith('grad::')))
            np.savez(artifact_dir / f'{variant}-{case}.npz', **arrays)
            report['cases'][case] = dict(total_sequence_tokens=list(lengths), response_tokens=denominator,
                warmup_repeats=1, timed_repeats=2, seconds=elapsed, median_seconds=statistics.median(elapsed),
                peak_allocated_bytes=max(peaks), peak_reserved_bytes=max(reserved),
                finite=finite, gradient_l2=gradient_norm, nonzero_gradient=gradient_norm > 0,
                sampled_kl_mean=kl_total / denominator, kl_coefficient=.01)
            if not math.isfinite(kl_total) or kl_total <= 0:
                raise RuntimeError('Perturbed LoRA must produce finite nonzero fixed-base KL')
            if not finite or not math.isfinite(gradient_norm) or gradient_norm <= 0:
                raise RuntimeError('Probe produced nonfinite or zero training gradients')
            del arrays, captured, retained_logps
            actor.zero_grad(set_to_none=True)
        report['passed'] = True
    except BaseException as exc:
        report['error_type'] = type(exc).__name__
        report['out_of_memory'] = isinstance(exc, torch.OutOfMemoryError)
        # Fixed context is sufficient; avoid dumping large/native exception text.
        report['error'] = 'GPU probe variant did not complete; inspect its local worker log'
        raise
    finally:
        write(artifact_dir / f'{variant}.json', report)


def compare(artifact_dir, reports):
    import numpy as np
    cases = {}
    for case in CASES:
        with np.load(artifact_dir / f'legacy-{case}.npz', allow_pickle=False) as legacy, \
             np.load(artifact_dir / f'batched-{case}.npz', allow_pickle=False) as paired:
            if set(legacy.files) != set(paired.files):
                raise ValueError('Gradient/log-probability inventories differ')
            diff2 = ref2 = paired2 = dot = grad_max = ref_max = log_sum = log_max = 0.
            log_count = 0
            for key in legacy.files:
                left, right = legacy[key].astype(np.float64), paired[key].astype(np.float64)
                if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
                    raise ValueError('Nonfinite or mismatched gradient/log-probability arrays')
                delta = left - right
                if key.startswith('grad::'):
                    diff2 += float(np.square(delta).sum())
                    ref2 += float(np.square(left).sum())
                    paired2 += float(np.square(right).sum())
                    dot += float((left * right).sum())
                    grad_max = max(grad_max, float(np.abs(delta).max()))
                    ref_max = max(ref_max, float(np.abs(left).max()))
                else:
                    log_sum += float(np.abs(delta).sum())
                    log_max = max(log_max, float(np.abs(delta).max()))
                    log_count += int(delta.size)
            relative = math.sqrt(diff2 / max(ref2, 1e-30))
            cosine = dot / max(math.sqrt(ref2 * paired2), 1e-30)
            log_mean = log_sum / max(log_count, 1)
        left, right = (reports[variant]['cases'][case] for variant in ('legacy', 'batched'))
        parity = (ref2 > 0 and paired2 > 0 and log_count > 0
                  and log_mean <= TOLERANCES['logp_mean_abs'] and log_max <= TOLERANCES['logp_max_abs']
                  and relative <= TOLERANCES['gradient_relative_l2'] and cosine >= TOLERANCES['gradient_cosine_min'])
        fraction = right['peak_allocated_bytes'] / reports['batched']['gpu_total_bytes']
        cases[case] = dict(logp_mean_abs=log_mean, logp_max_abs=log_max,
            gradient_relative_l2=relative, gradient_max_abs=grad_max,
            gradient_relative_max=grad_max / max(ref_max, 1e-30), gradient_cosine=cosine,
            parity_passed=parity, batched_memory_fraction=fraction,
            memory_passed=fraction < TOLERANCES['memory_fraction_max'],
            speedup=left['median_seconds'] / right['median_seconds'])
    speedup = (sum(reports['legacy']['cases'][case]['median_seconds'] for case in CASES)
               / sum(reports['batched']['cases'][case]['median_seconds'] for case in CASES))
    parity = all(case['parity_passed'] for case in cases.values())
    memory = all(case['memory_passed'] for case in cases.values())
    speed = speedup >= TOLERANCES['speedup_min'] and min(case['speedup'] for case in cases.values()) >= 1.
    return dict(cases=cases, parity_passed=parity, memory_passed=memory, speed_passed=speed,
                aggregate_speedup=speedup, recommended_learner_microbatch_size=2 if parity and memory and speed else 1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--worker', choices=('legacy', 'batched'), help=argparse.SUPPRESS)
    parser.add_argument('--artifact-dir', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        worker(str(args.model_path), args.worker, args.artifact_dir)
        return 0
    if args.output is None:
        parser.error('--output is required')
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError('Use a new probe report path')
    output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(passed=False, complete=False, recommended_learner_microbatch_size=1,
                  tolerances=TOLERANCES, variants={}, optimizer_steps=0, model_checkpoints_written=0,
                  limitations='Two synthetic long-sequence cases on one allocated GPU; not an end-to-end speed guarantee.')
    began = time.monotonic()
    try:
        env = parent_environment()
        result.update(job_id=os.environ['SLURM_JOB_ID'], node=os.environ.get('SLURMD_NODENAME'),
                      physical_rocr_mask=env['ROCR_VISIBLE_DEVICES'], logical_cuda_mask='0', logical_hip_mask='0')
        with tempfile.TemporaryDirectory(prefix='microbatch-probe-', dir=output.parent) as directory:
            artifacts = Path(directory)
            for variant in ('legacy', 'batched'):
                log_path = output.with_name(output.stem + f'-{variant}.log')
                with log_path.open('x') as log:
                    child = subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()),
                        '--model-path', str(args.model_path.resolve()), '--worker', variant,
                        '--artifact-dir', str(artifacts)], env=env, stdout=log, stderr=subprocess.STDOUT,
                        timeout=2400, check=False)
                path = artifacts / f'{variant}.json'
                report = json.loads(path.read_text()) if path.is_file() else dict(passed=False, missing_report=True)
                report.update(returncode=child.returncode, log_path=str(log_path))
                result['variants'][variant] = report
                if child.returncode != 0 or report.get('passed') is not True:
                    raise RuntimeError('Probe worker failed; retain learner microbatch size one')
            result.update(compare(artifacts, result['variants']))
            result['passed'] = result['parity_passed'] and result['memory_passed']
            result['complete'] = True
    except BaseException as exc:
        result.update(error_type=type(exc).__name__, error='Probe not approved; retain learner microbatch size one',
                      recommended_learner_microbatch_size=1)
    finally:
        result['elapsed_seconds'] = time.monotonic() - began
        write(output, result)
    print(json.dumps({key: result[key] for key in ('passed', 'complete', 'recommended_learner_microbatch_size', 'elapsed_seconds')}))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
