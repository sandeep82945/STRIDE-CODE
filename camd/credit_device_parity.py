"""Bounded synthetic CPU/ROCm parity check for the unchanged four-arm credits.

Create the CPU reference with a LAPACK-enabled PyTorch installation. Run check
inside the allocated ROCm container before training. This proves numerical
device parity on two explicit fixtures, not semantic reward quality or learning.
Every invocation writes an explicit ``passed`` JSON because the CAMD image can
incorrectly turn Python exceptions into process exit code zero at shutdown.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

SOURCE_FILES = (
    "training/credit.py", "training/production_stride.py",
    "training/experiment_credit.py", "training/feature_pooling.py",
    "training/runtime_config.py", "camd/credit_device_parity.py",
)
ATOL = 2e-4
RTOL = 2e-4
# The float32 Gram matrices have an explicit 1e-3 ridge. Different LAPACK and
# rocSOLVER reductions can accumulate rounding error; this tolerance is below
# one quarter of the ridge and well below the fixture's required >0.01 arm
# separation. Padding, masks, gating, solver devices, and zero GRPO bonus are
# checked exactly rather than weakened by this numerical tolerance.


def source_hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in SOURCE_FILES}


def fixture(name, device):
    """CPU reference features plus target-device mask, like annotate_groups."""
    import torch
    from feature_pooling import pool_reference_hidden

    eye = torch.eye(12, dtype=torch.float32)
    vectors = [
        [eye[0], eye[1], eye[2]],
        [eye[0], eye[3], eye[4]],
        [.8 * eye[1] + .2 * eye[5], eye[6], .3 * eye[2] + eye[7]],
        [eye[8], eye[0], .5 * eye[3] + .6 * eye[9]],
        [eye[0], eye[1], eye[2]],
        [eye[8], eye[9], eye[10]],
        [eye[4], eye[5], eye[6]],
        [eye[10], eye[0], eye[11]],
    ]
    lengths = [12, 11, 10, 12, 9, 12, 10, 11]
    spans = [[(1, 2), (4, 6), (8, 8)] for _ in vectors]
    whole, steps, pooled = [], [], []
    for i, (length, response_vectors) in enumerate(zip(lengths, vectors)):
        hidden = (.125 * eye[(i + 6) % 12]).repeat(length, 1)
        for j, (a, b) in enumerate(spans[i]):
            hidden[a:b + 1] = response_vectors[j] * (1. + .25 * j)
        w, s, p = pool_reference_hidden(hidden, spans[i], include_pooled=True)
        whole.append(w)
        steps.append(s)
        pooled.append(p)
    if name == "mixed_correctness_and_mapping":
        correct = [1, 1, 1, 1, 0, 1, 0, 1]
        eligible = [True, True, True, True, True, False, True, True]
    elif name == "all_correct":
        correct = [1] * 8
        eligible = [True] * 8
    else:
        raise ValueError(f"Unknown fixture {name}")
    mask = (torch.arange(max(lengths), device=device)[None, :]
            < torch.tensor(lengths, device=device)[:, None])
    return dict(answer_correct=correct, step_embeddings=steps,
                step_spans=spans, step_eligible=eligible,
                whole_embeddings=torch.stack(whole), response_mask=mask,
                pooled_step_embeddings=torch.stack(pooled), alpha=1.)


@contextmanager
def record_solvers():
    """Observe the actual solver calls; delegate their math unchanged."""
    import torch
    calls = []
    originals = {name: getattr(torch.linalg, name)
                 for name in ("cholesky", "slogdet")}

    def wrapped(name):
        def invoke(matrix, *args, **kwargs):
            calls.append(dict(operation=name, device=str(matrix.device),
                              dtype=str(matrix.dtype), shape=list(matrix.shape)))
            return originals[name](matrix, *args, **kwargs)
        return invoke

    try:
        for name in originals:
            setattr(torch.linalg, name, wrapped(name))
        yield calls
    finally:
        for name, function in originals.items():
            setattr(torch.linalg, name, function)


def run_case(name, device):
    import torch
    from experiment_credit import compute_group_credit
    from runtime_config import ARMS

    inputs = fixture(name, device)
    with record_solvers() as solver_calls:
        output = compute_group_credit(**inputs)
    expected_device = str(inputs["response_mask"].device)
    expected_arms = {"grpo", "local_positive", "single_step", "token_uniform"}
    if set(ARMS) != expected_arms or set(output["advantages"]) != expected_arms:
        raise AssertionError("Expected the release's four comparison arms")
    if not solver_calls or {c["operation"] for c in solver_calls} != {"cholesky", "slogdet"}:
        raise AssertionError("Fixture did not exercise both required solvers")
    if any(c["device"] != expected_device or c["dtype"] != "torch.float32"
           for c in solver_calls):
        raise AssertionError("Credit linear algebra did not stay on the requested float32 device")
    tensors = {"main": output["main"]}
    tensors.update({f"auxiliary/{arm}": output["auxiliary"][arm] for arm in ARMS})
    tensors.update({f"advantages/{arm}": output["advantages"][arm] for arm in ARMS})
    mask, mapped = inputs["response_mask"], output["mapped_mask"]
    for key, tensor in tensors.items():
        if tensor.device != mask.device or tensor.dtype != torch.float32:
            raise AssertionError(f"Wrong output device or dtype: {key}")
        if not bool(torch.isfinite(tensor).all()) or bool(tensor[~mask].any()):
            raise AssertionError(f"Nonfinite output or credit on padding: {key}")
    for i, (correct, eligible) in enumerate(zip(inputs["answer_correct"], inputs["step_eligible"])):
        if not (correct and eligible) and bool(mapped[i].any()):
            raise AssertionError("Incorrect or ineligible response received mapped credit")
    if bool(output["auxiliary"]["grpo"].any()):
        raise AssertionError("GRPO must have exactly zero auxiliary credit")
    positive_counts = {}
    for arm in ("local_positive", "single_step", "token_uniform"):
        auxiliary = output["auxiliary"][arm]
        positive_counts[arm] = int((auxiliary > 0).sum())
        if not positive_counts[arm] or bool((auxiliary < 0).any()) or bool(auxiliary[~mapped].any()):
            raise AssertionError(f"Fixture has invalid or inactive {arm} credit")
    local = output["auxiliary"]["local_positive"]
    uniform = output["auxiliary"]["token_uniform"]
    single = output["auxiliary"]["single_step"]
    torch.testing.assert_close(local.double().sum(1), uniform.double().sum(1),
                               atol=2e-6, rtol=2e-6)
    separations = dict(local_vs_uniform=float((local - uniform).abs().max()),
                       local_vs_single=float((local - single).abs().max()),
                       single_vs_uniform=float((single - uniform).abs().max()))
    if any(value <= .01 for value in separations.values()):
        raise AssertionError(f"Fixture does not distinguish the STRIDE arms: {separations}")
    if name == "all_correct" and bool(output["main"].any()):
        raise AssertionError("All-correct GRPO group must have zero main advantage")
    if name != "all_correct" and not bool(output["main"].any()):
        raise AssertionError("Mixed-correctness fixture must have a GRPO signal")
    return dict(tensors={key: value.cpu().tolist() for key, value in tensors.items()},
                mapped_mask=mapped.cpu().tolist(),
                diagnostics=output["diagnostics"],
                positive_tokens=positive_counts, arm_separation=separations,
                solver_calls=solver_calls, output_device=expected_device,
                input_feature_device="cpu", passed=True)


def compare_case(name, actual, expected):
    import torch
    if actual["mapped_mask"] != expected["mapped_mask"]:
        raise AssertionError(f"CPU/device mapped mask mismatch: {name}")
    errors = {}
    if set(actual["tensors"]) != set(expected["tensors"]):
        raise AssertionError("Reference tensor keys changed")
    for key, value in actual["tensors"].items():
        observed = torch.tensor(value, dtype=torch.float32)
        reference = torch.tensor(expected["tensors"][key], dtype=torch.float32)
        torch.testing.assert_close(observed, reference, atol=ATOL, rtol=RTOL,
                                   msg=f"CPU/device credit mismatch in {name}/{key}")
        errors[key] = float((observed - reference).abs().max())
    return errors


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("reference", "check"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to reuse existing evidence path: {args.output}")
    report = dict(passed=False, mode=args.mode, synthetic_fixture_only=True,
                  created_at=datetime.now(timezone.utc).isoformat(),
                  source_hashes=source_hashes(), python=sys.version,
                  atol=ATOL, rtol=RTOL)
    # Write failure first, so missing/truncated execution cannot look successful.
    write_json(args.output, report)
    try:
        import torch
        report.update(torch_version=torch.__version__, hip_version=torch.version.hip)
        device = torch.device("cpu" if args.mode == "reference" else args.device)
        reference = None
        if args.mode == "check":
            if not args.reference:
                raise ValueError("GPU check requires --reference")
            reference = json.loads(args.reference.read_text())
            if (reference.get("passed") is not True or reference.get("mode") != "reference"
                    or reference.get("source_hashes") != source_hashes()):
                raise ValueError("CPU reference failed, or relevant source hashes differ")
            if device.type != "cuda" or not torch.version.hip or not torch.cuda.is_available():
                raise ValueError("GPU check requires an allocated ROCm device")
            torch.cuda.set_device(device)
            device = torch.device("cuda", torch.cuda.current_device())
            properties = torch.cuda.get_device_properties(device)
            report.update(device=str(device), gpu_name=properties.name,
                          gcn_arch=getattr(properties, "gcnArchName", None),
                          reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest())
        elif not torch._C.has_lapack:
            raise ValueError("Reference must use a LAPACK-enabled CPU PyTorch build")
        cases = {}
        for name in ("mixed_correctness_and_mapping", "all_correct"):
            result = run_case(name, device)
            if reference:
                result["cpu_max_abs_error"] = compare_case(name, result, reference["cases"][name])
            cases[name] = result
        if source_hashes() != report["source_hashes"]:
            raise RuntimeError("Credit sources changed during the probe")
        report.update(cases=cases, passed=True)
    except BaseException as error:
        report.update(error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc(), passed=False)
        raise
    finally:
        write_json(args.output, report)
        print(json.dumps(dict(passed=report["passed"], output=str(args.output),
                              mode=args.mode)), flush=True)


if __name__ == "__main__":
    main()
