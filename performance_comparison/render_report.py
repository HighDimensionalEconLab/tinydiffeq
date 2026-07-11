"""Render CURRENT_RESULTS.md from the four canonical JSON result files."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUTPUT = ROOT / "CURRENT_RESULTS.md"
FILES = {
    "python_cpu_rodas": RESULTS / "python_cpu_rodas.json",
    "python_gpu_rodas": RESULTS / "python_gpu_rodas.json",
    "julia_cpu_rodas": RESULTS / "julia_cpu_rodas.json",
    "python_cpu_dae": RESULTS / "python_cpu_dae.json",
    "python_gpu_dae": RESULTS / "python_gpu_dae.json",
    "julia_cpu_dae": RESULTS / "julia_cpu_dae.json",
    "python_cpu_sde_large": RESULTS / "python_cpu_sde_large.json",
    "python_gpu_sde_large": RESULTS / "python_gpu_sde_large.json",
    "julia_cpu_sde_large": RESULTS / "julia_cpu_sde_large.json",
    "julia_cpu_sdae": RESULTS / "julia_cpu_sdae.json",
    "julia_cpu_vjp": RESULTS / "julia_cpu_vjp.json",
    # Canonical run.sh outputs come last so a rerun supersedes supplemental
    # slices from this initial measurement session.
    "python_cpu": RESULTS / "python_cpu.json",
    "python_gpu": RESULTS / "python_gpu.json",
    "julia_cpu": RESULTS / "julia_cpu.json",
    "julia_gpu": RESULTS / "julia_gpu.json",
}
LIBRARIES = ["tinydiffeq", "diffrax", "sciml"]


def load_results():
    payloads = {}
    for name, path in FILES.items():
        if path.exists():
            payloads[name] = json.loads(path.read_text())
    return payloads


def row_key(result):
    case = result["case"]
    return (
        case["equation"],
        case["size"],
        case["method"],
        case["stepping"],
        case["controller"],
        case["dtype"],
        case["transform"],
        case["batch"],
        result.get("tolerance_name", "common"),
    )


def format_case(key):
    equation, size, method, stepping, controller, dtype, transform, batch, tolerance = (
        key
    )
    label = f"{equation.replace('_', ' ')} n={size}"
    if "ensemble" in equation:
        label += f" B={batch}"
    policy = stepping if controller == "none" else f"{stepping}/{controller}"
    return f"{label}; {method}; {policy}; {dtype}; {transform}; {tolerance}"


def format_cell(record, baseline):
    if record is None:
        return "—"
    microseconds = record["median_seconds"] * 1e6
    iqr = record["iqr_seconds"] * 1e6
    ratio = ""
    if baseline is not None and record["library"] != "tinydiffeq":
        ratio = f"; {record['median_seconds'] / baseline['median_seconds']:.2f}×"
    step_text = ""
    if record["case"]["stepping"] == "adaptive":
        accepted = record.get("stats", {}).get("accepted_steps")
        rejected = record.get("stats", {}).get("rejected_steps")
        if accepted is not None:
            step_text = f"; {accepted} accepted"
            if rejected is not None:
                step_text += f"/{rejected} rejected"
    return f"{microseconds:,.2f} µs (IQR {iqr:,.2f}{ratio}{step_text})"


def table_for(payloads, backend, predicate):
    rows = defaultdict(dict)
    for name, payload in payloads.items():
        payload_backend = payload["metadata"]["backend"]
        if backend == "gpu":
            matches_backend = payload_backend in {"gpu", "cuda"}
        else:
            matches_backend = payload_backend == "cpu"
        if not matches_backend:
            continue
        for result in payload["results"]:
            if (
                result["library"] == "sciml"
                and result["case"]["transform"] == "vjp"
                and name != "julia_cpu_vjp"
            ):
                continue
            if predicate(result):
                rows[row_key(result)][result["library"]] = result
    lines = [
        "| Case | tinydiffeq | Diffrax | SciML |",
        "|---|---:|---:|---:|",
    ]
    for key in sorted(rows):
        records = rows[key]
        baseline = records.get("tinydiffeq")
        cells = [format_cell(records.get(library), baseline) for library in LIBRARIES]
        lines.append(f"| {format_case(key)} | " + " | ".join(cells) + " |")
    if len(lines) == 2:
        lines.append("| — | — | — | — |")
    return "\n".join(lines)


def metadata_table(payloads):
    lines = ["| Result file | Runtime metadata |", "|---|---|"]
    for name, payload in payloads.items():
        metadata = payload["metadata"]
        summary = ", ".join(
            f"{key}={value}"
            for key, value in metadata.items()
            if key not in {"environment"}
        )
        lines.append(f"| `{FILES[name].name}` | {summary} |")
    return "\n".join(lines)


def controller_table(payloads):
    checks = []
    for payload in payloads.values():
        for check in payload.get("controller_equivalence", []):
            item = (check["dtype"], check["controller"])
            if item not in [(row["dtype"], row["controller"]) for row in checks]:
                checks.append(check)
    lines = [
        "| Dtype | Controller | Exact mesh match | "
        "Accepted steps (tiny/Diffrax) | Max mesh difference |",
        "|---|---|---:|---:|---:|",
    ]
    for check in checks:
        difference = check["max_mesh_difference"]
        difference_text = "—" if difference is None else f"{difference:.6g}"
        lines.append(
            f"| {check['dtype']} | {check['controller']} | "
            f"{'yes' if check['equivalent'] else 'no'} | "
            f"{check['tinydiffeq_accepted']}/{check['diffrax_accepted']} | "
            f"{difference_text} |"
        )
    return "\n".join(lines)


def unavailable_summary(payloads):
    reasons = defaultdict(set)
    for payload in payloads.values():
        for entry in payload.get("unavailable", []):
            case = entry["case"]
            label = (
                f"{entry['library']} {case['equation']} "
                f"{case['method']} {case['transform']}"
            )
            reasons[label].add(entry["reason"].split("\n", 1)[0])
    if not reasons:
        return "No unavailable cells were recorded."
    return "\n".join(
        f"- `{label}`: {'; '.join(sorted(messages))}"
        for label, messages in sorted(reasons.items())
    )


def validation_table():
    lines = [
        "| Backend | ODE checks | DAE checks | Stochastic checks |",
        "|---|---:|---:|---:|",
    ]
    for backend in ["cpu", "gpu"]:
        path = RESULTS / f"validation_{backend}.json"
        if not path.exists():
            lines.append(f"| {backend} | — | — | — |")
            continue
        payload = json.loads(path.read_text())
        lines.append(
            f"| {backend} | {len(payload['ode'])} passed | "
            f"{len(payload['dae'])} passed | "
            f"{len(payload['stochastic'])} passed |"
        )
    return "\n".join(lines)


def main():
    payloads = load_results()

    def fixed(result):
        return result["case"]["stepping"] == "fixed"

    def adaptive(result):
        return result["case"]["stepping"] == "adaptive" and result["case"][
            "equation"
        ].startswith("ode")

    def dae(result):
        return result["case"]["equation"].startswith("dae")

    def stochastic(result):
        return "ensemble" in result["case"]["equation"]

    def fixed_ode(result):
        return fixed(result) and result["case"]["equation"].startswith("ode")

    text = f"""# Current tinydiffeq performance comparison

This snapshot reports median post-compilation endpoint latency. JAX calls synchronize
the complete result on every timed invocation; Julia calls are warmed before
`BenchmarkTools` sampling. Ratios in another library's cell are relative to
tinydiffeq only when both occupy the same row. An em dash means that combination was
not available or not measured.

The targeted [ODE loop optimization follow-up](OPTIMIZATION_RESULTS.md)
supersedes the affected fixed RK4/Tsit5 CPU and adaptive Tsit5 GPU rows below.

## Environment

{metadata_table(payloads)}

The CPU runs use one Julia thread and one BLAS thread. GPU results use the local RTX
3090. Compilation and first execution are excluded from the main timing cells but
remain in the raw JSON. The Rodas5P supplemental files were measured from the current
working tree on top of the recorded base commit; future runs record the dirty-worktree
flag explicitly.

All recorded Julia call wrappers had a concrete inferred return type. The validation
pass checks deterministic endpoints and JVPs across the JAX libraries, DAE constraint
residuals plus JVP/VJP finiteness, stochastic replay, independent paths, and SDE/SDAE
pathwise derivatives:

{validation_table()}

JAX VJPs differentiate the discrete solve. SciML VJP cells are shown only when
`ReverseDiffAdjoint` succeeds, so they have the same discrete-sensitivity semantics;
its default continuous-adjoint timings are deliberately excluded.

## Hyperparameter equivalence

Fixed rows use the identical interval, endpoint-only output, method, dtype, and step
count: scalar ODE 64 steps on `[0, 10]`, 256-state ODE 128 steps on `[0, 1]`, DAE
64 steps on `[0, 1]`, and SDE/SDAE ensembles 128 steps on `[0, 1]`. JAX adaptive
rows share `dt_0`, tolerance, attempt budget, safety 0.9, factor limits `[0.2, 5]`,
and max-norm scaling. SciML adaptive rows share `dt_0`, tolerance, and budget but
retain SciML's native controller.

{controller_table(payloads)}

Because the adaptive meshes do not match exactly, every adaptive result below is a
**same-tolerance native-controller comparison**, not an exact hyperparameter match.
Float32 and Float64 both use the common `rtol=1e-4`, `atol=1e-6` row; Float64 also
has a precision row at `rtol=1e-7`, `atol=1e-9`.

## CPU: exact fixed-step ODE methods

{table_for(payloads, "cpu", fixed_ode)}

## CPU: adaptive Tsit5 with native controller behavior

{table_for(payloads, "cpu", adaptive)}

## CPU: DAE exact and capability rows

Fixed `Rodas5P` rows compare the same published method, equations, interval, dtype,
step count, and endpoint-only output against SciML's original
[`OrdinaryDiffEqRosenbrock`](https://github.com/SciML/OrdinaryDiffEq.jl/tree/master/lib/OrdinaryDiffEqRosenbrock)
implementation. Tsit5/RK4 rows use tinydiffeq's nonlinear root-restoring formulation;
adaptive Rodas5P rows use each implementation's native controller and are
same-tolerance rather than same-mesh comparisons. The DAE initial values are already
consistent; tinydiffeq still performs its documented LM consistency solve, while
SciML uses its native consistent-initialization policy.

{table_for(payloads, "cpu", dae)}

## CPU: SDE and SDAE ensembles

JAX ensemble rows compile a `vmap` over independent keys. SciML CPU uses
`EnsembleSerial` to keep the matched resource count at one thread.

{table_for(payloads, "cpu", stochastic)}

## GPU results

{table_for(payloads, "gpu", lambda result: True)}

## Empty-cell reasons

{unavailable_summary(payloads)}

## Reproduction

Run `./run.sh --quick` for the current representative matrix or `./run.sh --full` for
all configured system sizes and ensemble sizes. Raw measurements are under `results/`.
The Python and Julia projects are isolated below this directory and do not alter the
tinydiffeq dependency graph.
"""
    OUTPUT.write_text(text)


if __name__ == "__main__":
    main()
