"""Render MARKOV_RESULTS.md from the canonical Markov benchmark JSON files."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUTPUT = ROOT / "MARKOV_RESULTS.md"


def load(name):
    path = RESULTS / name
    return json.loads(path.read_text()) if path.exists() else None


def records(payload):
    if payload is None:
        return {}
    return {
        tuple(
            result["case"][key]
            for key in ("kind", "states", "length", "batch", "dtype", "method")
        ): result
        for result in payload["results"]
    }


def microseconds(result):
    if result is None:
        return "—"
    return f"{result['median_seconds'] * 1e6:,.2f} µs"


def distribution_records(payload):
    if payload is None:
        return {}
    return {
        tuple(
            result["case"][key]
            for key in ("problem", "states", "points", "batch", "dtype", "method")
        ): result
        for result in payload["results"]
    }


def distribution_endpoint_table(payload):
    indexed = distribution_records(payload)
    lines = [
        "| Problem | States | Dtype | Dense/power | Matrix-free Krylov/scan |",
        "|---|---:|---|---:|---:|",
    ]
    for problem in ("dtmc_endpoint", "ctmc_endpoint"):
        for states in (8, 32, 128):
            for dtype in ("float32", "float64"):
                if problem == "dtmc_endpoint":
                    first_method, second_method, points = "power", "sequential", 1024
                    label = "DTMC endpoint"
                else:
                    first_method, second_method, points = "dense", "krylov_pytree", 1
                    label = "structured CTMC endpoint"
                prefix = (problem, states, points, 1, dtype)
                lines.append(
                    f"| {label} | {states} | {dtype} | "
                    f"{microseconds(indexed.get(prefix + (first_method,)))} | "
                    f"{microseconds(indexed.get(prefix + (second_method,)))} |"
                )
    return "\n".join(lines)


def distribution_path_table(payload):
    indexed = distribution_records(payload)
    lines = [
        "| Forecast | Dtype | Sequential/dense | Associative/Krylov |",
        "|---|---|---:|---:|",
    ]
    for dtype in ("float32", "float64"):
        dtmc = ("dtmc_path", 8, 1024, 256, dtype)
        ctmc = ("ctmc_grid", 32, 33, 1, dtype)
        lines.append(
            f"| DTMC full path, K=8, B=256 | {dtype} | "
            f"{microseconds(indexed.get(dtmc + ('sequential',)))} | "
            f"{microseconds(indexed.get(dtmc + ('associative',)))} |"
        )
        lines.append(
            f"| CTMC 33-point grid, K=32, B=1 | {dtype} | "
            f"{microseconds(indexed.get(ctmc + ('dense',)))} | "
            f"{microseconds(indexed.get(ctmc + ('krylov_pytree',)))} |"
        )
    return "\n".join(lines)


def backend_table(payload):
    indexed = records(payload)
    lines = [
        "| Process | Dtype | Batch | Sequential | Sequential unroll=16 | Associative |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for kind in ("discrete", "continuous"):
        for dtype in ("float32", "float64"):
            for batch in (256, 4096):
                prefix = (kind, 8, 1024, batch, dtype)
                cells = [
                    microseconds(indexed.get(prefix + (method,)))
                    for method in ("sequential", "sequential_unroll16", "associative")
                ]
                lines.append(
                    f"| {kind} | {dtype} | {batch} | " + " | ".join(cells) + " |"
                )
    return "\n".join(lines)


def sciml_table(python_cpu, julia_cpu):
    python = records(python_cpu)
    julia = records(julia_cpu)
    lines = [
        "| Dtype | tinydiffeq sequential | SciML Direct/SSAStepper | "
        "Ratio SciML/tiny |",
        "|---|---:|---:|---:|",
    ]
    for dtype in ("float32", "float64"):
        key = ("continuous", 2, 64, 256, dtype)
        tiny = python.get(key + ("sequential",))
        sciml = julia.get(key + ("sciml_direct_ssa",))
        ratio = (
            "—"
            if tiny is None or sciml is None
            else f"{sciml['median_seconds'] / tiny['median_seconds']:.2f}×"
        )
        lines.append(
            f"| {dtype} | {microseconds(tiny)} | {microseconds(sciml)} | {ratio} |"
        )
    return "\n".join(lines)


def preparation_table(payload):
    indexed = records(payload)
    lines = ["| Process | Dtype | Alias preparation |", "|---|---|---:|"]
    for kind in ("discrete", "continuous"):
        for dtype in ("float32", "float64"):
            values = [
                result["preparation_seconds"]
                for key, result in indexed.items()
                if key[0] == kind and key[1] == 8 and key[4] == dtype
            ]
            value = (
                "—"
                if not values
                else f"{1e6 * sorted(values)[len(values) // 2]:,.2f} µs"
            )
            lines.append(f"| {kind} | {dtype} | {value} |")
    return "\n".join(lines)


def main():
    python_cpu = load("python_cpu_markov.json")
    python_gpu = load("python_gpu_markov.json")
    distribution_cpu = load("python_cpu_markov_distribution.json")
    distribution_gpu = load("python_gpu_markov_distribution.json")
    julia_cpu = load("julia_cpu_markov.json")
    text = f"""# Finite-state Markov performance

These are median post-compilation primal path-simulation times. JAX calls
synchronize the full `(batch, path)` result. Alias-table preparation is excluded
from repeated timings. The local CPU benchmark uses one process; GPU results use
the RTX 3090. The main rows simulate 1,024 transitions or bounded jump attempts
for eight states and return every path row.

Sequential and associative methods consume the same uniform/exponential draws.
DTMC states match exactly. CTMC post-jump states match exactly, while associative
holding-time sums differ by floating-point reassociation.

## One-time preparation

{preparation_table(python_cpu)}

Preparation validates and normalizes the matrix and constructs the alias tables.
It is performed once on the host, outside transformed simulation and all repeated
latency cells below.

## CPU

{backend_table(python_cpu)}

CPU favors the default sequential scan. Unrolling and associative map composition
increase executable work and memory traffic.

## RTX 3090 GPU

{backend_table(python_gpu)}

The associative alias kernel exposes temporal parallelism and dominates the
medium-batch cases. For large CTMC float32 ensembles, unroll=16 can be competitive;
the explicit method choice is retained because the crossover depends on all four
of state count, path length, batch size, and dtype.

## SciML CPU capability comparison

{sciml_table(python_cpu, julia_cpu)}

This smaller common row is a capability comparison: two states, a 64-jump
budget/horizon scale, and 256 trajectories. tinydiffeq returns the complete
bounded 65-row paths and presamples randomness; SciML returns endpoints and
terminates chronologically at the horizon. SciML's
[`Direct` + `SSAStepper`](https://docs.sciml.ai/JumpProcesses/stable/jump_solve/)
terminates chronologically at the horizon. Julia is warmed, type-inferred, and
measured with `BenchmarkTools`; it uses one Julia and BLAS thread. Different RNGs
prevent path identity, so correctness is checked against analytical laws rather
than cross-library samples.

## Deterministic distribution forecasts

The CTMC benchmark is deliberately structured: each state transitions to its
neighbor on a ring. The dense baseline forms and exponentiates the full
generator; the pytree Krylov method evaluates the same forward equation with
an O(K) matrix-free flux action. All rows use the same generator, horizon,
initial distribution, dtype, and requested output. On CPU, maximum dense/Krylov
disagreement across the quick matrix was below `9e-8` in float32 and `1e-15` in
float64. The GPU single-vector rows had the same accuracy scale. In the batched
float32 rows, JAX's vmapped dense `expm` differed by as much as `1.2e-4`; a
separate float64 SciPy check found the Krylov result within `8e-8`, identifying
the dense batched exponential as the inaccurate side of that comparison.

### CPU endpoints, batch 1

{distribution_endpoint_table(distribution_cpu)}

The endpoint crossover is clear on CPU: dense exponentiation wins for eight
states, while structured matrix-free Krylov is substantially faster by 32 and
128 states. For a large batch sharing one small generator, dense execution can
reuse the exponential and remains preferable.

### RTX 3090 endpoints, batch 1

{distribution_endpoint_table(distribution_gpu)}

At these small-to-medium dimensions the GPU dense exponential remains faster;
matrix-free Krylov is intended for state spaces where the dense generator is
too expensive to form or store. Both paths remain JIT- and `vmap`-compatible.

### Full distribution paths and query grids

CPU:

{distribution_path_table(distribution_cpu)}

RTX 3090:

{distribution_path_table(distribution_gpu)}

The DTMC associative prefix method is useful on GPU. CTMC query times are
independent exponential actions and are vectorized over the time axis; whether
dense or Krylov wins depends strongly on device and state structure.

## Reproduce

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \\
  uv run python performance_comparison/python/benchmark_markov.py \\
  --quick --output performance_comparison/results/python_cpu_markov.json

JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \\
  uv run python performance_comparison/python/benchmark_markov_distribution.py \\
  --quick \\
  --output performance_comparison/results/python_cpu_markov_distribution.json

JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \\
  uv run python performance_comparison/python/benchmark_markov.py \\
  --quick --output performance_comparison/results/python_gpu_markov.json

JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \\
  uv run python performance_comparison/python/benchmark_markov_distribution.py \\
  --quick \\
  --output performance_comparison/results/python_gpu_markov_distribution.json

JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
  julia --project=performance_comparison/julia \\
  performance_comparison/julia/benchmark_markov.jl \\
  performance_comparison/results/julia_cpu_markov.json --quick

uv run python performance_comparison/render_markov_report.py
```
"""
    OUTPUT.write_text(text)


if __name__ == "__main__":
    main()
