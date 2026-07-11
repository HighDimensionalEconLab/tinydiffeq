#!/usr/bin/env bash
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mode="${1:---quick}"

if [[ "$mode" != "--quick" && "$mode" != "--full" ]]; then
    echo "usage: ./run.sh [--quick|--full]" >&2
    exit 2
fi

quick=()
if [[ "$mode" == "--quick" ]]; then
    quick=(--quick)
fi

uv sync --project "$here/python"
julia --project="$here/julia" "$here/julia/instantiate.jl"

JAX_PLATFORMS=cpu uv run --project "$here/python" python \
    "$here/python/benchmark.py" "${quick[@]}" --library both \
    --output "$here/results/python_cpu.json"

JAX_PLATFORMS=cuda uv run --project "$here/python" python \
    "$here/python/benchmark.py" "${quick[@]}" --library both \
    --output "$here/results/python_gpu.json"

JULIA_NUM_THREADS=1 julia --project="$here/julia" \
    "$here/julia/benchmark.jl" "${quick[@]}" \
    --output "$here/results/julia_cpu.json"
JULIA_NUM_THREADS=1 julia --project="$here/julia" \
    "$here/julia/benchmark.jl" "${quick[@]}" --transform vjp \
    --output "$here/results/julia_cpu_vjp.json"

JULIA_NUM_THREADS=1 julia --project="$here/julia" \
    "$here/julia/benchmark_gpu.jl" "${quick[@]}" \
    --output "$here/results/julia_gpu.json"

JAX_PLATFORMS=cpu uv run --project "$here/python" python \
    "$here/python/validate.py"
JAX_PLATFORMS=cuda uv run --project "$here/python" python \
    "$here/python/validate.py"

uv run --project "$here/python" python "$here/render_report.py"
