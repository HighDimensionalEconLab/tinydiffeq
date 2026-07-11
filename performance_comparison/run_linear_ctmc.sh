#!/usr/bin/env bash
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mode="${1:---quick}"

if [[ "$mode" != "--quick" && "$mode" != "--full" ]]; then
    echo "usage: ./run_linear_ctmc.sh [--quick|--full]" >&2
    exit 2
fi

quick=()
if [[ "$mode" == "--quick" ]]; then
    quick=(--quick)
fi

uv sync --project "$here/python"
julia --project="$here/julia" "$here/julia/instantiate.jl"

JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
    uv run --project "$here/python" python \
    "$here/python/benchmark_linear_ctmc.py" "${quick[@]}" \
    --output "$here/results/python_cpu_linear_ctmc.json"

JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
    uv run --project "$here/python" python \
    "$here/python/benchmark_linear_ctmc.py" "${quick[@]}" \
    --output "$here/results/python_gpu_linear_ctmc.json"

JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    julia --project="$here/julia" \
    "$here/julia/benchmark_linear_ctmc.jl" \
    "$here/results/julia_cpu_linear_ctmc.json" "${quick[@]}"
