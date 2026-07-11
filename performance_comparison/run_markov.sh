#!/usr/bin/env bash
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mode="${1:---quick}"

if [[ "$mode" != "--quick" && "$mode" != "--full" ]]; then
    echo "usage: ./run_markov.sh [--quick|--full]" >&2
    exit 2
fi

quick=()
if [[ "$mode" == "--quick" ]]; then
    quick=(--quick)
fi

uv sync --project "$here/python"
julia --project="$here/julia" "$here/julia/instantiate.jl"

JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 uv run --project "$here/python" python \
    "$here/python/benchmark_markov.py" "${quick[@]}" \
    --output "$here/results/python_cpu_markov.json"

JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 uv run --project "$here/python" python \
    "$here/python/benchmark_markov_distribution.py" "${quick[@]}" \
    --output "$here/results/python_cpu_markov_distribution.json"

JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
    uv run --project "$here/python" python \
    "$here/python/benchmark_markov.py" "${quick[@]}" \
    --output "$here/results/python_gpu_markov.json"

JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
    uv run --project "$here/python" python \
    "$here/python/benchmark_markov_distribution.py" "${quick[@]}" \
    --output "$here/results/python_gpu_markov_distribution.json"

JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 julia --project="$here/julia" \
    "$here/julia/benchmark_markov.jl" \
    "$here/results/julia_cpu_markov.json" "${quick[@]}"

uv run --project "$here/python" python "$here/render_markov_report.py"
