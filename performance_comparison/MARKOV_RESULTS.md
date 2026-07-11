# Finite-state Markov performance

These are median post-compilation primal path-simulation times. JAX calls
synchronize the full `(batch, path)` result. Alias-table preparation is excluded
from repeated timings. The local CPU benchmark uses one process; GPU results use
the RTX 3090. The main rows simulate 1,024 transitions or bounded jump attempts
for eight states and return every path row.

Sequential and associative methods consume the same uniform/exponential draws.
DTMC states match exactly. CTMC post-jump states match exactly, while associative
holding-time sums differ by floating-point reassociation.

## One-time preparation

| Process | Dtype | Alias preparation |
|---|---|---:|
| discrete | float32 | 691.13 µs |
| discrete | float64 | 745.30 µs |
| continuous | float32 | 707.47 µs |
| continuous | float64 | 743.90 µs |

Preparation validates and normalizes the matrix and constructs the alias tables.
It is performed once on the host, outside transformed simulation and all repeated
latency cells below.

## CPU

| Process | Dtype | Batch | Sequential | Sequential unroll=16 | Associative |
|---|---|---:|---:|---:|---:|
| discrete | float32 | 256 | 1,600.49 µs | 15,624.54 µs | 12,056.21 µs |
| discrete | float32 | 4096 | 44,514.12 µs | 115,006.41 µs | 244,366.48 µs |
| discrete | float64 | 256 | 1,781.87 µs | 13,974.30 µs | 17,849.73 µs |
| discrete | float64 | 4096 | 52,348.22 µs | 94,189.66 µs | 274,519.59 µs |
| continuous | float32 | 256 | 3,141.49 µs | 9,729.42 µs | 32,093.73 µs |
| continuous | float32 | 4096 | 83,762.19 µs | 204,802.35 µs | 605,698.31 µs |
| continuous | float64 | 256 | 3,679.82 µs | 10,777.96 µs | 37,191.40 µs |
| continuous | float64 | 4096 | 98,209.74 µs | 177,174.50 µs | 702,015.09 µs |

CPU favors the default sequential scan. Unrolling and associative map composition
increase executable work and memory traffic.

## RTX 3090 GPU

| Process | Dtype | Batch | Sequential | Sequential unroll=16 | Associative |
|---|---|---:|---:|---:|---:|
| discrete | float32 | 256 | 8,779.81 µs | 1,282.48 µs | 103.76 µs |
| discrete | float32 | 4096 | 8,976.48 µs | 2,107.68 µs | 1,179.06 µs |
| discrete | float64 | 256 | 8,804.74 µs | 2,028.73 µs | 131.99 µs |
| discrete | float64 | 4096 | 8,916.29 µs | 2,691.36 µs | 2,003.89 µs |
| continuous | float32 | 256 | 7,861.89 µs | 2,803.22 µs | 196.54 µs |
| continuous | float32 | 4096 | 8,434.73 µs | 2,811.16 µs | 2,987.28 µs |
| continuous | float64 | 256 | 8,692.64 µs | 4,928.54 µs | 306.36 µs |
| continuous | float64 | 4096 | 10,245.42 µs | 5,076.12 µs | 4,561.88 µs |

The associative alias kernel exposes temporal parallelism and dominates the
medium-batch cases. For large CTMC float32 ensembles, unroll=16 can be competitive;
the explicit method choice is retained because the crossover depends on all four
of state count, path length, batch size, and dtype.

## SciML CPU capability comparison

| Dtype | tinydiffeq sequential | SciML Direct/SSAStepper | Ratio SciML/tiny |
|---|---:|---:|---:|
| float32 | 388.26 µs | 346.83 µs | 0.89× |
| float64 | 489.62 µs | 310.85 µs | 0.63× |

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

| Problem | States | Dtype | Dense/power | Matrix-free Krylov/scan |
|---|---:|---|---:|---:|
| DTMC endpoint | 8 | float32 | 12.83 µs | 24.52 µs |
| DTMC endpoint | 8 | float64 | 12.28 µs | 26.97 µs |
| DTMC endpoint | 32 | float32 | 15.58 µs | 87.91 µs |
| DTMC endpoint | 32 | float64 | 13.24 µs | 108.80 µs |
| DTMC endpoint | 128 | float32 | 16.77 µs | 749.98 µs |
| DTMC endpoint | 128 | float64 | 15.57 µs | 1,327.22 µs |
| structured CTMC endpoint | 8 | float32 | 23.27 µs | 60.63 µs |
| structured CTMC endpoint | 8 | float64 | 24.82 µs | 77.48 µs |
| structured CTMC endpoint | 32 | float32 | 5,156.08 µs | 200.04 µs |
| structured CTMC endpoint | 32 | float64 | 5,985.38 µs | 260.07 µs |
| structured CTMC endpoint | 128 | float32 | 829.26 µs | 350.88 µs |
| structured CTMC endpoint | 128 | float64 | 16,476.26 µs | 468.86 µs |

The endpoint crossover is clear on CPU: dense exponentiation wins for eight
states, while structured matrix-free Krylov is substantially faster by 32 and
128 states. For a large batch sharing one small generator, dense execution can
reuse the exponential and remains preferable.

### RTX 3090 endpoints, batch 1

| Problem | States | Dtype | Dense/power | Matrix-free Krylov/scan |
|---|---:|---|---:|---:|
| DTMC endpoint | 8 | float32 | 40.73 µs | 5,744.92 µs |
| DTMC endpoint | 8 | float64 | 42.82 µs | 5,790.57 µs |
| DTMC endpoint | 32 | float32 | 43.01 µs | 6,831.30 µs |
| DTMC endpoint | 32 | float64 | 61.47 µs | 6,238.28 µs |
| DTMC endpoint | 128 | float32 | 49.89 µs | 7,672.58 µs |
| DTMC endpoint | 128 | float64 | 59.64 µs | 8,484.10 µs |
| structured CTMC endpoint | 8 | float32 | 539.30 µs | 834.43 µs |
| structured CTMC endpoint | 8 | float64 | 664.28 µs | 824.17 µs |
| structured CTMC endpoint | 32 | float32 | 562.70 µs | 1,526.43 µs |
| structured CTMC endpoint | 32 | float64 | 661.79 µs | 1,976.84 µs |
| structured CTMC endpoint | 128 | float32 | 816.24 µs | 1,552.80 µs |
| structured CTMC endpoint | 128 | float64 | 1,168.45 µs | 2,544.40 µs |

At these small-to-medium dimensions the GPU dense exponential remains faster;
matrix-free Krylov is intended for state spaces where the dense generator is
too expensive to form or store. Both paths remain JIT- and `vmap`-compatible.

### Full distribution paths and query grids

CPU:

| Forecast | Dtype | Sequential/dense | Associative/Krylov |
|---|---|---:|---:|
| DTMC full path, K=8, B=256 | float32 | 2,427.69 µs | 652.21 µs |
| CTMC 33-point grid, K=32, B=1 | float32 | 23,380.41 µs | 3,754.05 µs |
| DTMC full path, K=8, B=256 | float64 | 8,801.36 µs | 3,410.01 µs |
| CTMC 33-point grid, K=32, B=1 | float64 | 90,399.82 µs | 6,370.98 µs |

RTX 3090:

| Forecast | Dtype | Sequential/dense | Associative/Krylov |
|---|---|---:|---:|
| DTMC full path, K=8, B=256 | float32 | 8,404.81 µs | 127.71 µs |
| CTMC 33-point grid, K=32, B=1 | float32 | 331.67 µs | 2,441.26 µs |
| DTMC full path, K=8, B=256 | float64 | 8,835.31 µs | 286.66 µs |
| CTMC 33-point grid, K=32, B=1 | float64 | 787.08 µs | 3,542.49 µs |

The DTMC associative prefix method is useful on GPU. CTMC query times are
independent exponential actions and are vectorized over the time axis; whether
dense or Krylov wins depends strongly on device and state structure.

## Reproduce

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
  uv run python performance_comparison/python/benchmark_markov.py \
  --quick --output performance_comparison/results/python_cpu_markov.json

JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
  uv run python performance_comparison/python/benchmark_markov_distribution.py \
  --quick \
  --output performance_comparison/results/python_cpu_markov_distribution.json

JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run python performance_comparison/python/benchmark_markov.py \
  --quick --output performance_comparison/results/python_gpu_markov.json

JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run python performance_comparison/python/benchmark_markov_distribution.py \
  --quick \
  --output performance_comparison/results/python_gpu_markov_distribution.json

JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  julia --project=performance_comparison/julia \
  performance_comparison/julia/benchmark_markov.jl \
  performance_comparison/results/julia_cpu_markov.json --quick

uv run python performance_comparison/render_markov_report.py
```
