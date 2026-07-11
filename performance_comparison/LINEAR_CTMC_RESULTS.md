# Large linear CTMC terminal forecasts

This benchmark propagates a matrix-free ring CTMC to one terminal time. The
generator action is O(N), the horizon is 10, and every implementation uses a
30-vector Arnoldi space with two equal substeps. Inputs are already on device;
JAX calls are compiled, warmed, and synchronized, while Julia is warmed and
measured with `BenchmarkTools`. Only the terminal PMF is returned.

The local CPU is an Intel i9-10900K. Julia and BLAS use one thread. JAX uses its
normal CPU runtime. GPU rows use the local RTX 3090. Consequently the CPU table
is the direct language comparison; the GPU table is a hardware crossover, not
a SciML GPU comparison.

## CPU primal and terminal VJP

Times are post-compilation medians. The VJP differentiates with respect to the
initial PMF while holding the generator and terminal cotangent fixed.

| States | Dtype | tiny primal | SciML `expv` primal | tiny traced value+VJP | tiny hand-coded value+VJP | SciML adjoint value+VJP |
|---:|---|---:|---:|---:|---:|---:|
| 10,000 | float32 | 5.62 ms | 1.69 ms | 58.70 ms | 11.86 ms | 3.44 ms |
| 100,000 | float32 | 37.82 ms | 27.66 ms | 769.83 ms | 77.64 ms | 57.40 ms |
| 10,000 | float64 | 11.85 ms | 3.45 ms | 116.35 ms | 22.80 ms | 7.02 ms |
| 100,000 | float64 | 112.74 ms | 63.48 ms | 1,482.84 ms | 221.12 ms | 126.31 ms |

With the stable two-pass default, SciML is roughly 1.4--3.4x faster for the CPU primal. Its VJP is the mathematical
adjoint action `exp(t*A') * cotangent`; Zygote differentiation through
`ExponentialUtilities.expv` was unsupported in this test. tinydiffeq's public
`vjp_linear_ode` performs the same adjoint action and removes most of the cost
of tracing backward through Arnoldi: on the 10,000-state float32 case it
reduces value+VJP from 58.7 ms to 11.9 ms. The remaining CPU gap is in the
primal Arnoldi kernel.

The optimized basis is stored in row-contiguous form. With explicitly selected
one-pass reorthogonalization, the same rows become:

| States | Dtype | tiny primal, one pass | tiny hand-coded value+VJP | SciML primal | SciML value+VJP |
|---:|---|---:|---:|---:|---:|
| 10,000 | float32 | 3.96 ms | 7.38 ms | 1.69 ms | 3.44 ms |
| 100,000 | float32 | 23.91 ms | 51.14 ms | 27.66 ms | 57.40 ms |
| 10,000 | float64 | 6.97 ms | 12.54 ms | 3.45 ms | 7.02 ms |
| 100,000 | float64 | 63.68 ms | 134.84 ms | 63.48 ms | 126.31 ms |

Thus the 100,000-state one-pass primal matches or slightly exceeds SciML on
this structured CTMC. Two passes remain the package default because classical
Gram--Schmidt can lose orthogonality for difficult nonnormal operators.

The traced and hand-coded VJPs agree to `3e-6`--`7e-6` relative error in
float32 and about `1e-13` in float64. Terminal objectives agree with SciML to
the corresponding precision.

## Adaptive matrix-free action

The adaptive comparison starts with a 30-vector basis and uses matched relative
tolerances (`1e-5` for float32 and `1e-10` for float64). tinydiffeq keeps that
basis size static and adapts only the internal time slice; SciML's
Niesen--Wright `expv_timestep(adaptive=true)` adapts both. The tested ring
problem is mild enough that tinydiffeq accepts the full horizon in one slice.

| States | Dtype | tiny adaptive primal | SciML adaptive primal | tiny hand-coded value+VJP | SciML adjoint value+VJP |
|---:|---|---:|---:|---:|---:|
| 10,000 | float32 | 3.40 ms | 1.22 ms | 6.06 ms | 2.45 ms |
| 10,000 | float64 | 5.38 ms | 2.34 ms | 20.53 ms | 8.10 ms |

These are warmed CPU medians from the rerunnable quick harness. The adaptive
method removes the fixed benchmark's unnecessary second slice, but the small-N
Arnoldi kernel remains about 2.3--2.8x slower than SciML. Larger and genuinely
slice-limited cases should be rerun for each application because controller
policies differ. The benchmark JSON records the tinydiffeq accepted count and
success flag so a failed work budget cannot masquerade as a fast solve.

For a single 10,000-state float32 JVP, the CPU hand-coded action reduced
value+JVP from 23.6 ms to 10.3 ms. On GPU, ordinary traced JVP was already
faster (5.41 ms versus 8.35 ms) because it propagates the tangent through the
same Arnoldi computation instead of launching a second action. The package
therefore exposes both paths rather than globally replacing traced JVP.

## JAX on RTX 3090

| States | Dtype | Primal | Traced value+VJP | Hand-coded value+VJP |
|---:|---|---:|---:|---:|
| 10,000 | float32 | 4.02 ms | 9.73 ms | 8.94 ms |
| 100,000 | float32 | 6.40 ms | 31.31 ms | 13.04 ms |
| 10,000 | float64 | 5.22 ms | 20.26 ms | 15.60 ms |
| 100,000 | float64 | 14.69 ms | 60.69 ms | 30.32 ms |

At 100,000 states, the JAX GPU primal is about 3.3--3.8x faster than the
single-core SciML CPU result. The hand-coded GPU VJP is another roughly 2x
primal action. A same-GPU SciML benchmark is still required before claiming a
cross-library GPU win.

## Different initial conditions with `vmap`

`jax.vmap` is supported over array or pytree initial PMFs. Each initial
condition constructs its own Arnoldi basis, so memory scales approximately as
`batch * states * krylov_dim`.

| States | Dtype | Backend | Batch | Total | Per initial | Single latency |
|---:|---|---|---:|---:|---:|---:|
| 10,000 | float32 | CPU | 256 | 2.785 s | 10.88 ms | 5.62 ms |
| 100,000 | float32 | CPU | 16 | 1.707 s | 106.72 ms | 37.82 ms |
| 10,000 | float32 | GPU | 256 | 102.92 ms | 0.402 ms | 4.02 ms |
| 100,000 | float32 | GPU | 16 | 64.62 ms | 4.04 ms | 6.40 ms |
| 10,000 | float64 | GPU | 256 | 221.81 ms | 0.866 ms | 5.22 ms |
| 100,000 | float64 | GPU | 16 | 139.92 ms | 8.75 ms | 14.69 ms |

CPU `vmap` does not improve throughput here. GPU batching is highly beneficial
for many 10,000-state forecasts, but the 100,000-state crossover depends on
dtype and batch size. Large ensembles should be chunked and benchmarked at the
application's actual shape.

## Reproduce

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
  uv run --project performance_comparison/python python \
  performance_comparison/python/benchmark_linear_ctmc.py --quick \
  --output performance_comparison/results/python_cpu_linear_ctmc.json

JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run --project performance_comparison/python python \
  performance_comparison/python/benchmark_linear_ctmc.py --quick \
  --output performance_comparison/results/python_gpu_linear_ctmc.json

JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  julia --project=performance_comparison/julia \
  performance_comparison/julia/benchmark_linear_ctmc.jl \
  performance_comparison/results/julia_cpu_linear_ctmc.json --quick
```

The SciML reference is
[`ExponentialUtilities.expv`](https://docs.sciml.ai/ExponentialUtilities/stable/expv/).
