# Performance comparison

This directory is an isolated, rerunnable comparison of tinydiffeq, Diffrax,
and Julia SciML. Its nested Python and Julia environments do not modify the
root tinydiffeq dependencies.

Rodas5P comparisons use SciML's original
[`OrdinaryDiffEqRosenbrock`](https://github.com/SciML/OrdinaryDiffEq.jl/tree/master/lib/OrdinaryDiffEqRosenbrock)
implementation, the reference implementation from the ecosystem in which
Rodas5P was developed.

Run the representative matrix and rebuild the report with:

```bash
./performance_comparison/run.sh --quick
```

Run every configured dtype, transform, system size, and ensemble size with:

```bash
./performance_comparison/run.sh --full
```

The main artifact is [`CURRENT_RESULTS.md`](CURRENT_RESULTS.md); the raw JSON
is in `results/`.

Finite-state DTMC/CTMC simulation has a separate matrix because it compares
sequential and associative execution rather than differential-equation methods:

```bash
./performance_comparison/run_markov.sh --quick
```

Its generated artifact is [`MARKOV_RESULTS.md`](MARKOV_RESULTS.md). The Markov
benchmark includes alias preparation separately, returns complete JAX paths,
and compares the two-state CTMC CPU slice with warmed, type-inferred SciML
`Direct()`/`SSAStepper()` execution. It also benchmarks deterministic DTMC
distribution propagation and dense versus structured matrix-free Krylov CTMC
forecasts on CPU and GPU.

Large matrix-free terminal PMF and initial-state VJP results use SciML
`ExponentialUtilities.expv` as the reference and are recorded in
[`LINEAR_CTMC_RESULTS.md`](LINEAR_CTMC_RESULTS.md). The Python benchmark also
measures traced versus hand-coded exponential JVP/VJP and `vmap` over distinct
initial distributions. It records fixed and adaptive tinydiffeq actions. The
adaptive reference is SciML's Niesen--Wright `expv_timestep(adaptive=true)`,
which adapts both time slices and Krylov dimension; tinydiffeq deliberately
keeps the dimension static for JAX compilation. Both receive `krylov_dim=30`
initially and the same relative tolerance (`1e-5` for float32 and `1e-10` for
float64); tinydiffeq additionally uses `atol=0.01 * rtol`.

## Fairness rules

- Main timings exclude compilation and save only the endpoint.
- JAX inputs are already on the selected device, and every timed call blocks
  until its full output is ready.
- Julia calls are warmed before `BenchmarkTools` sampling. CPU comparisons use
  one Julia thread and one BLAS thread.
- Fixed-step rows match the numerical method, interval, dtype, and step count.
- Adaptive rows enter an exact-match table only if an accepted-mesh probe also
  matches. Otherwise they are labeled same-tolerance/native-controller.
- Fixed Rodas5P ODE/DAE rows match the method, equations, interval, dtype,
  step count, and endpoint-only output. Adaptive rows explicitly identify
  native-controller differences. Unsupported combinations remain empty.

The stochastic JAX comparisons generate fixed, independent Brownian increments
from per-trajectory keys inside the timed solve. Derivatives are pathwise with
those keys held fixed.
