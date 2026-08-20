# solve_bvp CPU scaling (2026-08-19)

i9-10900K, float64, `JAX_PLATFORMS=cpu uv run python -m benchmarks.bvp_scaling --repeat 7`,
with the collocation system solved by the structured orthogonal
factorization in `tinydiffeq.babd` (level-batched cyclic reduction).
Every case calls the solve through a jitted wrapper: `cold_ms` is
compile-plus-first-run (fresh callables per case), `run_ms` is the median
of 7 warm runs with `jax.block_until_ready` inside the timed function.
scipy reference uses its finite-difference Jacobians. Warm-cache reuse
verified per the cache-audit protocol (`--cache-audit` under
`JAX_EXPLAIN_CACHE_MISSES=1 JAX_LOG_COMPILES=1`): changing `p` and the
mesh values triggers 0 XLA compiles on the second solve.

`shock` at max_nodes=32 exhausts the node budget (status 1) and times the
run-to-failure path; `vmap_exp_b32` is a jitted 32-lane `vmap` of the
full solve, and `eigen_grad` is `jit(grad)` of an unknown-eigenvalue
solve via the implicit rule.

| case | max_nodes | cold (ms) | warm (ms) | scipy (ms) |
|---|---|---|---|---|
| exp | 32 | 821.8 | 0.250 | 0.601 |
| shock | 32 | 681.9 | 0.450 | 6.323 |
| eigen_grad | 32 | 1108.4 | 0.532 | — |
| vmap_exp_b32 | 32 | 1462.9 | 1.459 | — |
| exp | 128 | 830.9 | 0.470 | 0.593 |
| shock | 128 | 818.8 | 2.072 | 6.300 |
| eigen_grad | 128 | 1306.8 | 1.603 | — |
| vmap_exp_b32 | 128 | 1756.4 | 6.373 | — |

Warm jitted calls beat scipy on every case, including the tiny 5-node
exponential problem (0.25 ms vs 0.60 ms). Calling `solve_bvp` directly
from un-jitted Python instead adds ~3 ms of per-call wrapper work
(validation traces, pytree flattening, dispatch) — the compilation cache
still hits, but the wrapper Python re-runs each call. On an RTX 3090 in
float32, a 64-lane batch at max_nodes=32 solves in ~0.79 ms (~12
us/lane).

## Real-workload check: kernels neoclassical growth baseline

The `kernels` repo's `neoclassical_growth_benchmark` (detrended
saddle-path BVP, `tol=1e-10`, initial mesh `linspace(0, 200, 801)`,
refined to 1489 nodes) rewired to this solver at `max_nodes=1536`
(float64, CPU): mesh evolution is node-for-node identical to scipy (1489
nodes, 5 iterations, max rms 1.0e-10), paths agree to 4e-16, and the
warm solve — timed as a direct un-jitted call, so including the ~3 ms
wrapper overhead — takes **17 ms vs scipy's 17.4 ms**. A dense LU
factorization of the same padded system took 304 ms; the structured
factorization is what closes that gap. Compile is ~3 s, paid once;
changed calibrations reuse the compilation at warm speed.
