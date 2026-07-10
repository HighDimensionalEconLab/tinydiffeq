# Benchmarks

Run the opt-in CPU suite with:

```bash
JAX_PLATFORMS=cpu uv run --group benchmark pytest benchmarks --benchmark-only
```

It measures post-compilation primal, JVP, and VJP execution for RK4, Tsit5,
Euler-Maruyama, and the semi-explicit DAE/SDAE solvers. Each method is
exercised on a scalar array, a length-16 array, and an equal-sized two-leaf
pytree. The array cases are the performance-regression baseline; pytree
timings show the cost of executing separate leaves. Compilation is
deliberately performed outside the timed region.

Measure cold compilation separately (optionally selecting a subset) with:

```bash
JAX_PLATFORMS=cpu uv run --group benchmark python -m benchmarks.compile_times
JAX_PLATFORMS=cpu uv run --group benchmark python -m benchmarks.compile_times \
  --methods rk4 tsit5 --states vector16 --repeat 5
```

The script clears JAX's compilation caches between `timeit` repetitions and
reports the median. Keep compilation and execution results separate.

`test_aux_dense_output.py` separately measures the intentionally changed DAE
grid-output path across accepted-step and query-grid counts, both with and
without aux, plus batched adaptive interpolation. The no-aux cases can run
against an older checkout for a direct cross-version comparison. Those
results are re-baselined rather than held to the unchanged endpoint/steps
array-path gate.

For changes to the state arithmetic, compare the JSON outputs before and
after the change using `--benchmark-json=PATH`. Treat an array-path slowdown
larger than `max(5%, 1 us)` as a regression. GPU timing is informative rather
than a release gate because available devices vary; GPU correctness belongs
in `tests/test_gpu.py`.
