# tinydiffeq — agent notes

## Recompilation hygiene: a manual audit, not a test

Do not write tests against `fn._cache_size()`. It reads JAX's globally shared
C++ executable cache (capacity 8192, LRU, shared by every jitted function and
every `jnp` op in the process), so absolute entry-count assertions turn flaky
as the suite grows: entries get evicted before the assert and the count reads
zero.

Instead, after a big refactor or a substantial new feature, audit
recompilation with the environment-variable protocol from the `jax-project`
skill: two solves in one process with changed data leaves, a flushed marker
before each, and zero trace/compile events after the second marker.

```bash
JAX_EXPLAIN_CACHE_MISSES=1 JAX_LOG_COMPILES=1 \
  uv run python -m benchmarks.bvp_scaling --cache-audit > /tmp/audit.log 2>&1
grep -c "Compiling" /tmp/audit.log   # after the "=== SOLVE 1" marker: zero
```

A per-call `eval_shape` validation trace is expected and compiles nothing.
Leaf-value changes (tolerances, meshes, `p`, `args`, initial states) must
not recompile; only static configuration (attempt budgets, `max_nodes`,
pytree structure, function identity, jacobian modes, solver objects) may. A
changed initial mesh *length* for `solve_bvp` reuses the solve executable
but compiles trivial eager padding ops (`concatenate`, `broadcast_in_dim`)
once per new length — those events are expected in the audit log.
