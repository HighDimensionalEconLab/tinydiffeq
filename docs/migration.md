# Migration from Hand-Rolled Integrators

This page gives exact recipes for replacing the hand-coded `rk4_grid` /
`tsit5_free` integrators (the `kernels` repo's `integrators.py`, from which
tinydiffeq was extracted) with library calls. The test suite pins parity
against verbatim embedded copies of both.

## `rk4_grid`

Before:

```python
ys = rk4_grid(f, y0, n_steps, dt, project=project)   # (n_steps + 1,) states
```

After:

```python
sol = solve_ode(
    f, RK4(), 0.0, n_steps * dt, y0,
    dt0=dt, max_steps=n_steps,
    saveat=SaveAt(steps=True), project=project,
)
ys = sol.xs   # (n_steps + 1,) states, bit-for-bit identical
```

`ConstantStepSize()` is the default controller, and it accepts every step, so
the bounded scan reproduces the fixed grid exactly (`tests/test_solvers_fixed.py`
asserts bit-for-bit equality, clamp binding or not). If you also want the
grid times, they are `sol.ts`.

## `tsit5_free`

Before:

```python
ts, ys = tsit5_free(f, y0, T, n_iters, rtol=rtol, atol=atol, dt0=dt0,
                    project=project)
# poisoned to inf when the budget ran out
```

After:

```python
sol = solve_ode(
    f, Tsit5(), 0.0, T, y0,
    dt0=dt0,
    controller=IController(rtol=rtol, atol=atol),
    max_steps=n_iters,
    saveat=SaveAt(steps=True),
    project=project,
)
ts = jnp.where(sol.ok, sol.ts, jnp.inf)   # the old poisoning, now explicit
ys = jnp.where(sol.ok, sol.xs, jnp.inf)
```

The library never poisons; `sol.ok` says whether `t1` was reached, and the
one-line `jnp.where` reproduces the old behavior for callers whose residual
should reject truncated paths. When the solve completes, `ts`/`ys` match
`tsit5_free` bit-for-bit (`tests/test_adaptive.py`), including the duplicate
rows from rejections and the frozen tail that collocation residuals rely on.

`IController`'s defaults (`safety=0.9`, `factormin=0.2`, `factormax=5.0`,
`dtmin=1e-10`, max-norm error over `atol + rtol * max(|x0|, |x1|)`) are the
old constants.

## Behavior changes to be aware of

- **FSAL cache under a binding clamp** (deliberate fix): `tsit5_free`
  computed the next step's first stage as `k7 = f(y5)` but advanced the
  state to `project(y5)` — a stale cache whenever the clamp binds. tinydiffeq
  evaluates `k7 = f(project(x1))`, consistent with the state actually carried
  forward, at zero extra cost. When the clamp never binds (including all the
  parity tests), the two are identical.
- **Done detection** is relative (`4 * eps * max(1, |t1|)`) instead of the
  absolute `1e-9`. Parity tests confirm the difference is immaterial at
  float64; it matters only for horizons where `1e-9` would be enormous or
  invisible.
- **Final-step stretch**: a step whose remaining horizon is within
  `max_steps * eps` of the desired `dt` is stretched to land on `t1`
  exactly, so `dt0 = T/n` with `max_steps = n` never strands a one-ulp
  sliver.
- **`jax_enable_x64` is yours to set.** `integrators.py` enabled x64 at
  import; tinydiffeq never touches JAX config. Keep
  `jax.config.update("jax_enable_x64", True)` in your application.
- **Time is explicit.** Fields may take `(x, t, ...)`; autonomous fields
  keep the one-argument form and lose nothing.

## Quadrature

`cumulative_trapezoid(g, ts, substeps=...)` generalizes the hand-rolled
`integrate_time_derivative` (composite trapezoid of a time-only function
onto a nonuniform grid) to any output shape, with identical arithmetic —
`tests/test_quadrature.py` pins exact parity. It returns
`(integral, values)` with `integral[0] = 0`.
