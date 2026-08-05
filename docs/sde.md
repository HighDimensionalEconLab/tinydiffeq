# SDEs

`solve_sde` integrates the Itô SDE

$$
dX_t = \mu(X_t, t)\,dt + \sigma(X_t, t)\,dW_t
$$

with fixed steps and **diagonal noise**: `diffusion` returns the same pytree
structure as the state, multiplied leafwise and elementwise by independent
Brownian increments. `n_steps` must be a static Python int; adaptive SDE
stepping requires Brownian-bridge noise that can be re-evaluated on
subdivided intervals, which is diffrax territory.

```python
from tinydiffeq import solve_sde, SRA1, SaveAt

sol = solve_sde(
    drift, diffusion, SRA1(), 0.0, 1.0, x_0,
    key=jax.random.key(0),
    n_steps=256,
    p=(mu, sigma),
    save_at=SaveAt(steps=True),
)
```

`drift` and `diffusion` follow the same `(x)`, `(x, t)`, `(x, t, args)`,
`(x, t, args, p)` signature convention as `solve_ode`.

## Solvers

Three fixed-step schemes, each a frozen dataclass:

| Solver | Strong order | Noise contract |
|---|---|---|
| `EulerMaruyama()` | 0.5 | any diagonal diffusion |
| `Milstein()` | 1.0 | diagonal *commutative*: each diffusion component depends only on its own state component |
| `SRA1()` | 1.5 | *additive*: diffusion independent of the state (may depend on time) |

`Milstein` adds the correction $\tfrac12 g g' (\Delta W^2 - \Delta t)$, with
$g g'$ computed as the forward-mode derivative of the diffusion field in the
direction of its own value. `SRA1` is the Rößler additive-noise stochastic
Runge–Kutta scheme: two drift stages per step plus the time-Wiener integral
$I_{10}$. The contracts are documented, not runtime-checked: a
state-dependent diffusion under `SRA1` silently degrades the order, exactly
as in other SDE libraries.

Each solver declares its own per-step randomness through
`solver.sample_noise(x_0, key, n_steps, dt, dtype)`: Brownian increments
$\Delta W$ for `EulerMaruyama`/`Milstein`, an independent pair
$(\Delta W, \Delta Z)$ for `SRA1` (from which $I_{10}/\Delta t =
\tfrac12(\Delta W + \Delta Z/\sqrt 3)$ is formed internally).

## Explicit noise: the shocks as data

Pass `noise=` instead of `key=` (exactly one is required) to hand `solve_sde`
the realization directly — the same pytree `sample_noise` would produce,
validated against the solver's spec:

```python
noise = SRA1().sample_noise(x_0, key, n_steps, dt, x_0.dtype)
sol = solve_sde(drift, diffusion, SRA1(), 0.0, 1.0, x_0, noise=noise, n_steps=n_steps)
```

This is bit-identical to the `key=` call that would have drawn the same
`noise`, and it makes the noise **first-class data**: inspectable, storable
(e.g. as part of a training set alongside the initial condition), and
differentiable — `jax.grad` with respect to the `noise` pytree works, since
the steppers consume it as ordinary arrays.

## Key semantics: a fixed noise process

For an array state, the Brownian increments retain the original draw exactly,

```python
d_w = jnp.sqrt(dt) * jax.random.normal(key, (n_steps,) + x_0.shape)
```

so a fixed `key` pins the entire noise path:

- **Reproducible** — the same key gives the same path, every call.
- **Differentiable with respect to `x_0` and `p`** (not `key`), and with
  respect to `noise` when passed explicitly: with the path held fixed, the
  solution map is smooth, and jvp/vjp against finite differences are tested
  in `tests/test_sde.py`. This is exactly the common-random-numbers setup
  simulation-based estimators want.

For a pytree, tinydiffeq draws one `(n_steps, total_state_size)` array and
partitions it into leaves in JAX's deterministic leaf order, so the noise is
identical to that of the equivalent flat array. Changing the tree structure
changes the assignment of random components and requires a new compilation.

## Orders of convergence

The test suite verifies the strong rates on shared noise paths: because the
increments are exactly reproducible from the key, the exact solution (for
geometric Brownian motion, $X_T = X_0 \exp((\mu - \sigma^2/2)T + \sigma
W_T)$ with $W_T = \sum_k \Delta W_k$) is evaluated on the *same* path as the
numerical endpoint. Comparing against an independently sampled exact solution
would measure nothing.

## Auxiliary output

Only the drift owns saved aux:

```python
def drift(x, t, args, p):
    return drift_value, saved_aux


def diffusion(x, t, args, p):
    return diffusion_value
```

Aux is stored at the endpoint or at the fixed grid nodes and is
differentiated pathwise under the fixed noise realization. `has_aux=None`
auto-detects the form; `has_aux=False` selects the no-aux scan without the
detection trace.

## SaveAt for SDEs

`SaveAt(t_1=True)` (default) and `SaveAt(steps=True)` (`n_steps + 1` rows,
or a padded accepted prefix after an aux failure) are supported.
`SaveAt(ts=...)` **raises**: cubic Hermite interpolation assumes smooth
trajectories and is simply wrong between the points of a rough path. Land
your grid on the step boundaries instead by choosing `n_steps`.

`project` is applied to every drift/diffusion evaluation point and every
accepted state, as in `solve_ode`. With `diffusion ≡ 0`, `solve_sde`
reproduces `solve_ode` with `Euler()` on the same grid exactly.

## Performance: `unroll`

`unroll=` (a static int, default 1) unrolls that many steps per iteration of
the integration scan — identical values and gradients, fewer and larger GPU
dispatches, more compile time. For small-batch neural-network-drift
ensembles on an L40S, `unroll=8` cut reverse-mode solve time 2–3× and primal
time 1.2–1.7× (see `benchmarks/results/2026-08-04_vulcan-l40s-sde-fixed.md`).
The same argument exists on `solve_ode` for fixed stepping.
