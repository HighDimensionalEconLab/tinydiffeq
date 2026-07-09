# SDEs

`solve_sde` integrates the Itô SDE

$$
dX_t = \mu(X_t, t)\,dt + \sigma(X_t, t)\,dW_t
$$

with fixed-step Euler–Maruyama and **diagonal noise**: `diffusion` returns an
array of the state's shape, multiplied elementwise by independent Brownian
increments.

```python
from tinydiffeq import solve_sde, EulerMaruyama, SaveAt

sol = solve_sde(
    drift, diffusion, EulerMaruyama(), 0.0, 1.0, x0,
    key=jax.random.PRNGKey(0),
    n_steps=256,
    p=(mu, sigma),
    saveat=SaveAt(steps=True),
)
```

`drift` and `diffusion` follow the same `(x)`, `(x, t)`, `(x, t, args)`,
`(x, t, args, p)` signature convention as `solve_ode`.

## The static-shape contract, honestly

`n_steps` must be a static Python int. There is no adaptive SDE stepping in
v1 — adaptivity for SDEs requires noise that can be *re-evaluated* on
subdivided intervals (a Brownian-bridge / VirtualBrownianTree construction),
so that a rejected step resamples consistently. That is diffrax territory
for now; a roadmap issue sketches what it would take here.

## Key semantics: a fixed noise process

The Brownian increments are presampled once,

```python
dW = jnp.sqrt(dt) * jax.random.normal(key, (n_steps,) + x0.shape)
```

so a fixed `key` pins the entire noise path:

- **Reproducible** — the same key gives the same path, every call.
- **Differentiable with respect to `x0` and `p`** (not `key`): with the path
  held fixed, the solution map is smooth, and jvp/vjp against finite
  differences are tested in `tests/test_sde.py`. This is exactly the "common
  random numbers" setup simulation-based estimators want.

## Orders of convergence

Euler–Maruyama is strong order 0.5 (pathwise) and weak order 1.0 (in
distribution) for multiplicative noise. The test suite verifies the strong
rate on geometric Brownian motion, where the exact solution driven by the
*same* increments is available in closed form:

$$
X_T = X_0 \exp\left((\mu - \tfrac{\sigma^2}{2})T + \sigma W_T\right),
\qquad W_T = \textstyle\sum_k \Delta W_k .
$$

Because the increments `dW` are exactly reproducible from the key, the test
regenerates them, computes the exact endpoint on the same path, and checks
the mean absolute error slope across `dt` levels lands in `[0.35, 0.65]`.
That shared-path construction is the right way to measure strong error —
comparing against an independently sampled exact solution measures nothing.

## SaveAt for SDEs

`SaveAt(t1=True)` (default) and `SaveAt(steps=True)` (`n_steps + 1` rows,
all accepted) are supported. `SaveAt(ts=...)` **raises**: cubic Hermite
interpolation assumes smooth trajectories and is simply wrong between the
points of a rough path. Land your grid on the step boundaries instead by
choosing `n_steps`.

`project` is applied to every drift/diffusion evaluation point and every
accepted state, as in `solve_ode`. With `diffusion ≡ 0`, `solve_sde`
reproduces `solve_ode` with `Euler()` on the same grid exactly.
