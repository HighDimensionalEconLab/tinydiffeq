# tinydiffeq

`tinydiffeq` is a deliberately tiny set of differentiable ODE/SDE/DAE
integrators for JAX: fixed-step Euler and RK4, adaptive Tsit5 with integral or
proportional-integral step-size control, fixed-step Euler–Maruyama for Itô
SDEs, and nonstiff semi-explicit index-1 DAEs. Time integration runs
inside one bounded `lax.scan` with static shapes, and every solve is
differentiable in **both** forward and reverse mode — including
reverse-over-forward, the pattern a Levenberg–Marquardt optimizer with
geodesic acceleration needs when it differentiates through a rollout.

It is a jvp/vjp-friendly subset of
[diffrax](https://docs.kidger.site/diffrax/). **Use diffrax instead if you
need any of:**

- stiff or fully implicit solvers, or higher-index DAEs
- full derivative-term PID step-size control
- events, root-finding, or backward-time integration
- dense output / continuous interpolation objects
- checkpointed or backsolve adjoints for long horizons

tinydiffeq ships only the O(max_steps)-memory bounded-scan approach, because
that is the one that composes cleanly with `jax.jvp`, `jax.vjp`, `jax.vmap`,
and reverse-over-forward without custom adjoint machinery.

## Install

```bash
uv add tinydiffeq
```

For accelerator use, install the JAX build matching your hardware alongside
it, for example:

```bash
uv add tinydiffeq "jax[cuda13]"
```

## Vector-field interface

The vector field may take one to four positional arguments — always in this
order:

```python
f(x)                # autonomous, closes over everything
f(x, t)
f(x, t, args)
f(x, t, args, p)
```

- `x` is an array or pytree state. Leaves must share one real floating dtype;
  the vector field returns the same structure and dtype.
- `t` is time.
- `args` is pass-through data. By convention it is **not** an AD target —
  nothing stops you differentiating with respect to it, but the library's
  contracts and tests treat it as constants.
- `p` holds differentiable parameters — any pytree, e.g. neural-network
  weights. jvp/vjp with respect to `p` and `x_0` are first-class and tested.

The arity is inspected once and the function is wrapped into the canonical
four-argument form, so the compiled code is identical for all four. There is
no special autonomous code path: an unused `t` is dead-code-eliminated.
`drift` and `diffusion` in [`solve_sde`](sde.md) follow the same convention.
Semi-explicit DAE fields use `(y, z)`, `(y, z, t)`,
`(y, z, t, args)`, or `(y, z, t, args, p)`; see
[Semi-Explicit DAEs](dae.md).

## Minimal example

```python
import jax
import jax.numpy as jnp
from tinydiffeq import solve_ode, Tsit5, IController, SaveAt

jax.config.update("jax_enable_x64", True)  # your call, not the library's


def f(x, t, args, p):
    return -p * x


sol = solve_ode(
    f, Tsit5(), 0.0, 2.0, jnp.asarray(1.0),
    p=jnp.asarray(1.3),
    dt_0=0.1,
    controller=IController(rtol=1e-8, atol=1e-10),
    max_steps=512,
    save_at=SaveAt(ts=jnp.linspace(0.0, 2.0, 21)),
)
sol.xs  # (21,) states on the grid, however many internal steps were taken
sol.ok  # False if max_steps ran out before t_1
```

With no arguments, `IController()` and `PIController()` use precision-aware
tolerances: `rtol=1e-4, atol=1e-6` for float32 states and
`rtol=1e-7, atol=1e-9` for float64 states. Explicit values override the
policy and are cast to the state dtype. Automatic `dt_min` is
`10 * eps * max(1, abs(t_1))` in the time dtype.

Gradients go straight through the solve:

```python
def endpoint(p):
    return solve_ode(
        f, Tsit5(), 0.0, 2.0, jnp.asarray(1.0), p=p,
        dt_0=0.1, controller=IController(rtol=1e-10, atol=1e-12),
        max_steps=512,
    ).xs

jax.grad(endpoint)(jnp.asarray(1.3))          # reverse mode
jax.jvp(endpoint, (jnp.asarray(1.3),), (jnp.asarray(1.0),))  # forward mode
jax.grad(lambda p: jax.jvp(endpoint, (p,), (jnp.asarray(1.0),))[1])(
    jnp.asarray(1.3)
)  # reverse-over-forward
```

## Design contracts at a glance

- **`dt_0` is required.** There is no initial-step heuristic.
- **`max_steps` counts attempted internal steps**, including rejections. It
  controls the bounded scan and only becomes an output-row count in
  `SaveAt(steps=True)`, which returns `max_steps + 1` padded rows including
  the initial state. Accepted steps form a contiguous prefix; rejected
  attempts are omitted and the tail repeats the last accepted state.
- **Forward time only**: `t_1 > t_0`.
- **Never poisons.** `sol.ok` reports whether `t_1` was reached; callers that
  want diverging residuals can map `jnp.where(sol.ok, x, jnp.inf)` over
  `sol.xs`.
- **`project`** (an idempotent clamp, e.g. positivity) is applied at every
  point where the vector field is evaluated and to every accepted state.
- **Never sets `jax_enable_x64`.** The time dtype follows the common state
  dtype; float32 problems stay float32 even when x64 is enabled.
- Solvers, controllers, `SaveAt`, and `Solution` are frozen dataclasses
  registered as pytrees: numeric fields (tolerances, grids, `dt_0`, `x_0`) are
  data leaves, so changing them never recompiles.

Read next: [Static Shapes](static_shapes.md) for the bounded-scan design and
`SaveAt`, [Adaptive Stepping and AD](adaptive_ad.md) for what is and is not
differentiated, [SDEs](sde.md), and the [API Reference](api.md).
