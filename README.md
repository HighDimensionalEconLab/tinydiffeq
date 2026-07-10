# tinydiffeq

[![CI](https://github.com/HighDimensionalEconLab/tinydiffeq/actions/workflows/ci.yml/badge.svg)](https://github.com/HighDimensionalEconLab/tinydiffeq/actions/workflows/ci.yml)
[![Docs](https://github.com/HighDimensionalEconLab/tinydiffeq/actions/workflows/docs.yml/badge.svg)](https://highdimensionaleconlab.github.io/tinydiffeq/)
[![PyPI](https://img.shields.io/pypi/v/tinydiffeq.svg)](https://pypi.org/project/tinydiffeq/)
[![Python versions](https://img.shields.io/pypi/pyversions/tinydiffeq.svg)](https://pypi.org/project/tinydiffeq/)
[![License: MIT](https://img.shields.io/github/license/HighDimensionalEconLab/tinydiffeq)](https://github.com/HighDimensionalEconLab/tinydiffeq/blob/main/LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Tiny differentiable ODE/SDE solvers for JAX: fixed-step Euler/RK4, adaptive
Tsit5 with integral or proportional-integral step-size control, and
Euler–Maruyama for Itô SDEs.
One bounded `lax.scan` of exactly `max_steps` iterations serves fixed and
adaptive stepping, so shapes are static, nothing recompiles as tolerances or
curvature change, and every solve is differentiable in **both** forward and
reverse mode — including reverse-over-forward, the pattern a
Levenberg–Marquardt optimizer with geodesic acceleration needs when it
differentiates through a rollout. After a solve reaches its horizon, a
`lax.cond` skips solver and controller work during the padded scan tail.

This is a deliberately small, jvp/vjp-friendly subset of
[diffrax](https://docs.kidger.site/diffrax/). Use diffrax if you need pytree
states, stiff/implicit solvers, full derivative-term PID control, events,
dense output, or checkpointed/backsolve adjoints. tinydiffeq's single runtime
dependency is `jax`.

## Install

```bash
uv add tinydiffeq
```

For GPU use, install the JAX accelerator build that matches your hardware,
for example:

```bash
uv add tinydiffeq "jax[cuda13]"
```

## Minimal example

The vector field may take `(x)`, `(x, t)`, `(x, t, args)`, or
`(x, t, args, p)` — always in that order. `args` is pass-through data (not an
AD target by convention); `p` holds differentiable parameters (any pytree).

```python
import jax
import jax.numpy as jnp
from tinydiffeq import solve_ode, Tsit5, IController, SaveAt

jax.config.update("jax_enable_x64", True)  # your call — the library never sets it


def f(x, t, args, p):
    return -p * x


sol = solve_ode(
    f, Tsit5(), 0.0, 2.0, jnp.asarray(1.0),
    p=jnp.asarray(1.3),
    dt_0=0.1,
    controller=IController(rtol=1e-8, atol=1e-10),
    max_steps=512,
    save_at=SaveAt(ts=jnp.linspace(0.0, 2.0, 21)),  # fixed output shape,
)                                                  # however many steps adapt
print(sol.xs)   # states on the grid
print(sol.ok)   # reached t_1 within the max_steps budget?
```

`IController()` and `PIController()` choose tolerances from `x_0.dtype`:
`rtol=1e-4, atol=1e-6` for float32 and `rtol=1e-7, atol=1e-9` for
float64. Pass explicit values when tolerances are part of your model's
scientific specification. The default `dt_min` is
`10 * finfo(dtype).eps * max(1, abs(t_1))`.

`max_steps` is the total internal **attempt budget**: accepted steps plus
rejections. It is not normally the number of returned times. Endpoint mode
returns one time/state, `SaveAt(ts=...)` returns the requested grid, and
`SaveAt(steps=True)` returns the initial state and accepted internal steps as
a contiguous prefix of `max_steps + 1` rows. The remaining rows repeat the
last accepted state by default; `sol.accepted` distinguishes data from
padding. Rejected attempts never appear in the returned trajectory.

`SaveAt(ts=...)` also accepts a Python sequence. These are observation times:
the adaptive controller still chooses its own internal mesh, and cubic
Hermite interpolation evaluates the solution at every requested point.

## Gradients through the solve

```python
def endpoint(p):
    return solve_ode(
        f, Tsit5(), 0.0, 2.0, jnp.asarray(1.0), p=p,
        dt_0=0.1, controller=IController(rtol=1e-10, atol=1e-12),
        max_steps=512,
    ).xs

jax.grad(endpoint)(jnp.asarray(1.3))                      # reverse mode
jax.jvp(endpoint, (jnp.asarray(1.3),), (jnp.asarray(1.0),))  # forward mode
jax.grad(lambda p: jax.jvp(endpoint, (p,), (jnp.asarray(1.0),))[1])(
    jnp.asarray(1.3)
)                                                          # reverse-over-forward
```

The step-size controller is wrapped in `stop_gradient` (accept/reject is
non-differentiable either way, and the error-ratio power blows up at exactly
zero error); the states differentiate fully through the RK stages. See the
[docs](https://highdimensionaleconlab.github.io/tinydiffeq/) for the design
contracts: static shapes and `SaveAt`, AD through adaptive stepping, SDE key
semantics, and the package API.

## License

MIT
