# tinydiffeq

[![CI](https://github.com/HighDimensionalEconLab/tinydiffeq/actions/workflows/ci.yml/badge.svg)](https://github.com/HighDimensionalEconLab/tinydiffeq/actions/workflows/ci.yml)
[![Docs](https://github.com/HighDimensionalEconLab/tinydiffeq/actions/workflows/docs.yml/badge.svg)](https://highdimensionaleconlab.github.io/tinydiffeq/)
[![PyPI](https://img.shields.io/pypi/v/tinydiffeq.svg)](https://pypi.org/project/tinydiffeq/)
[![Python versions](https://img.shields.io/pypi/pyversions/tinydiffeq.svg)](https://pypi.org/project/tinydiffeq/)
[![License: MIT](https://img.shields.io/github/license/HighDimensionalEconLab/tinydiffeq)](https://github.com/HighDimensionalEconLab/tinydiffeq/blob/main/LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Tiny differentiable ODE/SDE/DAE/SDAE solvers for JAX: fixed-step Euler/RK4,
adaptive Tsit5, linearly implicit Rodas5P for stiff ODEs and index-1 DAEs, and
fixed-step Euler–Maruyama, Milstein, and SRA1 for Itô SDEs and semi-explicit
index-1 SDAEs. Solves run in bounded `lax.scan` loops with static shapes and
compose with `jit`, `vmap`, forward mode, reverse mode, and
reverse-over-forward. Finite-state Markov simulation, probability forecasts,
and general fixed homogeneous linear solves (dense or matrix-free Krylov
exponential actions, after SciML's
[`ExponentialUtilities.expv`](https://docs.sciml.ai/ExponentialUtilities/stable/expv/))
round out the package.

This is a deliberately small, jvp/vjp-friendly package. Rodas5P is a JAX
adaptation of Steinebach's method following SciML's
[`OrdinaryDiffEqRosenbrock`](https://github.com/SciML/OrdinaryDiffEq.jl/tree/master/lib/OrdinaryDiffEqRosenbrock)
implementation, and DAE algebraic roots delegate both the primal solve and
the implicit derivative to
[`nlls-gram`](https://highdimensionaleconlab.github.io/nlls_gram/). Use
[diffrax](https://docs.kidger.site/diffrax/) or
[SciML](https://docs.sciml.ai/DiffEqDocs/stable/) if you need general mass
matrices, fully implicit or higher-index DAEs, adaptive SDE stepping, events,
continuous solution objects, sparse/Krylov ODE/DAE stages, or specialized
adjoints.

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
AD target by convention); `p` holds differentiable parameters, and the state
may be any pytree of same-dtype real floating arrays.

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
print(sol.ok)   # reached t_1 with every requested output valid?
```

`max_steps` is the internal attempt budget (accepted plus rejected steps),
not the number of returned times: `SaveAt` picks the endpoint, a fixed
interpolation grid, or the padded accepted-step prefix, so output shapes
never depend on how many steps the controller took. Omitted controller
tolerances follow the state dtype (`1e-4`/`1e-6` in float32,
`1e-7`/`1e-9` in float64).

## SDEs with first-class noise

`solve_sde` integrates diagonal-noise Itô SDEs with `EulerMaruyama` (strong
order 0.5), `Milstein` (1.0, commutative diagonal noise), or `SRA1` (1.5,
additive noise). An Ornstein–Uhlenbeck process under SRA1:

```python
from tinydiffeq import solve_sde, SRA1

theta, sigma, n = 1.0, 0.5, 256


def ou_drift(x):
    return -theta * x


def ou_diffusion(x):
    return sigma * jnp.ones_like(x)


sol = solve_sde(
    ou_drift, ou_diffusion, SRA1(), 0.0, 1.0, jnp.asarray(1.0),
    key=jax.random.key(0), n_steps=n,
)
```

The noise realization can also be passed explicitly — the same pytree
`sample_noise` would draw, now inspectable, storable data that is
differentiable like any other input:

```python
x_0 = jnp.asarray(1.0)
noise = SRA1().sample_noise(x_0, jax.random.key(0), n, jnp.asarray(1.0 / n), x_0.dtype)
same_sol = solve_sde(
    ou_drift, ou_diffusion, SRA1(), 0.0, 1.0, x_0, noise=noise, n_steps=n
)  # bit-identical to the key= call
d_endpoint_d_noise = jax.grad(
    lambda noise: solve_sde(
        ou_drift, ou_diffusion, SRA1(), 0.0, 1.0, x_0, noise=noise, n_steps=n
    ).xs
)(noise)
```

A fixed key (or fixed noise) pins the whole path, so gradients with respect
to `x_0`, `p`, and `noise` are pathwise derivatives under common random
numbers — the setup simulation-based estimators want. `vmap` over
trajectories with per-trajectory `x_0` and noise composes with `jit` and
`grad`.

## Semi-explicit DAEs

For a square index-1 system `dy/dt = f(y, z, t, args, p)` and
`0 = g(y, z, t, args, p)`:

```python
from tinydiffeq import solve_semi_explicit_dae


def dae_f(y, z, t, args, p):
    dy = p * z
    return dy, {"flow": dy}


def dae_g(y, z, t, args, p):
    return z - y


dae_sol = solve_semi_explicit_dae(
    dae_f, dae_g, Tsit5(), 0.0, 1.0,
    jnp.asarray(1.0), jnp.asarray(0.5),
    p=jnp.asarray(2.0), dt_0=0.1,
    controller=IController(), max_steps=128,
)
print(dae_sol.ys, dae_sol.zs, dae_sol.aux["flow"])
```

`z_0` is a guess and is made consistent automatically. RK4 and Tsit5 restore
the algebraic root at every stage through `nlls-gram`, which also supplies
the root's implicit derivative; `Rodas5P()` instead performs one initial
consistency solve and then advances the block mass-matrix system with one
reused LU factorization per attempt — the stiff path. Stochastic
semi-explicit systems use `solve_semi_explicit_sdae` with `EulerMaruyama` or
`SRA1`. See the
[DAE](https://highdimensionaleconlab.github.io/tinydiffeq/dae/) and
[SDAE](https://highdimensionaleconlab.github.io/tinydiffeq/sdae/) docs.

## Gradients through the solve

```python
def endpoint(p):
    return solve_ode(
        f, Tsit5(), 0.0, 2.0, jnp.asarray(1.0), p=p,
        dt_0=0.1, controller=IController(rtol=1e-10, atol=1e-12),
        max_steps=512,
    ).xs

jax.grad(endpoint)(jnp.asarray(1.3))                         # reverse mode
jax.jvp(endpoint, (jnp.asarray(1.3),), (jnp.asarray(1.0),))  # forward mode
```

The step-size controller is wrapped in `stop_gradient` (accept/reject is
non-differentiable either way); states differentiate through the solver
stages on the realized, frozen mesh. See the
[docs](https://highdimensionaleconlab.github.io/tinydiffeq/) for the design
contracts: static shapes and `SaveAt`, AD through adaptive stepping, SDE
noise semantics, and the package API.

## License

MIT
