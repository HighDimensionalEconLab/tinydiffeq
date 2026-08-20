# tinydiffeq

[![CI](https://github.com/HighDimensionalEconLab/tinydiffeq/actions/workflows/ci.yml/badge.svg)](https://github.com/HighDimensionalEconLab/tinydiffeq/actions/workflows/ci.yml)
[![Docs](https://github.com/HighDimensionalEconLab/tinydiffeq/actions/workflows/docs.yml/badge.svg)](https://highdimensionaleconlab.github.io/tinydiffeq/)
[![PyPI](https://img.shields.io/pypi/v/tinydiffeq.svg)](https://pypi.org/project/tinydiffeq/)
[![Python versions](https://img.shields.io/pypi/pyversions/tinydiffeq.svg)](https://pypi.org/project/tinydiffeq/)
[![License: MIT](https://img.shields.io/github/license/HighDimensionalEconLab/tinydiffeq)](https://github.com/HighDimensionalEconLab/tinydiffeq/blob/main/LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

tinydiffeq is an unsupported research repo of vibe-coded ports of
well-established ODE, DAE, and SDE algorithms to JAX. Heavily AI-generated —
but the algorithms are well established, with
[SciML](https://docs.sciml.ai/DiffEqDocs/stable/),
[scipy](https://docs.scipy.org/doc/scipy/reference/integrate.html), and
[diffrax](https://docs.kidger.site/diffrax/) as the reference
implementations — so correctness and performance are often reasonable. The
method set is intentionally minimal, though the package is no longer
especially tiny.

Fixed-step Euler/RK4, adaptive Tsit5, and linearly implicit Rodas5P for
stiff ODEs and index-1 DAEs; Euler–Maruyama, Milstein, and SRA1 for Itô SDEs
and SDAEs; a port of scipy's collocation solver for two-point BVPs with
unknown parameters; finite-state Markov chains and dense or Krylov linear
exponential actions. Every solve runs in bounded `lax` loops with static
shapes and composes with `jit`, `vmap`, forward mode, reverse mode, and
reverse-over-forward; iterative solves (BVP, DAE roots) differentiate
implicitly at the solution, never through the iterations.

Use [diffrax](https://docs.kidger.site/diffrax/) or
[SciML](https://docs.sciml.ai/DiffEqDocs/stable/) if you need general mass
matrices, fully implicit or higher-index DAEs, adaptive SDE stepping,
events, continuous solution objects, or specialized adjoints.

## Install

```bash
uv add tinydiffeq
```

For GPU use, add the JAX accelerator build matching your hardware, for
example `uv add tinydiffeq "jax[cuda13]"`.

## Example

The vector field may take `(x)`, `(x, t)`, `(x, t, args)`, or
`(x, t, args, p)` — always in that order. `args` is pass-through data (not
an AD target by convention); `p` holds differentiable parameters, and the
state may be any pytree of same-dtype real floating arrays.

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
    save_at=SaveAt(ts=jnp.linspace(0.0, 2.0, 21)),
)
sol.xs  # states on the grid, however many internal steps were taken
sol.ok  # False if integration or a requested output failed
```

`max_steps` bounds attempted internal steps (accepted plus rejected);
`SaveAt` fixes the output shape regardless of how many steps the controller
takes. Gradients go straight through the solve:

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

An SDE ensemble — per-key noise, `vmap` over trajectories:

```python
from tinydiffeq import solve_sde, SRA1


def ou_drift(x):
    return -x


def ou_diffusion(x):
    return 0.5 * jnp.ones_like(x)


def ou_path(key):
    return solve_sde(
        ou_drift, ou_diffusion, SRA1(), 0.0, 1.0, jnp.asarray(1.0),
        key=key, n_steps=256, save_at=SaveAt(steps=True),
    ).xs


paths = jax.vmap(ou_path)(jax.random.split(jax.random.key(0), 1000))  # (1000, 257)
```

SDEs with first-class differentiable noise, semi-explicit DAEs and SDAEs,
two-point BVPs, Markov chains, linear exponential solves, and the design
contracts (static shapes and `SaveAt`, AD through adaptive stepping,
failure-as-data) are in the
[docs](https://highdimensionaleconlab.github.io/tinydiffeq/).

## License

MIT
