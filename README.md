# tinydiffeq

[![CI](https://github.com/HighDimensionalEconLab/tinydiffeq/actions/workflows/ci.yml/badge.svg)](https://github.com/HighDimensionalEconLab/tinydiffeq/actions/workflows/ci.yml)
[![Docs](https://github.com/HighDimensionalEconLab/tinydiffeq/actions/workflows/docs.yml/badge.svg)](https://highdimensionaleconlab.github.io/tinydiffeq/)
[![PyPI](https://img.shields.io/pypi/v/tinydiffeq.svg)](https://pypi.org/project/tinydiffeq/)
[![Python versions](https://img.shields.io/pypi/pyversions/tinydiffeq.svg)](https://pypi.org/project/tinydiffeq/)
[![License: MIT](https://img.shields.io/github/license/HighDimensionalEconLab/tinydiffeq)](https://github.com/HighDimensionalEconLab/tinydiffeq/blob/main/LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Tiny differentiable ODE/SDE/DAE/SDAE solvers for JAX: fixed-step Euler/RK4,
adaptive Tsit5, linearly implicit Rodas5P for stiff ODEs and index-1 DAEs,
and Euler–Maruyama for Itô SDEs and semi-explicit index-1 SDAEs. The package
also includes primal, vmap-friendly finite-state DTMC and CTMC simulators with
sequential and associative parallel-prefix execution. Deterministic probability
forecasts are differentiable in the initial mass and include DTMC matrix powers,
dense CTMC exponentials, and matrix-free Arnoldi/Krylov actions over probability
pytrees. The same dense and matrix-free backends are available directly through
`solve_linear_ode` for any fixed homogeneous linear array or pytree operator;
`jvp_linear_ode` and `vjp_linear_ode` apply the exact initial-state tangent and
adjoint exponential actions without differentiating Arnoldi orthogonalization.
Bounded `lax.scan` loops provide exactly `max_steps` attempt slots for fixed and
adaptive stepping, so shapes are static, nothing recompiles as tolerances or
curvature change, and every solve is differentiable in **both** forward and
reverse mode — including reverse-over-forward, the pattern a
Levenberg–Marquardt optimizer with geodesic acceleration needs when it
differentiates through a rollout. Adaptive attempts are grouped into static
chunks, allowing one `lax.cond` to
skip solver and controller work for an entire padded chunk.

This is a deliberately small, jvp/vjp-friendly package. Rodas5P is a JAX
adaptation of Steinebach's method and follows SciML's
[`OrdinaryDiffEqRosenbrock`](https://github.com/SciML/OrdinaryDiffEq.jl/tree/master/lib/OrdinaryDiffEqRosenbrock)
implementation. Use [diffrax](https://docs.kidger.site/diffrax/) or
[SciML](https://docs.sciml.ai/DiffEqDocs/stable/) if you need general mass
matrices, fully implicit or higher-index DAEs, events, continuous solution
objects, sparse/Krylov linear solvers for ODE/DAE stages, or specialized
adjoints. Initial DAE consistency and explicit DAE stages use `nlls-gram`;
`LMRootSolver` accepts a `MAX_STEPS` iterate by default and offers strict
`CONVERGED`-only status via `max_steps_is_success=False`.

The linear exponential-action API follows SciML
[`ExponentialUtilities.expv`](https://docs.sciml.ai/ExponentialUtilities/stable/expv/).
It includes fixed and residual-controlled adaptive matrix-free time slicing;
the latter keeps the Krylov dimension static for predictable JAX compilation.
SciML's
[`ExponentialIntegrators.jl`](https://docs.sciml.ai/ExponentialIntegrators/stable/)
is the reference for the broader nonlinear exponential-integrator family.

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
The state may also be any JAX pytree. It must contain at least one leaf, and
every leaf must be a nonempty real floating array with the same dtype; vector
fields and `project` preserve that structure. Output keeps the structure and
adds the saved-time axis to each
leaf.

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
the adaptive controller still chooses its own internal mesh. Explicit methods
use cubic Hermite interpolation; Rodas5P uses its published stiff-aware
fourth-order continuous extension.

## Semi-explicit DAEs

For a square index-1 system `dy/dt = f(y, z, t, args, p)` and
`0 = g(y, z, t, args, p)`:

```python
from tinydiffeq import IController, Rodas5P, Tsit5, solve_semi_explicit_dae


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

# One initial nonlinear consistency solve, then linear Rodas5P stages.
stiff_dae_sol = solve_semi_explicit_dae(
    dae_f, dae_g, Rodas5P(), 0.0, 1.0,
    jnp.asarray(1.0), jnp.asarray(0.5),
    p=jnp.asarray(2.0), dt_0=0.1,
    controller=IController(), max_steps=128,
)
```

`z_0` is a guess and is made consistent automatically. RK4 and Tsit5 restore
the algebraic root at every stage. Rodas5P performs no nonlinear solves after
initialization: it advances the corresponding block mass-matrix system using
one reused LU factorization per attempt. Differential fields may return a
floating saved-aux pytree stored at accepted nodes and interpolated on requested
deterministic grids. Algebraic equations may separately return internal context
passed to the dynamics. JVP, VJP, and reverse-over-forward propagate through both
implicit initialization and the time integrator. See the
[DAE documentation](https://highdimensionaleconlab.github.io/tinydiffeq/dae/)
for root controls, `SaveAt`, and scope limits.

Fixed-step semi-explicit Itô SDAEs use the corresponding
`solve_semi_explicit_sdae` interface with `EulerMaruyama`, a PRNG key, and
`n_steps`; see the [SDAE documentation](https://highdimensionaleconlab.github.io/tinydiffeq/sdae/).

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
zero error); the states differentiate fully through the solver stages. See the
[docs](https://highdimensionaleconlab.github.io/tinydiffeq/) for the design
contracts: static shapes and `SaveAt`, AD through adaptive stepping, SDE key
semantics, and the package API.

## License

MIT
