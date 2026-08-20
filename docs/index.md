# tinydiffeq

tinydiffeq is an unsupported research repo of vibe-coded ports of
well-established ODE, DAE, and SDE algorithms to JAX. Heavily AI-generated —
but the algorithms are well established, with
[SciML](https://docs.sciml.ai/DiffEqDocs/stable/),
[scipy](https://docs.scipy.org/doc/scipy/reference/integrate.html), and
[diffrax](https://docs.kidger.site/diffrax/) as the reference
implementations — so correctness and performance are often reasonable. The
method set is intentionally minimal, though the package is no longer
especially tiny.

Fixed-step Euler and RK4, adaptive Tsit5, and linearly implicit Rodas5P for
stiff [ODEs](ode.md) and index-1 [DAEs](dae.md); Euler–Maruyama, Milstein,
and SRA1 for Itô [SDEs](sde.md) and [SDAEs](sdae.md); a faithful port of
scipy's collocation [boundary-value solver](bvp.md) for two-point BVPs with
unknown parameters; [Markov chains](markov_chains.md) and
[linear exponential solves](exponential.md). Every solve runs in bounded
`lax` loops with static shapes and composes with `jit`, `vmap`, forward
mode, reverse mode, and reverse-over-forward; iterative solves (BVP, DAE
roots) differentiate implicitly at the solution, never through the
iterations.

**Use [SciML](https://docs.sciml.ai/DiffEqDocs/stable/) or
[diffrax](https://docs.kidger.site/diffrax/) instead if you need any of:**

- general mass matrices, fully implicit solvers, or higher-index DAEs
- sparse/Krylov linear solves and preconditioners inside ODE/DAE stages
- adaptive SDE stepping (Brownian-bridge noise), full PID step-size control
- events, root-finding, or backward-time integration
- dense output objects or checkpointed/backsolve adjoints for long horizons
- multipoint or complex-valued boundary value problems, or BVP meshes beyond
  a few hundred nodes

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
  the field returns the same structure and dtype.
- `args` is pass-through data — by convention **not** an AD target.
- `p` holds differentiable parameters (any pytree, e.g. network weights).
  JVP/VJP with respect to `p` and `x_0` are first-class and tested.

The arity is inspected once and wrapped into the canonical four-argument
form, so the compiled code is identical for all four. `drift` and `diffusion`
in [`solve_sde`](sde.md) follow the same convention; semi-explicit DAE fields
use `(y, z)` through `(y, z, t, args, p)` — see
[Semi-Explicit DAEs](dae.md) and [SDAEs](sdae.md). Fields may return
`(value, saved_aux)` to save extra quantities with the solution.

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
sol.ok  # False if integration or a requested output failed
```

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
```

## Design contracts at a glance

- **`dt_0` is required.** There is no initial-step heuristic.
- **`max_steps` counts attempted internal steps**, including rejections.
  `sol.num_steps` reports attempts, `sol.num_accepted` successful advances.
- **`SaveAt` is the shape contract**: endpoint, fixed interpolation grid, or
  padded accepted-step prefix — output shapes never depend on how many steps
  the controller took. See [ODEs](ode.md#static-shapes-and-saveat).
- **Fixed-step times do not depend on the attempt budget.** They are formed
  arithmetically from the accepted-step index.
- **The controller is stop-gradiented.** States differentiate through solver
  stages on the realized, frozen mesh; see
  [AD through adaptive stepping](ode.md#ad-through-adaptive-stepping).
- **Forward time only**: `t_1 > t_0`.
- **Never poisons.** `sol.ok` reports failure; callers map
  `jnp.where(sol.ok, x, jnp.inf)` when they want loud divergence.
- **Never sets `jax_enable_x64`.** The time dtype follows the state dtype;
  float32 problems stay float32 even when x64 is enabled.
- Solvers, controllers, `SaveAt`, and `Solution` are frozen dataclasses
  registered as pytrees: numeric fields (tolerances, grids, `dt_0`, `x_0`)
  are data leaves, so changing them never recompiles.

Read next: [ODEs](ode.md), [SDEs](sde.md), [Semi-Explicit DAEs](dae.md),
[SDAEs](sdae.md), [Boundary Value Problems](bvp.md),
[Markov Chains](markov_chains.md),
[Linear Exponential Solves](exponential.md), and the
[API Reference](api.md).
