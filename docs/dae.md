# Semi-Explicit Index-1 DAEs

`solve_semi_explicit_dae` integrates nonstiff systems of the form

$$
\dot y = f(y, z, t, \mathrm{args}, p), \qquad
0 = g(y, z, t, \mathrm{args}, p),
$$

where the algebraic equation is square and $g_z$ is nonsingular along the
solution. `y` and `z` may independently be array or pytree states. Leaves
within each state share one real floating dtype; the `y` and `z` dtypes may
differ. The residual `g` is a single array whose flattened size matches the
total size of `z`. The implementation supports fixed-step RK4 and Tsit5 with
fixed or adaptive control.

The algebraic solve uses
[`nlls-gram`](https://highdimensionaleconlab.github.io/nlls_gram/square_systems/)'s
solve-only `SquareLevenbergMarquardt`: an augmented-QR damped step for the
primal root and a direct implicit Jacobian solve for derivatives.

## Minimal examples

Consider

$$
\dot y = pz, \qquad 0=z-y,
$$

whose reduced solution is $y(t)=z(t)=y_0e^{pt}$.

```python
import jax.numpy as jnp

from tinydiffeq import (
    IController,
    RK4,
    Tsit5,
    solve_semi_explicit_dae,
)


def f(y, z, t, args, p):
    return p * z


def g(y, z):
    return z - y


fixed = solve_semi_explicit_dae(
    f, g, RK4(), 0.0, 1.0,
    jnp.asarray(1.0), jnp.asarray(0.5),
    p=jnp.asarray(2.0), dt_0=0.01, max_steps=100,
)

adaptive = solve_semi_explicit_dae(
    f, g, Tsit5(), 0.0, 1.0,
    jnp.asarray(1.0), jnp.asarray(0.5),
    p=jnp.asarray(2.0), dt_0=0.1,
    controller=IController(), max_steps=128,
)
```

`z_0` is a root-finding guess, not an assumed-consistent initial value. Both
calls first solve `g(y_0, z, t_0, args, p) = 0`, so the `0.5` guess becomes
the consistent value `1.0`.

## Nonlinear-solve and AD contract

The default root configuration is:

```python
from tinydiffeq import LMRootSolver

root_solver = LMRootSolver(
    max_steps=8,
    atol=None,          # 1e-6 float32, 1e-10 float64
    init_damping=1e-3,
)
```

The outer `max_steps` counts attempted time steps, including adaptive
rejections. `root_solver.max_steps` separately bounds one algebraic root.

Every root passes `(y, t, p)` through nlls-gram's differentiated parameter
pytree. Thus it differentiates the defining constraint,

$$
\dot z = -g_z^{-1}
  (g_y\dot y + g_t\dot t + g_p\dot p),
$$

rather than differentiating the LM iterations. The warm-start guess has zero
derivative by design. `args` is fixed data; put every differentiated model
quantity in `p`. JVP, VJP, `vmap`, and reverse-over-forward compose through
the complete DAE solve.

An adaptive stage-root failure rejects the time-step attempt and asks the
controller for a smaller step. A fixed-step root failure terminates the solve.
In either case `sol.ok` is false if the endpoint is not reached with valid
algebraic states.

## Saving output

All `SaveAt` modes are supported:

- `SaveAt(t_1=True)` returns the consistent endpoint.
- `SaveAt(steps=True)` returns the initial point and accepted internal steps
  as a padded `max_steps + 1` buffer with the usual `accepted` mask.
- `SaveAt(ts=grid)` Hermite-interpolates `y` onto the requested grid and then
  solves `g=0` at each requested time. It never interpolates `z` independently,
  so every returned algebraic state satisfies the constraint.

The result is `DAESolution(ts, ys, zs, ok, num_accepted, accepted)`.
For pytree states, saved rows are a leading axis on every `ys`/`zs` leaf and
the one `accepted` mask applies to the complete state.

## Deliberate limits

This is an explicit, nonstiff, semi-explicit index-1 integrator. It does not
support mass-matrix or fully implicit DAEs, stiff implicit time stepping,
higher-index constraints, or automatic index reduction. It is an initial-value
solver: it does not determine unknown initial costates or solve boundary-value
or saddle-path conditions.
