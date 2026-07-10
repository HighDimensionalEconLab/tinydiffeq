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


def g(y, z, t, args, p):
    return z - y, {"flow": p * z, "level": y + z}


fixed = solve_semi_explicit_dae(
    f, g, RK4(), 0.0, 1.0,
    jnp.asarray(1.0), jnp.asarray(0.5),
    p=jnp.asarray(2.0), dt_0=0.01, max_steps=100, has_aux=True,
)

adaptive = solve_semi_explicit_dae(
    f, g, Tsit5(), 0.0, 1.0,
    jnp.asarray(1.0), jnp.asarray(0.5),
    p=jnp.asarray(2.0), dt_0=0.1,
    controller=IController(), max_steps=128, has_aux=True,
)

adaptive.aux["flow"]
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

With `has_aux=True`, `g` returns `(residual, aux)`. Aux is a nonempty pytree
of nonempty real floating arrays; different leaves may use different floating
dtypes. It is never an input to the time integrator or nonlinear solver.
tinydiffeq evaluates it once at the initial consistent root and once at each
accepted endpoint. Ordinary JAX differentiation of this evaluation composes
with the root's implicit derivative, so aux tangents and cotangents include
both its direct dependence on `p` and its indirect dependence through `z`.
Every aux leaf must be finite. A nonfinite initial aux value sets `ok=False`
and the bounded scan performs no stage or root work; a later nonfinite aux
value terminates at the previous accepted node.

An adaptive stage-root failure rejects the time-step attempt and asks the
controller for a smaller step. A fixed-step root failure terminates the solve.
In either case `sol.ok` is false if the endpoint is not reached with valid
algebraic states. Nonconverged roots receive a zero implicit tangent before
the linear solve, and aux at a failed initial root is a zero pytree of the
declared shape. Callers that want to retain successful-lane JVPs/VJPs from a
mixed-success `vmap` batch should pass
`failure_ad_reference=(y_ref, z_ref, t_ref, p_ref)`, choosing a point where
the residual and aux maps are finite and differentiable. Inactive lanes are
linearized at that point before their tangents are zeroed. It never affects a
successful lane or the primal solve. Without an explicit reference, an
all-ones best-effort default is used; gradients of a batch containing failures
are not guaranteed if the model is undefined there. A failed lane itself is
never a valid solution.

## Saving output

All `SaveAt` modes are supported:

- `SaveAt(t_1=True)` returns the consistent endpoint.
- `SaveAt(steps=True)` returns the initial point and accepted internal steps
  as a padded `max_steps + 1` buffer with the usual `accepted` mask.
- `SaveAt(ts=grid)` cubic-Hermite-interpolates `y`, `z`, and aux from accepted
  internal knots. It performs no query-time nonlinear solves.

The result is `DAESolution(ts, ys, zs, ok, num_accepted, accepted, aux)`.
For pytree states, saved rows are a leading axis on every state and aux leaf;
the one `accepted` mask applies to the complete output.

### Why dense algebraic output uses implicit Hermite slopes

At a consistent knot, differentiating the constraint gives

$$
g_z\dot z = -(g_y\dot y + g_t).
$$

tinydiffeq solves this linear system once per accepted knot only when a query
grid is requested. It then obtains `aux_dot` by a JVP of the aux map along
$(\dot y,\dot z,1)$. Values and total derivatives feed the same normalized
cubic Hermite basis used for ODE states. This is an order-3 continuous
extension—uniform interpolation error $O(h^4)$—when `f` and `g` are $C^4$,
$g_z$ stays uniformly nonsingular near the solution, and root error is no
larger than the desired dense-output error. RK4 and Tsit5 knot errors meet the
required order under their usual assumptions.

The normalized coordinate stays in `[0, 1]` and Hermite basis coefficients
are bounded by 3, which is favorable in float32. SciML's specialized Tsit5
dense polynomial has one higher order for `y`, but requires all seven stages,
does not directly supply `z`/aux output, and has much larger coefficients.
Using one Hermite construction keeps `y`, `z`, and aux at the same dense order
with substantially less storage.

Interpolated `z` and aux are approximations: away from accepted knots they
need not satisfy `g=0` exactly. The constraint defect is $O(h^4)$ under the
conditions above. Use `SaveAt(steps=True)` when every returned row must be an
actual converged root. Dense output also requires one `g_z` factorization per
accepted knot, rather than one nonlinear solve per requested time; its cost
therefore scales with internal steps rather than grid length.

Knot selection and adaptive step sizes remain non-differentiable, consistent
with the frozen-controller convention. Values, implicit slopes, and aux are
fully differentiated. If `sol.ok` is false, neither outputs nor their
derivatives should be treated as a valid solution.

## Deliberate limits

This is an explicit, nonstiff, semi-explicit index-1 integrator. It does not
support mass-matrix or fully implicit DAEs, stiff implicit time stepping,
higher-index constraints, or automatic index reduction. It is an initial-value
solver: it does not determine unknown initial costates or solve boundary-value
or saddle-path conditions. A warm-start guess follows one local root branch;
branch selection and jumps between multiple roots are not differentiable.
