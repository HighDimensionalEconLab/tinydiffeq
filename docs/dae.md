# Semi-Explicit Index-1 DAEs

`solve_semi_explicit_dae` integrates systems of the form

$$
\dot y = f(y, z, t, \mathrm{args}, p), \qquad
0 = g(y, z, t, \mathrm{args}, p),
$$

where the algebraic equation is square and $g_z$ is nonsingular along the
solution. `y` and `z` may independently be array or pytree states. Leaves
within each state share one real floating dtype; the `y` and `z` dtypes may
differ. The residual `g` is a single array whose flattened size matches the
total size of `z`. The implementation supports root-restored RK4/Tsit5 and
linearly implicit Rodas5P with fixed or adaptive control.

The algebraic solve uses
[`nlls-gram`](https://highdimensionaleconlab.github.io/nlls_gram/)'s
general Levenberg–Marquardt solver with `linear_solver="augmented_qr"`: a direct
augmented-QR damped step for the
primal root and a direct implicit Jacobian solve for derivatives.

Rodas5P is a JAX adaptation of Steinebach's method following SciML's
[`OrdinaryDiffEqRosenbrock`](https://github.com/SciML/OrdinaryDiffEq.jl/tree/master/lib/OrdinaryDiffEqRosenbrock)
implementation. See [Rodas5P](rodas5p.md) for direct links to SciML's tableau,
step, and interpolation sources.

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
    Rodas5P,
    Tsit5,
    solve_semi_explicit_dae,
)


def f(y, z, t, args, p):
    dy = p * z
    return dy, {"flow": dy, "level": y + z}


def g(y, z, t, args, p):
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

adaptive.aux["flow"]

linearly_implicit = solve_semi_explicit_dae(
    f, g, Rodas5P(), 0.0, 1.0,
    jnp.asarray(1.0), jnp.asarray(0.5),
    p=jnp.asarray(2.0), dt_0=0.1,
    controller=IController(), max_steps=128,
)
```

`z_0` is a root-finding guess, not an assumed-consistent initial value. Both
calls first solve `g(y_0, z, t_0, args, p) = 0`, so the `0.5` guess becomes
the consistent value `1.0`. RK4 and Tsit5 then solve the algebraic equation at
every stage. Rodas5P performs no further nonlinear solves.

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
rejections. `root_solver.max_steps` separately bounds one algebraic root. For
Rodas5P it affects only initial consistency; the method's later stages reuse
one dense LU factorization per attempted time step.

Every nonlinear root passes `(y, t, p)` through nlls-gram's differentiated parameter
pytree. Thus it differentiates the defining constraint,

$$
\dot z = -g_z^{-1}
  (g_y\dot y + g_t\dot t + g_p\dot p),
$$

rather than differentiating the LM iterations. The warm-start guess has zero
derivative by design. Rodas5P differentiates through its exact JAX Jacobian,
time derivative, LU factorization, and linear stage solves. `args` is fixed
data; put every differentiated model quantity in `p`. JVP, VJP, `vmap`, and
reverse-over-forward compose through the complete DAE solve.

The differential field may return `(dy, saved_aux)`. Saved aux is a nonempty
pytree of nonempty real floating arrays; different leaves may use different
floating dtypes. tinydiffeq evaluates it at required saved nodes. Ordinary JAX
differentiation composes with either the root's implicit derivative or the
Rodas5P stages, so aux tangents and cotangents include both direct dependence
on `p` and indirect dependence through `z`.

The algebraic function may instead or additionally return
`(residual, algebraic_aux)`. In that case `f` takes
`(y, z, t, args, p, algebraic_aux)`. This value is internal cached context:
the nonlinear solver sees only the residual, and only differential-field
saved aux appears in `sol.aux`. See [Auxiliary Outputs](aux.md) for the four
supported combinations and flag behavior.

Every saved aux leaf and every inexact algebraic-aux leaf must be finite.
Invalid algebraic context at initialization sets `ok=False` before any
time-step work. `SaveAt(steps=True)` and `SaveAt(ts=...)` check saved aux at
the initial and accepted nodes, so an invalid value freezes the previous valid
prefix. Endpoint mode evaluates saved aux only after integration; an invalid
final value retains the endpoint state, returns zero aux, and sets `ok=False`.

An adaptive stage-root or Rodas5P linear failure rejects the time-step attempt
and asks the controller for a smaller step. A fixed-step failure terminates.
In either case `sol.ok` is false if the endpoint is not reached with valid
algebraic states. Nonconverged roots receive a zero implicit tangent before
the linear solve, and aux at a failed initial root is a zero pytree of the
declared shape. Callers that want to retain successful-lane JVPs/VJPs from a
mixed-success `vmap` batch should pass
`failure_ad_reference=(y_ref, z_ref, t_ref, p_ref)`, choosing a point where
the residual, context, and saved-aux maps are finite and differentiable.
Inactive lanes are linearized at that point before their tangents are zeroed.
It never affects a successful lane or the primal solve. Without an explicit
reference, an
all-ones best-effort default is used; gradients of a batch containing failures
are not guaranteed if the model is undefined there. A failed lane itself is
never a valid solution.

## Saving output

All `SaveAt` modes are supported:

- `SaveAt(t_1=True)` returns the endpoint.
- `SaveAt(steps=True)` returns the initial point and accepted internal steps
  as a padded `max_steps + 1` buffer with the usual `accepted` mask.
- `SaveAt(ts=grid)` uses cubic Hermite for root-restored methods and Rodas5P's
  stiff-aware continuous extension for `(y, z)`. Aux uses cubic Hermite in
  both cases. It performs no query-time nonlinear solves.

The result is `DAESolution(ts, ys, zs, ok, num_accepted, accepted, aux)`.
For pytree states, saved rows are a leading axis on every state and aux leaf;
the one `accepted` mask applies to the complete output.

### Dense output for root-restored RK4 and Tsit5

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

### Dense output for Rodas5P

Rodas5P stores the three coefficient pytrees defined by Steinebach's
fourth-order stiff-aware continuous extension. tinydiffeq evaluates the same
polynomial form used by
[SciML's Rosenbrock interpolant](https://github.com/SciML/OrdinaryDiffEq.jl/blob/master/lib/OrdinaryDiffEqRosenbrock/src/rosenbrock_interpolants.jl)
for the combined `(y, z)` state. No `g_z` factorization or nonlinear solve is
performed for requested times.

Aux remains a stored accepted-knot quantity. Its cubic-Hermite endpoint
tangents come from the Rodas polynomial's endpoint derivatives and a JVP of
the aux map. Aux is therefore interpolated rather than recalculated at every
query. Rodas5P accepted knots are not root-restored: their constraint defect,
and that of dense output, is controlled by integration accuracy rather than
`LMRootSolver.atol`.

Knot selection and adaptive step sizes remain non-differentiable, consistent
with the frozen-controller convention. Values, implicit slopes, and aux are
fully differentiated. If `sol.ok` is false, neither outputs nor their
derivatives should be treated as a valid solution.

## Deliberate limits

Only the internally constructed constant block mass matrix
`diag(I_y, 0_z)` is supported; there is no public general mass-matrix or fully
implicit residual API. Rodas5P uses dense Jacobians and dense pivoted LU, not
sparse or Krylov linear algebra. Higher-index constraints and automatic index
reduction are unsupported. This is an initial-value solver: it does not
determine unknown initial costates or solve boundary-value or saddle-path
conditions. Initial branch selection and jumps between multiple roots are not
differentiable.
