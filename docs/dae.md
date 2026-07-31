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
general Levenberg–Marquardt solver for both the primal root and its implicit
derivative. The default `Cholesky()` primal solver selects the dense normal form
for the square algebraic system, while nlls's default `ad_solver` selects the
direct nonsymmetric `LU()` rule for that square system. The implicit rule
differentiates the defining equation rather than the LM iterations; no implicit
ridge is added. Tinydiffeq applies the DAE validity policy around that solve but
does not implement a separate square-system IFT.

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
    max_steps_is_success=False,  # compatibility field; MAX_STEPS stays invalid
    atol=None,          # 1e-6 float32, 1e-10 float64
    gtol=0.0,           # required: residual stopping only
    xtol=0.0,           # required: residual stopping only
    predictor="previous",
    solver_options=(),  # nlls-gram constructor defaults
)
```

The outer `max_steps` counts attempted time steps, including adaptive
rejections. `root_solver.max_steps` separately bounds one algebraic root. For
Rodas5P it affects only initial consistency; the method's later stages reuse
one dense LU factorization per attempted time step.

Root tolerances are independent of the outer controller tolerances. Explicit
`atol` must be positive; `None` selects the dtype default, and `atol=0` is
invalid. `gtol` and `xtol` must both equal zero; `LMRootSolver` rejects nonzero
values so `CONVERGED` can only come from the residual test. Every accepted
algebraic root must report `CONVERGED` and have Euclidean residual norm
`sqrt(sum(residual**2))` strictly below the root `atol`.
`max_steps_is_success` remains in the configuration for source compatibility,
but Tinydiffeq always asks nlls to treat `MAX_STEPS` as a failed implicit solve;
setting the field to `True` does not broaden DAE root acceptance.

The primal nonlinear solve uses nlls-gram's dense `Cholesky()` normal-equation
default. For a successful square root, nlls's implicit rule forms `dg/dz`,
forms the right-hand side with respect to `(y, t, p)`, and applies its direct
`LU()` square solve. Transposing that nlls rule supplies the VJP. The primal LM
factorization is not reused because it represents a damped normal equation,
not the accepted root's generally nonsymmetric Jacobian. A failed nlls status
has zero implicit tangent by the nlls solve contract. For the rare root solve
that needs to depart from the defaults, `solver_options` forwards constructor
arguments for either the primal `linear_solver` or the implicit `ad_solver`:

```python
from nlls_gram import QR

root_solver = LMRootSolver(solver_options={"linear_solver": QR()})
```

The names and semantics are nlls-gram's, so they track that package rather
than being mirrored here. Pass a mapping or key/value pairs; it is normalized
to a sorted tuple so equal configurations remain hashable and share one
compiled solver. Algebraic roots fix `cache_jacobian=False` and
`geodesic_acceleration=False` — each DAE stage changes the root problem and
the intended path is the ordinary dense LM step — and `solver_options` rejects
those two options rather than silently honoring an override. `ad_solver`
remains nlls-owned and may be supplied explicitly; the square default is
`LU()`.

`predictor="previous"` is the default: each explicit RK stage starts from the
most recent successful algebraic root. `predictor="secant"` extrapolates from
the accepted-step root through the most recent successful stage at a later
time. Duplicate RK4 stage times, non-forward targets, and failed stages fall
back to the previous root. The predictor is stop-gradiented, so a successful
root still uses the same implicit derivative. Its time-derived extrapolation
scale is cast separately to each algebraic leaf's dtype, preserving the `z`
dtype when the differential/time and algebraic dtypes differ. Secant prediction
assumes the continued branch is locally unique; with multiple roots or a tight
finite iteration budget, changing the guess can change the selected branch,
value, or status.

Every nonlinear root passes `(y, t, p)` to nlls-gram. Its implicit rule then
differentiates the defining constraint,

$$
\dot z = -g_z^{-1}
  (g_y\dot y + g_t\dot t + g_p\dot p),
$$

rather than differentiating the LM iterations. The warm-start guess has zero
derivative by design. Rodas5P differentiates through its exact JAX Jacobian,
time derivative, LU factorization, and linear stage solves. `args` is fixed
data; put every differentiated model quantity in `p`. On the default bounded
integration path, JVP, VJP, `vmap`, and reverse-over-forward compose through
the complete DAE solve.

`sol.num_steps` counts logical time-step attempts, including adaptive
rejections. `sol.num_root_solves` counts active nonlinear root calls, including
the initial consistency solve and failed calls, and `sol.num_root_steps` sums
their LM update counts. Rodas5P therefore reports one root solve regardless of
its time-step count; later linear stages are not nonlinear roots. These
counters have exact-zero tangents. Under `vmap`, a masked lane may still execute
physically while remaining absent from its logical counters.

The default `adaptive_loop="bounded"` uses a reverse-mode-capable bounded scan.
`adaptive_loop="forward"` uses a dynamic actual-work loop for adaptive Tsit5
and Rodas5P. It supports primal evaluation, JVP, and nested forward mode, but
not ordinary reverse mode. A vmapped forward loop runs until the slowest lane
finishes.

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

An adaptive stage-root failure rejects the time-step attempt and asks the
controller for a smaller step; a fixed-step failure terminates. Rodas5P linear
failures follow the same controller policy.
In either case `sol.ok` is false if the endpoint is not reached with valid
algebraic states. nlls supplies the primal LM iterate, diagnostics, and the
implicit JVP/VJP. Tinydiffeq applies the DAE's residual-and-status acceptance
check and returns the differentiation-inert warm-start guess when that check
fails; aux at a failed initial root is a zero pytree of the declared shape.
Callers that want to retain successful-lane JVPs/VJPs after another lane has
already become inactive should pass
`failure_ad_reference=(y_ref, z_ref, t_ref, p_ref)`, choosing a point where
the residual, context, and saved-aux maps are finite and differentiable.
tinydiffeq substitutes this point into an already-inactive root call before
entering nlls, and also uses it for inactive algebraic aux, saved aux, and
differential-field evaluations. It is not an nlls solve argument and never
changes an active root attempt.

Every newly attempted root must therefore be JVP-safe at its actual
`(y, z_guess, t, p)`: the residual and any model context evaluated there must
have valid derivatives. The reference cannot rescue an intrinsically invalid
active attempt after the fact. Without an explicit reference, an all-ones
best-effort default is used for inactive work; gradients are not guaranteed if
the model is undefined there. A failed lane itself is never a valid solution.

## Saving output

All `SaveAt` modes are supported:

- `SaveAt(t_1=True)` returns the endpoint.
- `SaveAt(steps=True)` returns the initial point and accepted internal steps
  as a padded `max_steps + 1` buffer with the usual `accepted` mask.
- `SaveAt(ts=grid)` uses cubic Hermite for root-restored methods and Rodas5P's
  stiff-aware continuous extension for `(y, z)`. Aux uses cubic Hermite in
  both cases. It performs no query-time nonlinear solves.

`SaveAt(ts=..., exact=True)` is not a DAE mode; exact knot gathering is limited
to explicit fixed-step ODEs.

The result is a `DAESolution` with `ts`, `ys`, `zs`, `ok`, `num_accepted`,
`accepted`, `aux`, `num_steps`, `num_root_solves`, and `num_root_steps` fields.
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

Knot selection and adaptive step sizes remain differentiation-inert under the
frozen-controller convention. Values, implicit slopes, and aux differentiate
on that realized mesh, but adaptive `SaveAt(steps=True)` omits mesh motion.
See [Frozen mesh versus a moving-mesh derivative](adaptive_ad.md#frozen-mesh-versus-a-moving-mesh-derivative).
If `sol.ok` is false, neither outputs nor their derivatives should be treated
as a valid solution.

## Deliberate limits

Only the internally constructed constant block mass matrix
`diag(I_y, 0_z)` is supported; there is no public general mass-matrix or fully
implicit residual API. Rodas5P uses dense Jacobians and dense pivoted LU, not
sparse or Krylov linear algebra. Higher-index constraints and automatic index
reduction are unsupported. This is an initial-value solver: it does not
determine unknown initial costates or solve boundary-value or saddle-path
conditions. Initial branch selection and jumps between multiple roots are not
differentiable.
