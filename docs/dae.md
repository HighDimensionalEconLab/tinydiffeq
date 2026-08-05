# Semi-Explicit Index-1 DAEs

`solve_semi_explicit_dae` integrates systems of the form

$$
\dot y = f(y, z, t, \mathrm{args}, p), \qquad
0 = g(y, z, t, \mathrm{args}, p),
$$

where the algebraic equation is square and $g_z$ is nonsingular along the
solution. `y` and `z` may independently be array or pytree states; their
dtypes may differ. The residual `g` is a single array whose flattened size
matches the total size of `z`. Root-restored RK4/Tsit5 and linearly implicit
Rodas5P are supported, with fixed or adaptive control.

The algebraic solve uses
[`nlls-gram`](https://highdimensionaleconlab.github.io/nlls_gram/)'s
Levenberg–Marquardt solver for both the primal root and its implicit
derivative: the primal defaults to dense `Cholesky()` normal equations, and
the square implicit rule defaults to a direct nonsymmetric `LU()` solve of
the defining constraint rather than a differentiation of the LM iterations.

## Minimal examples

Consider $\dot y = pz$, $0 = z - y$, whose reduced solution is
$y(t) = z(t) = y_0 e^{pt}$.

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
```

`z_0` is a root-finding guess, not an assumed-consistent initial value: both
calls first solve `g(y_0, z, t_0, args, p) = 0`, so the `0.5` guess becomes
the consistent value `1.0`. RK4 and Tsit5 then re-solve the algebraic
equation at every stage.

## Nonlinear-solve and AD contract

The default root configuration is:

```python
from tinydiffeq import LMRootSolver

root_solver = LMRootSolver(
    max_steps=8,
    atol=None,          # 1e-6 float32, 1e-10 float64
    gtol=0.0,           # required: residual stopping only
    xtol=0.0,           # required: residual stopping only
    predictor="previous",
    solver_options=(),  # nlls-gram constructor defaults
)
```

The outer `max_steps` counts attempted time steps; `root_solver.max_steps`
separately bounds one algebraic root, and root tolerances are independent of
the controller tolerances. `gtol` and `xtol` must be zero so `CONVERGED` can
only come from the residual test: every accepted root must report
`CONVERGED` with Euclidean residual norm strictly below `atol`, and a
`MAX_STEPS` iterate is never a differentiable root.

`solver_options` forwards constructor arguments verbatim to nlls-gram's
`LevenbergMarquardt` — either the primal `linear_solver` or the implicit
`ad_solver`:

```python
from nlls_gram import QR

root_solver = LMRootSolver(solver_options={"linear_solver": QR()})
```

The names and semantics are nlls-gram's, so they track that package rather
than being mirrored here. Options are normalized to a sorted tuple so equal
configurations stay hashable and share one compiled solver.
`cache_jacobian` and `geodesic_acceleration` are fixed to `False` and
rejected — each DAE stage changes the root problem, and the intended path is
the ordinary dense LM step.

`predictor="previous"` (default) starts each explicit RK stage from the most
recent successful root. `predictor="secant"` extrapolates from the
accepted-step root through the most recent successful stage at a strictly
later time; duplicate stage times and failed stages fall back to the
previous root. Predictor values are stop-gradiented, so successful roots keep
the same implicit derivative — but with multiple algebraic roots a different
warm start can select a different branch, so secant assumes the continued
root is locally unique.

Every nonlinear root passes `(y, t, p)` to nlls-gram, whose implicit rule
differentiates the defining constraint,

$$
\dot z = -g_z^{-1}(g_y\dot y + g_t\dot t + g_p\dot p),
$$

rather than the LM iterations; the warm-start guess has zero derivative by
design. `args` is fixed data; put every differentiated quantity in `p`. On
the default bounded path, JVP, VJP, `vmap`, and reverse-over-forward compose
through the complete solve. `adaptive_loop="forward"` runs an actual-work
loop for adaptive Tsit5 and Rodas5P (primal, JVP, and nested forward mode
only), as for [ODEs](ode.md#static-shapes-and-saveat).

`sol.num_steps` counts logical time-step attempts including rejections;
`sol.num_root_solves` counts active nonlinear root calls (including the
initial consistency solve and failures) and `sol.num_root_steps` sums their
LM updates. All have exact-zero tangents. An adaptive stage-root failure
rejects the attempt and retries with a smaller step; a fixed-step failure
terminates. Either way `sol.ok` is false if `t_1` is not reached with valid
algebraic states.

For batched differentiation where a lane may fail or leave the model domain,
pass `failure_ad_reference=(y_ref, z_ref, t_ref, p_ref)` at a point where
the residual, context, and aux maps are finite and differentiable. It is
substituted into already-inactive root calls and inactive aux/field
evaluations so masked lanes cannot poison the JVP/VJP of successful lanes. A
newly attempted root must still be JVP-safe at its actual
`(y, z_guess, t, p)`; the reference never rescues an active failure, and a
failed lane's outputs are not a valid solution. Without an explicit
reference, an all-ones best-effort default is used.

## Auxiliary outputs

Saved output belongs to the differential field, and the algebraic function
may separately expose internal context:

```python
def g(y, z, t, args, p):
    return residual, algebraic_aux


def f(y, z, t, args, p, algebraic_aux):
    return dy, saved_aux
```

All four combinations are supported: neither aux, saved aux only, algebraic
aux only, or both. When algebraic aux is present, `f` must take the full
six-argument form. `has_aux` / `has_algebraic_aux` default to abstract
auto-detection; explicit `False` skips those traces.

Algebraic aux is internal cached context: it is passed to `f`, included in
the implicit derivative path, ignored by the nonlinear solver's residual
interface, and never stored or interpolated. It may be a nonempty pytree of
bool, integer, real, or complex arrays; every inexact leaf must be finite.
Only the differential field's real-floating `saved_aux` becomes `sol.aux`,
stored at accepted nodes and interpolated on requested grids. Invalid
algebraic context at initialization fails before any time-step work; invalid
saved aux in a prefix mode terminates at the previous accepted node, while
endpoint mode keeps the endpoint state with zero aux and `ok=False`.

## Rodas5P for DAEs

For the stiff path, tinydiffeq constructs the flattened mass-matrix system
internally:

$$
M\dot u=F(u,t), \qquad
u=(y,z), \quad M=\operatorname{diag}(I_y,0_z), \quad F=(f,g).
$$

```python
linearly_implicit = solve_semi_explicit_dae(
    f, g, Rodas5P(), 0.0, 1.0,
    jnp.asarray(1.0), jnp.asarray(0.5),
    p=jnp.asarray(2.0), dt_0=0.1,
    controller=IController(), max_steps=128,
)
```

`LMRootSolver` is used once, for initial consistency. Every later Rodas5P
stage solves $\left(M/(\gamma h) - F_u\right)k_i = r_i$ with one reused
pivoted LU factorization per attempted step — no nonlinear endpoint
restoration. Returned internal `z` values therefore satisfy the constraint
to the method's integration accuracy, not to `LMRootSolver.atol`, and the
solver reports one root call regardless of its attempt count. This
intentionally differs from RK4/Tsit5, which root-solve every stage; Rodas5P
is the choice when those algebraic solves dominate runtime or the coupled
dynamics are stiff. See [Stiff ODEs: Rodas5P](ode.md#stiff-odes-rodas5p) for
the method, its AD boundaries, and the SciML/Steinebach credit and links.

## Saving output

All `SaveAt` modes are supported:

- `SaveAt(t_1=True)` returns the endpoint.
- `SaveAt(steps=True)` returns the initial point and accepted internal steps
  as a padded `max_steps + 1` buffer with the usual `accepted` mask.
- `SaveAt(ts=grid)` uses cubic Hermite for root-restored methods and
  Rodas5P's stiff-aware continuous extension for `(y, z)`; aux uses cubic
  Hermite in both cases. No query-time nonlinear solves are performed.
  `exact=True` is not a DAE mode.

The result is a `DAESolution` with `ts`, `ys`, `zs`, `ok`, `num_accepted`,
`accepted`, `aux`, `num_steps`, `num_root_solves`, and `num_root_steps`.

### Dense output for root-restored RK4 and Tsit5

At a consistent knot, differentiating the constraint gives
$g_z\dot z = -(g_y\dot y + g_t)$. tinydiffeq solves this linear system once
per accepted knot (only when a query grid is requested), obtains `aux_dot`
by a JVP of the aux map along $(\dot y, \dot z, 1)$, and feeds values and
derivatives into the same normalized cubic Hermite basis used for ODE
states — an order-3 continuous extension with uniform error $O(h^4)$ when
`f` and `g` are $C^4$, $g_z$ stays uniformly nonsingular, and root error is
below the dense-output error. Interpolated `z` and aux are approximations:
away from knots the constraint defect is $O(h^4)$. Use `SaveAt(steps=True)`
when every returned row must be an actual converged root.

### Dense output for Rodas5P

Rodas5P stores the three coefficient pytrees of Steinebach's fourth-order
stiff-aware continuous extension and evaluates the same polynomial form as
[SciML's Rosenbrock interpolant](https://github.com/SciML/OrdinaryDiffEq.jl/blob/master/lib/OrdinaryDiffEqRosenbrock/src/rosenbrock_interpolants.jl)
for the combined `(y, z)` state — no $g_z$ factorization or nonlinear solve
at query times. Aux endpoint tangents come from the Rodas polynomial's
endpoint derivatives and a JVP of the aux map.

Knot selection and step sizes are differentiation-inert under the
frozen-controller convention; see
[AD through adaptive stepping](ode.md#ad-through-adaptive-stepping). If
`sol.ok` is false, neither outputs nor derivatives are a valid solution.

## Deliberate limits

Only the internally constructed constant block mass matrix
`diag(I_y, 0_z)` is supported; there is no general mass-matrix or fully
implicit residual API. Rodas5P uses dense Jacobians and dense pivoted LU.
Higher-index constraints and automatic index reduction are unsupported. This
is an initial-value solver: it does not determine unknown initial costates
or solve boundary-value problems, and jumps between multiple root branches
are not differentiable.
