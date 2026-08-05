# ODEs

`solve_ode` integrates `dx/dt = f(x, t, args, p)` from `t_0` to `t_1 > t_0`.
The state is an array or pytree whose leaves share one real floating dtype;
`args` is pass-through data (not an AD target by convention) and `p` holds
differentiable parameters. `dt_0` is required — there is no initial-step
heuristic.

## Solvers

| Solver | Order | Stepping |
|---|---|---|
| `Euler()` | 1 | fixed only |
| `RK4()` | 4 | fixed only |
| `Tsit5()` | 5(4) | fixed or adaptive (embedded estimate, FSAL) |
| `Rodas5P()` | 5(4) | fixed or adaptive, linearly implicit (stiff) |

Fixed stepping uses `ConstantStepSize()` (the default controller):
`dt_0 = (t_1 - t_0)/n` with `max_steps = n` reproduces a uniform grid
exactly. Times are formed arithmetically as `t_0 + i * dt_0` with a small
endpoint snap that never scales with `max_steps`, so increasing a nonbinding
attempt budget cannot change the numerical method. `unroll=` (a static int,
fixed stepping only) unrolls that many steps per scan iteration — identical
values, fewer GPU dispatches, more compile time; on small-batch
neural-network fields it cut reverse-mode solve time 2–3× on an L40S (see
`benchmarks/results/`).

Adaptive stepping uses `IController()` or `PIController()` with an embedded
error estimate. Omitted tolerances follow the state precision:

| State dtype | `rtol` | `atol` | unit-scale `dt_min` |
|---|---:|---:|---:|
| float32 | `1e-4` | `1e-6` | `10 * eps` ≈ `1.19e-6` |
| float64 | `1e-7` | `1e-9` | `10 * eps` ≈ `2.22e-15` |

Explicit values override the policy and are cast to the state dtype; use them
whenever tolerances are part of a reproducibility contract. Every attempt is
clipped to the remaining horizon, and the controller's next-step proposal is
computed from the clipped step — the clip doubles as the growth guard that
keeps near-flat fields from growing steps into quarter-horizon leaps.
`PIController` additionally damps step-size oscillations through the previous
accepted error ratio; `p_coeff=0, i_coeff=1` reproduces `IController` exactly.

## Static shapes and `SaveAt`

`max_steps` bounds attempted internal steps, including rejections. The
default `adaptive_loop="bounded"` runs a bounded `lax.scan` with exactly
`max_steps` attempt slots grouped into chunks that are skipped once the solve
completes; shapes are static, and changing tolerances, curvature, or initial
conditions never recompiles (pinned by `tests/test_recompile.py`).
`adaptive_loop="forward"` instead runs a dynamic `lax.while_loop` that
executes only actual attempts; it supports primal evaluation, JVP, and nested
forward mode, but JAX cannot transpose it, so reverse mode requires the
bounded loop. Under `vmap`, both run until the slowest lane finishes: the
bounded loop's skip conds are gated on a scalar all-lanes predicate whose
batching rule reduces over the batch axis, so the frozen tail after the
slowest lane is genuinely skipped rather than lowered to a both-branches
select — a vmapped adaptive solve costs actual attempts, not `max_steps`
(reverse mode still stores per-slot scan residuals, so keep budgets
realistic when differentiating).

Exactly one `SaveAt` mode is set:

- **`SaveAt(t_1=True)`** (default): the endpoint only.
- **`SaveAt(ts=grid)`**: dense interpolation onto a fixed query grid. Output
  shape is `(len(grid), ...)` regardless of how many internal steps the
  controller takes; `ts` is a data leaf, so a new grid of the same length
  retraces nothing. Explicit methods use cubic Hermite (4th-order accurate
  between 5th-order knots); Rodas5P uses its stiff-aware fourth-order
  continuous extension. Queries are observation times, not internal stops.
  For an explicit constant-step solve, `exact=True` instead requires every
  query to coincide with a realized knot and gathers those states directly,
  with no interpolation work.
- **`SaveAt(steps=True)`**: the initial state plus accepted steps as a
  contiguous prefix of a `max_steps + 1` buffer. Rejected attempts are
  omitted; `fill="last"` (default) repeats the last valid row through the
  tail, `fill="inf"` fills it with `inf`, and `sol.accepted` masks the valid
  prefix.

If the budget runs out before `t_1`, `sol.ok` is `False` and the outputs hold
the reached prefix. Nothing is poisoned; callers that want diverging
residuals map `jnp.where(sol.ok, x, jnp.inf)`. `sol.num_steps` counts
attempts, `sol.num_accepted` counts advances.

### Residuals on adaptive output

Collocation-style residuals evaluated on an adaptive
`SaveAt(steps=True)` rollout keep a static shape by evaluating the pointwise
residual on **every** padded row and zeroing the tail with the `accepted`
mask. With the default `fill="last"`, padded rows repeat the last accepted
state, so the residual there is finite wherever the endpoint is — a single
`where` is safe for both values and gradients (no double-`where` needed):

```python
sol = jax.vmap(solve_one)(x_0s)            # SaveAt(steps=True), (B, rows, ...)
rows = pointwise_residual(sol.xs, p)       # evaluated on all rows in parallel
masked = jnp.where(sol.accepted, rows, 0.0)
count = jax.lax.stop_gradient(sol.accepted.sum())      # actual points, inert
residual = jnp.where(
    sol.ok[:, None], masked / jnp.sqrt(count.astype(masked.dtype)), jnp.inf
).reshape(-1)                              # static length B * rows
```

Zero rows contribute nothing to a least-squares objective, so this is
exactly the fixed-shape analogue of dropping rejected steps. Two properties
to choose deliberately: normalizing by the accepted count makes the loss a
mean over actual collocation points but changes discontinuously when the
mesh changes, and lanes with more accepted steps contribute more rows — the
adaptive mesh curvature-weights the collocation. When the residual
definition needs fixed sample times and a smoother parameter dependence,
use `SaveAt(ts=fixed_grid)` instead: no mask, no mesh-dependent weighting,
and the output times carry none of the frozen-mesh ambiguity of adaptive
internal knots.

`project` (an idempotent clamp, e.g. positivity) is applied at every point
where the field is evaluated and to every accepted state.

## AD through adaptive stepping

The bounded loop supports `jax.grad`, `jax.jvp`, `vmap`, and
reverse-over-forward through one adaptive solve, verified against closed
forms in `tests/test_ad.py`.

The step-size controllers compute their error norms, decisions, and
next-step factors inside `stop_gradient`: accept/reject is a discrete branch,
and `E**(-1/order)` is singular at the exact-zero error of a flat-start
policy — without the stop-gradient and the machine-epsilon error floor the
backward pass is NaN from iteration one.

States therefore differentiate through the solver stages on a **frozen
mesh**: the derivative holds realized step sizes and accept/reject patterns
fixed and omits the mesh-motion term `(dq/dt) * dt_k/dtheta`. Near a smooth
residual root that term is `O(‖r‖)`, so a frozen-mesh residual Jacobian is
asymptotically exact there; far from a root it can be material. Endpoint and
fixed-requested-grid outputs do not have a moving output-time axis; adaptive
`SaveAt(steps=True)` times have exactly zero tangent.

`SaveAt(ts=...)` brackets queries with `searchsorted` — integer indices, no
gradient. Values differentiate through the bracketing states and slopes;
zero-width brackets are handled with the double-`where` trick so neither the
primal nor its transpose evaluates `0/0`.

When several tangent or cotangent directions are needed at one primal point,
cache the linearization (`jax.linearize` / `jax.vjp`) and `vmap` the
pushforward or pullback instead of repeating the primal trajectory in a fused
`vmap(jvp)`.

On float32 GPUs, XLA serves `dot_general` from TF32 tensor cores by default
(~1e-3 precision). When adaptive accept/reject decisions depend on
neural-network outputs, that roundoff can move the mesh; set
`jax_default_matmul_precision="highest"` in reproducibility entry points.
tinydiffeq never changes application-global JAX settings, including
`jax_enable_x64` — a float32 problem stays float32 even with x64 enabled.

## Auxiliary outputs

The field may return `(dx, aux)` to save quantities it already computes
without adding them to the state:

```python
def f(x, t, args, p):
    dx = -p * x
    return dx, {"flow": dx, "moment": x**2}
```

Saved aux is a nonempty pytree of real floating arrays (leaves may differ in
dtype). It follows `SaveAt` — stored at the endpoint or accepted nodes, or
interpolated onto a requested grid with endpoint slopes obtained by JVP — and
participates in JVP/VJP. `has_aux=None` (default) detects the contract with
one `jax.eval_shape` trace; `has_aux=False` skips detection and selects the
minimal value-only path. Every saved aux leaf must remain finite: an invalid
value in a prefix-saving mode terminates at the previous accepted node, while
endpoint mode keeps the endpoint state, zero-fills aux, and sets `ok=False`.
For batched differentiation where inactive lanes may leave the model domain,
`failure_ad_reference=(x, t, p)` supplies a finite point used only for safe
linearization.

## Stiff ODEs: Rodas5P

`Rodas5P()` is an eight-stage, fifth-order Rosenbrock–Wanner method with an
embedded estimator and a stiff-aware fourth-order continuous extension. It is
A-stable and stiffly accurate.

```python
import jax.numpy as jnp

from tinydiffeq import IController, Rodas5P, solve_ode


def stiff_field(x, t):
    # Exact solution x(t) = cos(t), with a fast transient eigenvalue -1000.
    return -1000.0 * (x - jnp.cos(t)) - jnp.sin(t)


sol = solve_ode(
    stiff_field, Rodas5P(), 0.0, 1.0, jnp.asarray(1.0),
    dt_0=0.01, controller=IController(), max_steps=512,
)
```

At each attempted step it forms the exact JAX Jacobian $J = \partial f /
\partial x$, factors $W = I/(\gamma h) - J$ once with pivoted LU, and reuses
the factors for all eight stage solves — linear solves only, no Newton
iteration. JVP/VJP propagate through Jacobian construction and the stage
solves via the linear-solve rule $dx = A^{-1}(db - dA\,x)$; pivot selection
is not differentiated. Dense Jacobians and dense LU only — use SciML or
diffrax for sparse/Krylov stages.

### Credit

Rodas5P was constructed by Gerd Steinebach and introduced in
[Steinebach (2023)](https://doi.org/10.1007/s10543-023-00967-x). tinydiffeq's
JAX implementation deliberately follows SciML's authoritative MIT-licensed
implementation — the
[`OrdinaryDiffEqRosenbrock` package](https://github.com/SciML/OrdinaryDiffEq.jl/tree/master/lib/OrdinaryDiffEqRosenbrock),
its
[tableau and dense coefficients](https://github.com/SciML/OrdinaryDiffEq.jl/blob/master/lib/OrdinaryDiffEqRosenbrock/src/rosenbrock_tableaus.jl),
[step implementation](https://github.com/SciML/OrdinaryDiffEq.jl/blob/master/lib/OrdinaryDiffEqRosenbrock/src/rosenbrock_perform_step.jl),
and
[dense interpolants](https://github.com/SciML/OrdinaryDiffEq.jl/blob/master/lib/OrdinaryDiffEqRosenbrock/src/rosenbrock_interpolants.jl).
The tableau, stage equations, embedded estimate, and continuous extension are
ported rather than redesigned, and regression tests compare fixed steps
against SciML-produced values to prevent silent divergence.
