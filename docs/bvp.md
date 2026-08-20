# Boundary Value Problems

`solve_bvp` solves two-point boundary value problems of the form

$$
\frac{dy}{dt} = f(t, y, z, \mathrm{args}, p) + \frac{S\,y}{t - t_a},
\qquad
\mathrm{bc}\big(y(t_a),\, y(t_b),\, z, \mathrm{args}, p\big) = 0,
$$

on $t \in [t_a, t_b]$. It is a faithful JAX port of
[`scipy.integrate.solve_bvp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_bvp.html):
the same 4th-order Lobatto IIIA collocation, the same damped Newton method
with an affine-invariant criterion, the same 5-point Lobatto residual
estimator and insert-1/insert-2 mesh refinement, and the same constants and
default tolerances. The optional singular term ($S$ an $n \times n$ matrix on
the flattened state, requiring $S\,y(t_a) = 0$) covers Lane–Emden style
problems.

## `z` versus `p`

The one distinction to internalize:

- **`z` are scipy's *unknown parameters*** — solved jointly with $y$
  (an eigenvalue, a free constant chosen by an extra boundary condition).
  You pass a guess `z_0` (any pytree) and read the solved value from
  `sol.z`. When present, `bc` must return `n + size(z)` residuals. Like
  every guess, `z_0` is differentiation-inert.
- **`p` are *known differentiable parameters*** — the only AD input, as
  everywhere else in tinydiffeq. JVP and VJP rules differentiate `sol.y`,
  `sol.yp`, `sol.z`, and `sol.aux` with respect to `p` implicitly at the
  solution, never through the iterations.
- **`args` is inert pass-through data.**

scipy names the unknowns `p`; they are renamed here because in tinydiffeq
`p` always means the differentiable input. The solved unknowns are exactly
the DAE interface's unknown-with-a-guess role, hence `z`.

## Interface

```python
solve_bvp(fun, bc, t, y_0, z_0=None, *, p=None, args=None, S=None,
          fun_jac_ad="auto", bc_jac_ad="auto", tol=1e-3, bc_tol=None,
          max_nodes=128, has_aux=None)
```

`fun` and `bc` are **pointwise** — a scalar `t` and a single node's state
pytree, vmapped over the mesh internally (scipy instead passes the whole
`(n, m)` mesh; port scipy code by deleting the vectorization). Both may take
two to five positional arguments, always in this order:

```python
fun(t, y) | fun(t, y, z) | fun(t, y, z, args) | fun(t, y, z, args, p)
bc(ya, yb) | bc(ya, yb, z) | bc(ya, yb, z, args) | bc(ya, yb, z, args, p)
```

`t` is the initial mesh (scipy's `x`): strictly increasing, at least two
nodes, at most `max_nodes`. `y_0` is the initial guess — any pytree with
leading axis `len(t)` on every leaf and one shared real floating dtype,
which becomes the working dtype. `fun` may return `(value, aux)`; aux is
evaluated once at the solution and participates in AD.

```python
import jax.numpy as jnp
from tinydiffeq import solve_bvp

# Sturm–Liouville: y'' = -z^2 y, y(0) = y(pi) = 0, y'(0) = z.
def fun(t, y, z):
    return jnp.array([y[1], -z[0]**2 * y[0]])

def bc(ya, yb, z):
    return jnp.array([ya[0], yb[0], ya[1] - z[0]])

t = jnp.linspace(0.0, jnp.pi, 5)
sol = solve_bvp(fun, bc, t, jnp.ones((5, 2)), jnp.array([0.5]))
sol.z           # the eigenvalue, ~1.0
sol.num_nodes   # active nodes on the refined mesh
```

## Static shapes

The mesh grows under refinement, so every returned array is padded to the
static `max_nodes` (default 128 — smaller than scipy's 1000; every loop and
the factorization run over all `max_nodes` padded intervals, so cost grows
linearly in the budget):

- `sol.t` has shape `(max_nodes,)`; entries past `sol.num_nodes` repeat
  $t_b$ exactly.
- `sol.y` and `sol.yp` leaves have leading axis `max_nodes`; tail rows
  repeat the last active row bitwise. `sol.yp` holds the (singular-term
  corrected) right-hand side at the nodes.
- `sol.rms_residuals` has shape `(max_nodes - 1,)` and is exactly zero on
  inactive intervals.
- `sol.aux` rows past the last active node duplicate the endpoint value.

`max_nodes` is static; changing it recompiles, and `fun` and `bc` key the
compilation cache by object identity — define them at module scope rather
than rebuilding closures at a hot call site.
Everything else — mesh values, guesses, `p`, `args`, `S` values, `tol`, and
`bc_tol` — is traced data and never retraces. The compiled solve is also
shared across initial mesh lengths (inputs are padded to `max_nodes` before
dispatch), though under an outer `jit` a changed input length retraces that
outer function, as any shape change does.

## Dense output

There is no callable solution object. The padded tails are what make the
plain arrays sufficient: `hermite_interpolate(ts, sol.t, sol.y, sol.yp)`
evaluates **exactly** the C1 cubic spline scipy returns as `sol.sol(ts)`
(scipy's `create_spline` is the cubic Hermite interpolant of `(y, yp)`),
and `hermite_derivative(ts, sol.t, sol.y, sol.yp)` is `sol.sol(ts, 1)`.
Queries outside $[t_a, t_b]$ clamp to the endpoint values (scipy's `PPoly`
extrapolates the cubic instead); derivatives outside the span are zero.

## Statuses and failure

The solve never raises inside traced code: failures are reported as data.
A failed status returns the last iterate, which — as in scipy — may be
non-finite when the final Newton candidate diverged; check `sol.ok` before
trusting values. `sol.status` carries scipy's codes:

- `0` — converged to the desired accuracy (`sol.ok`).
- `1` — the refinement wanted more than `max_nodes` nodes; the reported
  mesh and solution are the last completed iteration's.
- `2` — a singular collocation Jacobian; detected as a non-finite or
  exactly rank-deficient LU factor (scipy's `splu` raises here), with the
  last iterate returned.
- `3` — the boundary-condition tolerance was not satisfied within 10
  iterations after the mesh stopped refining.

`tol` is floored at `100 * eps` of the working dtype silently (scipy warns;
`tol` may be a tracer here). In float32 that floor is ~1.2e-5, so tighten
tolerances only as far as the dtype supports.

## AD contract

The whole solve sits behind one `custom_jvp`: the implicit function theorem
applied to the collocation system $F(Y, z; p) = 0$ on the frozen final mesh,
using the same assembled Jacobian the Newton method factors. Reverse mode is
JAX's transposition of that rule — there is no separate VJP rule, and the
iteration count, damping, and mesh are never differentiated through.

- Only `p` is an AD input. `sol.y`, `sol.yp`, `sol.z`, and `sol.aux` carry
  tangents; `sol.t`, `sol.rms_residuals`, and every counter and status are
  differentiation-inert with exact-zero tangents.
- The guesses `t`, `y_0`, `z_0` and the inert `args` and `S` have exact-zero
  gradients by contract. A `p`-dependent singular term belongs inside `fun`.
- Higher-order derivatives (hessians, reverse-over-forward) are exact on the
  frozen mesh: the rule leaves the solution and Jacobian differentiable, so
  outer transforms recurse through the same implicit rule.
- A failed solve (`status != 0`) has exact-zero, finite tangents; under
  `vmap`, a failed lane's tangent program is evaluated at the inert initial
  guess so it cannot poison successful lanes. The one loud exception: a
  *converged* solve whose final-mesh Jacobian fails to refactor inside the
  AD rule has no computable derivative and reports NaN tangents,
  lane-locally.
- Wrap gradient computations in `jax.jit` — op-by-op assembly and
  factorization of the collocation Jacobian is an order of magnitude slower.
- To differentiate with respect to the endpoints $t_a, t_b$, rescale the
  problem to a fixed interval and put the endpoints in `p`.

## Jacobians and the linear solve

Local Jacobians of `fun` and `bc` come from AD, not scipy's forward
differences — `fun_jac_ad` and `bc_jac_ad` select `"jvp"` (`jacfwd`),
`"vjp"` (`jacrev`), or `"auto"` (forward when square or tall, reverse when
strictly fat, block by block). There is no analytic-Jacobian argument. The
finite-difference parameter Jacobians are the piece of scipy most prone to
pushing a marginal collocation system singular; AD removes that failure
mode.

The collocation Jacobian is bordered almost block diagonal — a staircase of
`n`-square blocks coupling adjacent nodes, a dense column border for `z`,
and boundary rows tying the two endpoints — a structure fixed by the
discretization, not the problem. Where scipy hands the assembled sparse
matrix to SuperLU, each Newton refresh here runs a structured orthogonal
factorization (`tinydiffeq.babd`): cyclic reduction eliminates all pair
midpoints per level through one batched complete QR, ~`log2(max_nodes)`
batched calls in total, leaving one dense `(2n + size(z))`-square boundary
system. Factorization costs `O(max_nodes n^3)` and each solve
`O(max_nodes n^2)`, so padding to the static `max_nodes` is nearly free, and
orthogonal eliminations are stable on saddle-path dichotomies where naive
condensation overflows. The factorization is reused across the backtracking
line search and fixed-Jacobian iterations, exactly as scipy reuses its
`splu` object, and the AD rule reuses it through `lax.custom_linear_solve`
with its transpose solve.

On GPUs in float32, set
`jax.config.update("jax_default_matmul_precision", "highest")` as for the
[ODE solvers](ode.md#stiff-odes-rodas5p): the Jacobian assembly and the
singular-term products are matmuls that XLA otherwise serves from TF32.

The whole solve is compiled, with `lax.while_loop` outer, Newton, and
backtracking loops. For repeated solves, put the `solve_bvp` call inside
your own `jax.jit` (or `vmap`): calling it from un-jitted Python re-runs
the wrapper's validation, flattening, and dispatch every call — a few
milliseconds that dwarf a small compiled solve — while inside a jitted
function that work happens once at trace time and warm calls run at
compiled speed (faster than scipy even on 5-node problems; see
`benchmarks/results/`).

## Credit and deviations

The algorithm is a direct port of
[scipy's `_bvp.py`](https://github.com/scipy/scipy/blob/v1.18.0/scipy/integrate/_bvp.py)
(BSD-3), which implements the residual-control collocation method of
Kierzenka and Shampine, *A BVP Solver Based on Residual Control and the
MATLAB PSE* (ACM TOMS 27(3), 2001), with the damped Newton method of
Ascher, Mattheij, and Russell, *Numerical Solution of Boundary Value
Problems for ODEs* (SIAM, 1995). Same collocation residuals, Jacobian
blocks, Newton constants, Lobatto quadrature weights, refinement thresholds,
and status semantics; regression tests cross-check meshes, iterates, and
residuals against scipy run with analytic Jacobians.

Deliberate deviations: AD local Jacobians replace `fun_jac`/`bc_jac` and
the finite-difference estimators; `fun`/`bc` are pointwise; scipy's unknown
parameters are `z` and the differentiable parameters `p`; outputs are
padded to a static `max_nodes` (default 128, not 1000); statuses are data
rather than exceptions and there is no `verbose`; the `tol` floor is
silent; real dtypes only (split complex problems into real and imaginary
parts); no dense-output object — reuse `hermite_interpolate`; extrapolation
clamps. Node removal is not implemented, as in scipy.
