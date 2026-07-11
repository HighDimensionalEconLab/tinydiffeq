# Adaptive Stepping and AD

tinydiffeq's central claim is that one adaptive solve is safely
differentiable in forward mode, reverse mode, and reverse-over-forward. That
holds because of three deliberate choices about **what is not
differentiated**.

## Default tolerances follow the state precision

`IController()` and `PIController()` resolve omitted tolerances from the
state dtype at trace time:

| State/time dtype | `rtol` | `atol` | unit-scale `dt_min` |
|---|---:|---:|---:|
| float32 | `1e-4` | `1e-6` | `10 * eps` ≈ `1.19e-6` |
| float64 | `1e-7` | `1e-9` | `10 * eps` ≈ `2.22e-15` |

The policy never changes `jax_enable_x64`: a float32 `x_0` stays float32 even
when x64 is enabled globally. Explicit tolerances override the defaults and
are cast to the state dtype. Use explicit values whenever tolerances are part
of a reproducibility or accuracy contract.

Automatic `dt_min` is `10 * eps * max(1, abs(t_1))`; set it explicitly when
the relevant time scale differs materially from the absolute horizon. The
controller also floors exact-zero error ratios at machine epsilon before
applying a negative power; the growth-factor clip then selects maximal step
growth without introducing infinities.

## The controller is stop-gradiented

The adaptive step-size controllers (`IController` and `PIController`) compute
their scaled error norms and next-step factors inside `stop_gradient`:

> accept/reject is a non-differentiable branch either way, the gradient of
> `E**(-1/order)` blows up at the exact-zero error of a flat-start policy,
> and the `d(dt)/dtheta` term only slides sample points along the visited
> trajectory — irrelevant to a residual that must vanish at every state.

The states themselves remain fully differentiable through the solver stages: the
gradient you get is the derivative of the numerical flow map *for the step
pattern actually taken*. This is the same convention diffrax uses.

The `E**(-1/5)` blow-up is not hypothetical: a policy initialized flat gives
an exactly-zero error estimate on the first step, and without the
`stop_gradient` (plus the machine-epsilon error floor) the backward pass is NaN
from iteration one. `tests/test_ad.py::test_grad_finite_on_flat_field` pins
this.

`PIController` additionally carries the previous accepted error ratio. Its
step-size factor is

```text
safety * E_n**(-(p_coeff + i_coeff) / order)
       * E_prev**(p_coeff / order)
```

and `E_prev` changes only after acceptance. The whole recurrence is
controller-internal and stop-gradiented. Setting `p_coeff=0, i_coeff=1`
reproduces `IController` bit for bit; the default `p_coeff=0.4, i_coeff=0.3`
is less sensitive to oscillatory error estimates and typically rejects fewer
attempts on harder problems.

## The horizon clip is the growth guard

Every attempt is clipped so it cannot step past `t_1`, and — deliberately
unlike diffrax — the controller's next-step proposal is computed from the
**clipped** step:

> the horizon clip doubles as the guard on step growth: without it, a
> near-flat vector field lets steps quintuple into quarter-horizon leaps
> whose Gauss–Newton linearization stalls a trust-region optimizer
> differentiating through the rollout.

With `factor_max = 5`, a flat field would otherwise reach `dt ≈ t_1/4` within
a few accepted steps; residuals sampled from three or four giant steps make
the optimizer's linear model useless. Clipping first means the proposal can
never exceed `factor_max × remaining horizon`.

One refinement over a bare `min(dt, remaining)`: when the remaining horizon
is within `max_steps × eps` of the desired step, the step is stretched to
land on `t_1` exactly. Summing `n` rounded steps of `(t_1 - t_0)/n` can leave
`t` one accumulated ulp short of `t_1`, and without the stretch that sliver
would cost an extra iteration a `max_steps = n` budget doesn't have.

## Interpolation knots are non-differentiable

`SaveAt(ts=...)` brackets each query with `searchsorted` — integer indices,
no gradient. This is consistent with the stop-gradiented controller: the
term excluded is again "the knots slide along the trajectory as parameters
change". Values differentiate fully through the bracketing states and
derivatives (`xL`, `xR`, `fL`, `fR`).

Zero-width brackets (duplicate rows from rejections and the frozen tail) use
the **double-where trick**: the divisor is replaced by 1 *before* dividing,

```python
width_safe = jnp.where(degenerate, 1.0, width)
s = jnp.clip((tau - t_left) / width_safe, 0.0, 1.0)
value = jnp.where(degenerate, x_left, hermite(s, ...))
```

so neither the primal nor its jvp/vjp ever evaluates `0/0`. A single `where`
on the output is not enough — reverse mode differentiates both branches, and
`NaN * 0 = NaN`.

## What this buys you

- `jax.grad`, `jax.jvp`, and `jax.grad(jax.jvp(...))` (the
  Levenberg–Marquardt geodesic-acceleration pattern) all work through
  adaptive solves and interpolated output, verified against closed forms in
  `tests/test_ad.py`.
- `jax.vmap` over `x_0` or `p` gives genuinely per-lane adaptivity: each lane
  accepts/rejects independently through the masked scan, and batched results
  equal the individual solves exactly.

## Reusing a linearization

A single `jax.jvp` must evaluate the primal trajectory and propagate its
tangent together. If several tangent directions are needed at the same
`x_0`, `p`, and other primal inputs, cache the linearization instead:

```python
value, pushforward = jax.linearize(endpoint, x_0)
tangent_batch = jax.jit(jax.vmap(pushforward))(directions)

value, pullback = jax.vjp(endpoint, x_0)
cotangent_batch = jax.jit(
    jax.vmap(lambda cotangent: pullback(cotangent)[0])
)(cotangents)
```

The setup computes and stores residuals for that one primal point. Reuse the
pushforward or pullback only while every primal input is unchanged; otherwise
linearize again. This is particularly important for Rodas5P and implicit DAE
roots: their custom linear-solve rules retain primal LU/Cholesky factors, so
multiple directions reuse those factors rather than refactoring independently.

On the 256-state fixed Tsit5 benchmark, cached pushforwards were about 8–18%
faster than a fused `vmap(jvp)` on CPU after excluding setup. On the RTX 3090,
cached pushforwards and pullbacks were roughly 2.2–2.5x faster for 1–16
directions because the primal trajectory was not replicated across mapped
lanes. For one direction at a new primal point, ordinary `jax.jvp` or
`jax.vjp` remains the right interface.

## What to watch

- Reductions over the valid prefix of `SaveAt(steps=True)` are
  **discontinuous** in the inputs: its length changes when an accept flips to
  a reject. Use `sol.accepted` when padding must not contribute. Because the
  default tail repeats the endpoint, `xs[-1]` remains the reached final state.
- Finite-difference checks of adaptive solves are noisy for the same reason;
  compare AD against closed forms or use fixed-step solvers for FD tests.

## Custom-rule audit

tinydiffeq uses hand-coded derivative boundaries only where they remove
iteration or factorization work without changing the mathematical derivative:

- Rodas5P's factored linear solve has a custom JVP
  \(\delta x=A^{-1}(\delta b-\delta A\,x)\). Every stage tangent and the
  transposed VJP reuse the attempt's pivoted LU factors; pivot selection is not
  differentiated.
- Semi-explicit DAE roots use the implicit-function rule
  \(\delta z=-g_z^{-1}(g_y\delta y+g_t\delta t+g_p\delta p)\). The exact
  constraint Jacobian generally cannot reuse the primal LM factor because the
  latter is damped and may be based on an earlier iterate. A cached
  `jax.linearize`/`jax.vjp` does reuse the implicit factor across directions.
- nlls-gram's public solve boundary already supplies implicit Cholesky/CG
  derivative rules, including aux derivatives. tinydiffeq does not
  differentiate its optimizer iterations.
- Dense linear exponential actions use a Fréchet custom JVP for active matrix
  or time tangents and reuse the matrix exponential when only the initial state
  varies.
- Matrix-free terminal exponential sensitivities expose
  `jvp_linear_ode`/`vjp_linear_ode`, applying the forward or transposed
  exponential directly instead of differentiating Arnoldi orthogonalization.
- `AdaptiveKrylovExponential` uses a bounded scan and residual-controlled
  internal slices. Ordinary AD follows the realized controller path; the
  hand-coded initial-state rules apply independent adaptive forward or
  transposed actions and avoid differentiating the Arnoldi basis.

Explicit Runge--Kutta, Euler--Maruyama, Hermite/Rodas interpolation, and
cumulative trapezoids remain ordinary JAX programs. Their recurrences and
polynomials already transpose efficiently, and a custom rule would either
duplicate JAX's work or introduce a different continuous-adjoint derivative.
