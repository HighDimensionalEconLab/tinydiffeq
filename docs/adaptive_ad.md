# Adaptive Stepping and AD

tinydiffeq's central claim is that one adaptive solve is safely
differentiable in forward mode, reverse mode, and reverse-over-forward. That
holds because of three deliberate choices about **what is not
differentiated**.

## The controller is stop-gradiented

The step-size controller (`IController`) computes its scaled error norm and
next-step factor inside `stop_gradient`:

> accept/reject is a non-differentiable branch either way, the gradient of
> `E**(-1/order)` blows up at the exact-zero error of a flat-start policy,
> and the `d(dt)/dtheta` term only slides sample points along the visited
> trajectory — irrelevant to a residual that must vanish at every state.

The states themselves remain fully differentiable through the RK stages: the
gradient you get is the derivative of the numerical flow map *for the step
pattern actually taken*. This is the same convention diffrax uses.

The `E**(-1/5)` blow-up is not hypothetical: a policy initialized flat gives
an exactly-zero error estimate on the first step, and without the
`stop_gradient` (plus the `max(E, 1e-12)` floor) the backward pass is NaN
from iteration one. `tests/test_ad.py::test_grad_finite_on_flat_field` pins
this.

## The horizon clip is the growth guard

Every attempt is clipped so it cannot step past `t1`, and — deliberately
unlike diffrax — the controller's next-step proposal is computed from the
**clipped** step:

> the horizon clip doubles as the guard on step growth: without it, a
> near-flat vector field lets steps quintuple into quarter-horizon leaps
> whose Gauss–Newton linearization stalls a trust-region optimizer
> differentiating through the rollout.

With `factormax = 5`, a flat field would otherwise reach `dt ≈ t1/4` within
a few accepted steps; residuals sampled from three or four giant steps make
the optimizer's linear model useless. Clipping first means the proposal can
never exceed `factormax × remaining horizon`.

One refinement over a bare `min(dt, remaining)`: when the remaining horizon
is within `max_steps × eps` of the desired step, the step is stretched to
land on `t1` exactly. Summing `n` rounded steps of `(t1 - t0)/n` can leave
`t` one accumulated ulp short of `t1`, and without the stretch that sliver
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
- `jax.vmap` over `x0` or `p` gives genuinely per-lane adaptivity: each lane
  accepts/rejects independently through the masked scan, and batched results
  equal the individual solves exactly.

## What to watch

- Reductions over `SaveAt(steps=True)` rows are **discontinuous** in the
  inputs: the number of duplicate rows changes when an accept flips to a
  reject. Consume steps mode with residuals that vanish at every state (so
  duplicates are harmless), or reduce to `xs[-1]`.
- Finite-difference checks of adaptive solves are noisy for the same reason;
  compare AD against closed forms or use fixed-step solvers for FD tests.
