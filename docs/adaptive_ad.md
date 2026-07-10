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

The states themselves remain fully differentiable through the RK stages: the
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

## What to watch

- Reductions over the valid prefix of `SaveAt(steps=True)` are
  **discontinuous** in the inputs: its length changes when an accept flips to
  a reject. Use `sol.accepted` when padding must not contribute. Because the
  default tail repeats the endpoint, `xs[-1]` remains the reached final state.
- Finite-difference checks of adaptive solves are noisy for the same reason;
  compare AD against closed forms or use fixed-step solvers for FD tests.
