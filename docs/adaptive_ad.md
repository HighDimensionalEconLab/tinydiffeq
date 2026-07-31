# Adaptive Stepping and AD

The default `adaptive_loop="bounded"` supports forward mode, reverse mode, and
reverse-over-forward through one adaptive solve. The alternative
`adaptive_loop="forward"` executes a data-dependent number of attempts and
supports primal evaluation, JVP, and nested forward mode, but not ordinary
reverse mode. Both paths use the same deliberate frozen-controller derivative
convention described below.

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
their scaled error norms, decisions, and next-step factors inside
`stop_gradient`. Accept/reject is a discrete branch, and differentiating
`E**(-1/order)` is singular at the exact-zero error of a flat-start policy.

States remain differentiable through solver stages, but the derivative holds
the realized step sizes and accept/reject pattern fixed. It is the derivative
of the discrete flow on a **frozen mesh**, not the total derivative of a mesh
that moves with parameters. In particular, for parameter-only differentiation,
the returned adaptive knot times in `SaveAt(steps=True)` have exactly zero
tangent.

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
reproduces `IController` bit for bit.

`IController` remains the default. On the smooth ODE and DAE screening
problems, the default PI coefficients did not reduce rejections, and
matched-work accuracy was comparable or favored I, except that PI sometimes
improved common-grid ODE interpolation. PI remains useful as an opt-in for
genuinely rejection-prone or step-size-oscillatory problems; those cases were
not covered by this screen.

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

Adaptive steps use `min(dt, remaining)` and are never enlarged to consume a
floating-point sliver. Fixed-step times are instead formed arithmetically as
`t_0 + i * dt_0`, avoiding drift from repeated addition, with a small local
endpoint snap capped below a quarter step. No time tolerance scales with
`max_steps`: changing a nonbinding attempt budget must not stretch a step or
change the numerical method.

## Frozen mesh versus a moving-mesh derivative

Let a parameter-dependent controller choose an internal knot
\(t_k(\theta)\), and let \(q(x,t,\theta)\) be a saved quantity. Along the exact
trajectory, its total derivative contains

$$
\frac{d q_k}{d\theta}
=q_x\left(x_\theta\rvert_t+f\,t_{k,\theta}\right)
 +q_t t_{k,\theta}+q_\theta.
$$

tinydiffeq's frozen-mesh rule returns the terms holding \(t_k\) fixed. The
omitted mesh-motion term is

$$
\left(q_t+q_x f\right)t_{k,\theta}
=\frac{d q}{dt}\,t_{k,\theta}.
$$

For a collocation residual \(r(x_k,t_k,\theta)\), this omission is the total
trajectory derivative \((d r/dt)t_{k,\theta}\). If the learned policy approaches
a smooth root for which the residual vanishes along the trajectory, then
\(d r/dt\) vanishes with it locally; under that regularity, the omitted term is
\(O(\lVert r\rVert)\), so the frozen-mesh residual Jacobian is asymptotically
exact near the root. Pointwise cancellation at a few knots alone is not enough
for that conclusion.

Far from a root the omitted term can be material, so a Gauss--Newton or LM
predicted reduction and trust ratio may be misleading. A targeted
finite-difference audit found that adaptive `SaveAt(steps=True)` state
derivatives need not approach the moving-mesh derivative under tolerance
refinement. Endpoint output at fixed `t_1` and fixed requested-grid output do
not have this particular moving-output-time ambiguity and did converge in the
same audit.

## Interpolation knots are non-differentiable

`SaveAt(ts=...)` brackets each query with `searchsorted` — integer indices,
no gradient. Values differentiate through the bracketing states and
derivatives (`xL`, `xR`, `fL`, `fR`) while internal knot locations remain
frozen. The requested times themselves are fixed outputs, unlike
`SaveAt(steps=True)`'s adaptive internal times.

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

- With the default bounded loop, `jax.grad`, `jax.jvp`, and
  `jax.grad(jax.jvp(...))` (the
  Levenberg–Marquardt geodesic-acceleration pattern) all work through
  adaptive solves and interpolated output, verified against closed forms in
  `tests/test_ad.py`.
- With `adaptive_loop="forward"`, primal evaluation, `jax.jvp`, and nested
  forward mode work; `jax.vjp`, `jax.grad`, and reverse-over-forward do not.
- `jax.vmap` over `x_0` or `p` gives genuinely per-lane adaptivity: each lane
  accepts/rejects independently. Execution continues until the slowest lane
  finishes.

The dynamic loop is most useful when actual attempts are much fewer than
`max_steps`. When an integration uses most of its budget, primal performance
can be similar or slower than the bounded scan; compilation and forward-mode
paths may still improve. Choose from representative end-to-end measurements,
not from the loop form alone.

On float32 GPUs, different loop lowerings can also fuse neural-network matrix
products differently. With accelerator-default reduced-precision matrix
multiplication, that roundoff was large enough in one width-32 policy test to
move the adaptive mesh even though both loop contracts were correct. Setting
`jax_default_matmul_precision="highest"` restored matching paths. tinydiffeq
does not change this application-global JAX setting; set it in reproducibility
entry points when adaptive decisions depend on neural-network outputs.

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
linearize again. Rodas5P's custom linear-solve rule explicitly retains and
reuses its primal pivoted LU factors. Semi-explicit DAE roots have a different
boundary: nlls-gram supplies both the primal LM root and its implicit derivative,
selecting direct `LU()` by default for a square residual system. The primal LM
Cholesky factors are not reused because they represent damped normal equations,
not the accepted root's nonsymmetric Jacobian. `jax.linearize` still avoids
repeating the primal trajectory, but the API makes no promise that the DAE
root's direct factorization is cached across pushforward directions.

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
- Adaptive `SaveAt(steps=True)` derivatives hold the internal times fixed and
  omit mesh motion. Use a fixed requested grid when the residual definition
  requires fixed sample times, or treat the frozen-mesh Jacobian explicitly as
  a near-root approximation.
- Finite-difference checks of adaptive solves are noisy for the same reason;
  compare AD against closed forms or use fixed-step solvers for FD tests.

## Custom-rule audit

tinydiffeq uses hand-coded derivative boundaries where they enforce the intended
mathematical derivative or avoid differentiating iteration/factorization work:

There is no custom reverse rule that replays an adaptive solve on a recorded
mesh. The bounded path uses ordinary traced AD through its scan, subject to the
controller `stop_gradient`; the dynamic forward path deliberately exposes
JAX's no-reverse boundary. A replay/custom-VJP design would be a distinct future
API and would not restore the omitted moving-mesh term by itself.

- Rodas5P's factored linear solve has a custom JVP
  \(\delta x=A^{-1}(\delta b-\delta A\,x)\). Every stage tangent and the
  transposed VJP reuse the attempt's pivoted LU factors; pivot selection is not
  differentiated.
- Semi-explicit DAE roots delegate both the primal solve and implicit AD to
  nlls-gram. Its implicit-function rule is
  \(\delta z=-g_z^{-1}(g_y\delta y+g_t\delta t+g_p\delta p)\); the default
  square `LU()` implementation differentiates the defining constraint rather
  than the optimizer iterations, and reverse mode transposes that rule.
  `LMRootSolver` requires residual-only convergence (`gtol=xtol=0`) and never
  accepts `MAX_STEPS` as a differentiable root. `ad_solver` remains an
  nlls-owned option forwarded through `solver_options`.
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
