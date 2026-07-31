# tinydiffeq

`tinydiffeq` is a deliberately tiny set of ODE/SDE/DAE/SDAE integrators and
finite-state Markov simulators for JAX: fixed-step Euler and RK4, adaptive Tsit5 with integral or
proportional-integral step-size control, linearly implicit Rodas5P for stiff
ODEs and index-1 DAEs, and fixed-step Euler–Maruyama for Itô SDEs and SDAEs.
Fixed stepping and the default adaptive path use bounded `lax.scan` loops with
static shapes. These solves support forward mode, reverse mode, and
reverse-over-forward. Adaptive ODE and DAE solves also offer
`adaptive_loop="forward"`, a dynamic actual-work loop for primal, JVP, and
nested forward AD; JAX cannot reverse-transpose this path.
Finite-state DTMC/CTMC simulation is primal-only and offers chronological scan
and associative parallel-prefix methods. Fixed-chain probability forecasts use
binary matrix powers for DTMC endpoints, dense exponentials for small CTMCs, or
fixed or adaptive matrix-free Arnoldi/Krylov actions over array or pytree
probabilities; see
[Markov Chains](markov_chains.md). The same backends solve general fixed
homogeneous linear systems; see [Linear Exponential Solves](exponential.md).

Rodas5P follows SciML's
[`OrdinaryDiffEqRosenbrock`](https://github.com/SciML/OrdinaryDiffEq.jl/tree/master/lib/OrdinaryDiffEqRosenbrock)
implementation and Steinebach's published method. **Use SciML or
[diffrax](https://docs.kidger.site/diffrax/) instead if you need any of:**

- general mass matrices, fully implicit solvers, or higher-index DAEs
- sparse/Krylov linear solves and preconditioners inside general ODE/DAE stages
- full derivative-term PID step-size control
- events, root-finding, or backward-time integration
- dense output / continuous interpolation objects
- checkpointed or backsolve adjoints for long horizons

Use the default bounded loop when reverse mode is required. Use the forward
loop when the actual adaptive attempt count is much smaller than `max_steps`
and the caller needs only primal or forward-mode execution. Both retain static
public output shapes; under `vmap`, a dynamic loop still runs until the slowest
lane finishes.

## 2.4.0 migration note

- `SaveAt(ts=..., exact=True)` now gathers realized knots for explicit
  fixed-step ODEs. Every query must align with a knot; adaptive methods,
  Rodas5P, DAEs, SDEs, and SDAEs continue to reject exact mode.
- `Solution.num_steps` and `DAESolution.num_steps` count logical attempts,
  including rejections. DAE results additionally expose `num_root_solves` and
  `num_root_steps`; `num_accepted` retains its existing meaning.
- Adaptive ODE and DAE solves may opt into `adaptive_loop="forward"` for an
  actual-work loop. It supports primal, JVP, and nested forward AD but not
  reverse mode; `adaptive_loop="bounded"` remains the reverse-mode-capable
  default. Under `vmap`, the forward loop runs to the slowest lane.
- `LMRootSolver(predictor="secant")` is an opt-in continuation warm start for
  locally unique algebraic branches; `predictor="previous"` remains the
  default.

DAE root acceptance is stricter in 2.4.0. nlls-gram owns both the primal root
solve and implicit derivative; square implicit AD defaults to direct `LU()`.
Only `CONVERGED` roots whose residual norm is below `atol` are accepted, so
`gtol` and `xtol` must both be zero. `max_steps_is_success` remains for source
compatibility, now defaults to `False`, and never makes `MAX_STEPS` a valid
root. Upgrading configurations should remove nonzero `gtol`/`xtol`; if they
relied on budget exhaustion, increase the root budget or adjust the residual
tolerance instead.

## Install

```bash
uv add tinydiffeq
```

For accelerator use, install the JAX build matching your hardware alongside
it, for example:

```bash
uv add tinydiffeq "jax[cuda13]"
```

## Vector-field interface

The vector field may take one to four positional arguments — always in this
order:

```python
f(x)                # autonomous, closes over everything
f(x, t)
f(x, t, args)
f(x, t, args, p)
```

- `x` is an array or pytree state. Leaves must share one real floating dtype;
  the vector field returns the same structure and dtype.
- `t` is time.
- `args` is pass-through data. By convention it is **not** an AD target —
  nothing stops you differentiating with respect to it, but the library's
  contracts and tests treat it as constants.
- `p` holds differentiable parameters — any pytree, e.g. neural-network
  weights. jvp/vjp with respect to `p` and `x_0` are first-class and tested.

The arity is inspected once and the function is wrapped into the canonical
four-argument form, so the compiled code is identical for all four. There is
no special autonomous code path: an unused `t` is dead-code-eliminated.
`drift` and `diffusion` in [`solve_sde`](sde.md) follow the same convention.
Semi-explicit DAE fields use `(y, z)`, `(y, z, t)`,
`(y, z, t, args)`, or `(y, z, t, args, p)`; see
[Semi-Explicit DAEs](dae.md).
Differential fields and stochastic drifts may return `(value, saved_aux)`;
DAE/SDAE algebraic functions may separately return internal context consumed
by the dynamics. Saved aux is differentiated and follows `SaveAt`. See
[Auxiliary Outputs](aux.md) for all supported contracts and
[Semi-Explicit SDAEs](sdae.md) for the stochastic form.

## Minimal example

```python
import jax
import jax.numpy as jnp
from tinydiffeq import solve_ode, Tsit5, IController, SaveAt

jax.config.update("jax_enable_x64", True)  # your call, not the library's


def f(x, t, args, p):
    return -p * x


sol = solve_ode(
    f, Tsit5(), 0.0, 2.0, jnp.asarray(1.0),
    p=jnp.asarray(1.3),
    dt_0=0.1,
    controller=IController(rtol=1e-8, atol=1e-10),
    max_steps=512,
    save_at=SaveAt(ts=jnp.linspace(0.0, 2.0, 21)),
)
sol.xs  # (21,) states on the grid, however many internal steps were taken
sol.ok  # False if integration or a requested output failed
```

With no arguments, `IController()` and `PIController()` use precision-aware
tolerances: `rtol=1e-4, atol=1e-6` for float32 states and
`rtol=1e-7, atol=1e-9` for float64 states. Explicit values override the
policy and are cast to the state dtype. Automatic `dt_min` is
`10 * eps * max(1, abs(t_1))` in the time dtype.

Gradients go straight through the solve:

```python
def endpoint(p):
    return solve_ode(
        f, Tsit5(), 0.0, 2.0, jnp.asarray(1.0), p=p,
        dt_0=0.1, controller=IController(rtol=1e-10, atol=1e-12),
        max_steps=512,
    ).xs

jax.grad(endpoint)(jnp.asarray(1.3))          # reverse mode
jax.jvp(endpoint, (jnp.asarray(1.3),), (jnp.asarray(1.0),))  # forward mode
jax.grad(lambda p: jax.jvp(endpoint, (p,), (jnp.asarray(1.0),))[1])(
    jnp.asarray(1.3)
)  # reverse-over-forward
```

## Design contracts at a glance

- **`dt_0` is required.** There is no initial-step heuristic.
- **`max_steps` counts attempted internal steps**, including rejections. It
  controls the bounded scan and only becomes an output-row count in
  `SaveAt(steps=True)`, which returns `max_steps + 1` padded rows including
  the initial state. Accepted steps form a contiguous prefix; rejected
  attempts are omitted and the tail repeats the last accepted state.
  `sol.num_steps` reports actual attempts and `sol.num_accepted` reports
  successful advances.
- **Fixed-step times do not depend on the attempt budget.** They are formed
  arithmetically from the accepted-step index, with only a small local endpoint
  snap. Increasing a nonbinding `max_steps` therefore does not change the
  numerical method.
- **Exact fixed-step output is explicit-ODE-only.**
  `SaveAt(ts=grid, exact=True)` gathers internal knots without interpolation;
  every query must align with a realized knot.
- **Adaptive loop choice is an AD choice.** `adaptive_loop="bounded"` is the
  reverse-mode-capable default. `adaptive_loop="forward"` executes actual
  attempts but supports only primal, JVP, and nested forward mode.
- **Forward time only**: `t_1 > t_0`.
- **Never poisons.** `sol.ok` reports whether `t_1` was reached and every
  requested output was valid; callers that want diverging residuals can map
  `jnp.where(sol.ok, x, jnp.inf)` over `sol.xs`.
- **`project`** (an idempotent clamp, e.g. positivity) is applied at every
  point where the vector field is evaluated and to every accepted state.
- **Never sets `jax_enable_x64`.** The time dtype follows the common state
  dtype; float32 problems stay float32 even when x64 is enabled.
- Solvers, controllers, `SaveAt`, and `Solution` are frozen dataclasses
  registered as pytrees: numeric fields (tolerances, grids, `dt_0`, `x_0`) are
  data leaves, so changing them never recompiles.

Read next: [Static Shapes](static_shapes.md) for the loop and
`SaveAt`, [Adaptive Stepping and AD](adaptive_ad.md) for what is and is not
differentiated, [Auxiliary Outputs](aux.md), [Rodas5P](rodas5p.md) for the SciML-derived linearly implicit
method, [DAEs](dae.md), [SDEs](sde.md), [SDAEs](sdae.md), and the [API
Reference](api.md).
