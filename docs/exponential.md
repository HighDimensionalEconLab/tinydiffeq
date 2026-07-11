# Linear Exponential Solves

`solve_linear_ode` solves the autonomous homogeneous linear problem

\[
    \frac{dx}{dt} = A x,
    \qquad
    x(t) = \exp((t-t_0)A)x(t_0).
\]

It is the common exponential-action engine used by continuous-time Markov
distribution forecasts. It supports endpoint output and fixed query grids,
JIT, `vmap`, general state pytrees, and ordinary JVPs and VJPs.

## Dense operator

For a matrix, the interface uses the usual column-state convention `A @ x`:

```python
import jax.numpy as jnp

from tinydiffeq import DenseExponential, SaveAt, solve_linear_ode

A = jnp.asarray([[-2.0, 1.0], [0.5, -0.4]])
x_0 = jnp.asarray([1.0, 0.0])
solution = solve_linear_ode(
    A,
    DenseExponential(),
    0.0,
    4.0,
    x_0,
    save_at=SaveAt(ts=jnp.linspace(0.0, 4.0, 41)),
)
```

`DenseExponential` uses JAX's scaling-and-squaring matrix exponential. If the
operator is supplied as a callable, dense mode materializes its matrix with
forward-mode Jacobian columns at zero. This is exact for a linear callable and
is useful as a correctness baseline for small systems.

## Matrix-free pytree operator

`KrylovExponential` applies the exponential without constructing `A` or
`exp(A)`. The callable sees and returns the original pytree:

```python
from tinydiffeq import KrylovExponential

rates = jnp.linspace(0.1, 1.0, 100_000)

def operator(state):
    flat = jnp.concatenate([state["low"], state["high"]])
    flux = rates * flat
    derivative = jnp.roll(flux, 1) - flux
    split = state["low"].size
    return {"low": derivative[:split], "high": derivative[split:]}

x_0 = {
    "low": jnp.zeros(40_000).at[0].set(1.0),
    "high": jnp.zeros(60_000),
}
solution = solve_linear_ode(
    operator,
    KrylovExponential(krylov_dim=30, num_substeps=2),
    0.0,
    10.0,
    x_0,
)
```

The implementation ravels the state only for Arnoldi orthogonalization. Each
operator evaluation and the returned solution preserve the pytree. Two-pass
reorthogonalization limits loss of basis orthogonality, and a
precision-scaled happy-breakdown test avoids normalizing roundoff after the
Krylov subspace has closed. `solution.ok` combines finite-output checks with a
leading-term Arnoldi error estimate. Increase `krylov_dim` or
`num_substeps` if it is false.

## Adaptive matrix-free propagation

Use `AdaptiveKrylovExponential` when the appropriate number of internal
time slices is not known in advance:

```python
from tinydiffeq import AdaptiveKrylovExponential

method = AdaptiveKrylovExponential(krylov_dim=30, max_steps=128)
solution = solve_linear_ode(operator, method, 0.0, 10.0, x_0)
```

The Krylov dimension is a static compilation parameter. The method attempts an
initial slice spanning the remaining interval, estimates its leading Arnoldi
residual, and accepts or rejects it. Subsequent slice lengths use a bounded
integral controller. Local error is budgeted in proportion to the fraction of
the full interval advanced, so accepted residual budgets sum to the requested
endpoint scale. Defaults are `rtol=1e-5`, `atol=1e-7` for float32 and
`rtol=1e-10`, `atol=1e-12` for float64.

`max_steps` counts both accepted and rejected attempts; it is a static work
budget, not an output length. `solution.num_accepted` is the number of accepted
internal slices. If the budget is exhausted before the endpoint,
`solution.ok` is false and `solution.xs` is the last accepted finite state.
`initial_step` can cap the first attempted slice. `SaveAt(ts=...)` evaluates
independent adaptive actions for each requested time, and `num_accepted` is the
largest accepted count among them.

This is the fixed-dimension counterpart of adaptive time stepping in the
Niesen--Wright `expv` family. SciML's
[`expv_timestep`](https://docs.sciml.ai/ExponentialUtilities/stable/expv/)
can adapt both time slice and Krylov dimension. tinydiffeq keeps the dimension
static because that gives predictable JAX shapes and efficient `jit`/`vmap`;
changing it dynamically would execute padded or branched basis constructions.

Each Arnoldi vector is stored as a contiguous row internally. Although papers
usually write the basis as an n-by-m column matrix, the transposed storage is
substantially faster with JAX/XLA's row-major layouts and leaves the mathematics
unchanged. `reorthogonalization_passes=2` is the robust default. Setting it to
1 approximately halves projection memory traffic and can be appropriate for a
well-conditioned CTMC generator after float32/float64 validation; it is an
explicit choice because one-pass classical Gram--Schmidt can lose
orthogonality on difficult nonnormal operators.

The algorithm follows the Arnoldi `expv` family implemented in SciML's
[`ExponentialUtilities.jl`](https://docs.sciml.ai/ExponentialUtilities/stable/expv/).
SciML's broader
[`ExponentialIntegrators.jl`](https://docs.sciml.ai/ExponentialIntegrators/stable/)
also provides nonlinear exponential Runge--Kutta and multistep methods.

## Differentiation contract

All three methods are composed from JAX primitives. JVPs and VJPs pass through
endpoint and queried-grid values, through the initial state, and through arrays
used by a differentiable operator. For a truncated Krylov space, derivatives
are derivatives of the actual finite Arnoldi computation, including its basis.

For terminal sensitivities with a fixed operator, prefer the mathematical
linear-map rules:

```python
from tinydiffeq import jvp_linear_ode, vjp_linear_ode

solution, terminal_tangent = jvp_linear_ode(
    operator, method, 0.0, 10.0, x_0, x_0_tangent
)
solution, initial_cotangent = vjp_linear_ode(
    operator, method, 0.0, 10.0, x_0, terminal_cotangent
)
```

They evaluate

\[
    \delta x_1 = \exp((t_1-t_0)A)\delta x_0,
    \qquad
    \bar x_0 = \exp((t_1-t_0)A^\mathsf{T})\bar x_1,
\]

instead of differentiating Arnoldi orthogonalization. This makes the zero
initial state well-defined and sharply reduces CPU reverse-mode cost. Pass
`batched=True` and give every tangent or cotangent leaf a leading direction
axis to evaluate multiple directions after one primal solve. Dense mode forms
one matrix exponential and reuses it for all directions. Krylov mode vectorizes
independent forward or transposed actions; distinct right-hand sides generally
do not share one ordinary Arnoldi basis.

These hand-coded functions differentiate only with respect to the initial
state. The operator is fixed. Use ordinary `jax.jvp`/`jax.vjp` when operator
entries or arrays captured by a callable are differentiation targets. Dense
mode has a custom Fréchet derivative for matrix, initial-state, and horizon
tangents and avoids the Fréchet calculation when only the initial state is
active.

Ordinary traced Krylov AD remains available and differentiates the finite
Arnoldi computation. Because Arnoldi normalizes its starting vector, that path
assumes the initial state has nonzero norm. At exactly zero, use the hand-coded
initial-state functions above or `DenseExponential`.

For the adaptive method, ordinary AD differentiates the numerical computation
on the realized accepted/rejected path; the discrete controller decisions are
locally constant. The hand-coded initial-state JVP/VJP instead apply an
independent adaptive exponential action to each tangent or cotangent. They are
the preferred mathematical derivatives for a fixed operator and remain
well-defined at a zero primal state.

The operator must be fixed in time, homogeneous, and linear. This API does not
silently treat a nonlinear vector field as linear. A nonlinear exponential
Euler or exponential Rosenbrock method needs explicit Jacobian and
\(\varphi_k\)-action policies, local error estimation, and interpolation; that
is intentionally a separate future solver design.
