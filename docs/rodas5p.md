# Rodas5P: Stiff ODEs and Index-1 DAEs

`Rodas5P()` is an eight-stage, fifth-order Rosenbrock–Wanner method with an
embedded adaptive estimator and a stiff-aware fourth-order continuous
extension. It is A-stable and stiffly accurate. In tinydiffeq it supports
ordinary ODEs and semi-explicit index-1 DAEs through the existing solve APIs.

## Credit and implementation lineage

Rodas5P was constructed by Gerd Steinebach and introduced and benchmarked in
[Steinebach (2023)](https://doi.org/10.1007/s10543-023-00967-x). The method was
developed in the Julia Differential Equations ecosystem, and tinydiffeq's JAX
implementation deliberately follows SciML's authoritative implementation:

- [SciML `OrdinaryDiffEqRosenbrock` package](https://github.com/SciML/OrdinaryDiffEq.jl/tree/master/lib/OrdinaryDiffEqRosenbrock)
- [SciML Rodas5P tableau and dense coefficients](https://github.com/SciML/OrdinaryDiffEq.jl/blob/master/lib/OrdinaryDiffEqRosenbrock/src/rosenbrock_tableaus.jl)
- [SciML consolidated Rosenbrock step implementation](https://github.com/SciML/OrdinaryDiffEq.jl/blob/master/lib/OrdinaryDiffEqRosenbrock/src/rosenbrock_perform_step.jl)
- [SciML Rosenbrock dense interpolants](https://github.com/SciML/OrdinaryDiffEq.jl/blob/master/lib/OrdinaryDiffEqRosenbrock/src/rosenbrock_interpolants.jl)
- [SciML solver documentation and selection guidance](https://docs.sciml.ai/DiffEqDocs/stable/api/ordinarydiffeq/massmatrixdae/Rosenbrock/)

SciML's implementation is MIT-licensed. Its source is the reference for the
port and receives explicit attribution here, in the kernel source, and in the
cross-library regression tests and benchmarks.

The tableau, stage equations, embedded estimate, and continuous extension are
ported rather than redesigned. Regression tests compare fixed Rodas5P steps
against values produced by SciML to prevent the implementations from silently
diverging.

## ODE use

```python
import jax.numpy as jnp

from tinydiffeq import IController, Rodas5P, solve_ode


def stiff_field(x, t):
    # Exact solution x(t) = cos(t), with a fast transient eigenvalue -1000.
    return -1000.0 * (x - jnp.cos(t)) - jnp.sin(t)


sol = solve_ode(
    stiff_field,
    Rodas5P(),
    0.0,
    1.0,
    jnp.asarray(1.0),
    dt_0=0.01,
    controller=IController(),
    max_steps=512,
)
```

At the beginning of an attempted step, Rodas5P forms the exact JAX Jacobian
(J=\partial f/\partial x), the time derivative (f_t), and

$$
W = \frac{I}{\gamma h} - J.
$$

It computes one pivoted LU factorization of (W) and reuses it for all eight
stage right-hand sides. It therefore solves linear systems but performs no
Newton or Levenberg–Marquardt iteration.

## Semi-explicit DAE use

For

$$
\dot y=f(y,z,t), \qquad 0=g(y,z,t),
$$

tinydiffeq constructs the flattened mass-matrix system internally:

$$
M\dot u=F(u,t), \qquad
u=(y,z), \quad M=\operatorname{diag}(I_y,0_z), \quad F=(f,g).
$$

```python
from tinydiffeq import Rodas5P, solve_semi_explicit_dae


sol = solve_semi_explicit_dae(
    lambda y, z, t, args, p: p * z,
    lambda y, z: z**2 - y - 2.0,
    Rodas5P(),
    0.0,
    1.0,
    jnp.asarray(1.0),
    jnp.sqrt(jnp.asarray(3.0)),
    p=jnp.asarray(-0.2),
    dt_0=0.1,
    controller=IController(),
    max_steps=128,
)
```

`LMRootSolver` is used once to make the initial algebraic state consistent.
Every later Rodas5P stage solves

$$
\left(\frac{M}{\gamma h}-F_u\right)k_i=r_i
$$

with the reused LU factors. There is no nonlinear endpoint restoration.
Consequently, returned internal `z` values satisfy the constraint to the
method's integration accuracy, not necessarily to `LMRootSolver.atol`.

This differs intentionally from the RK4 and Tsit5 DAE paths, which perform a
nonlinear algebraic solve at every stage and accepted endpoint. Rodas5P is
especially useful when those algebraic solves dominate runtime or when the
coupled dynamics are stiff.

## Dense output, aux, and AD

`SaveAt(ts=...)` uses Rodas5P's SciML/paper continuous extension for the ODE
state or combined `(y, z)` state. The three stored coefficient pytrees form a
fourth-order stiff-aware polynomial; no query-time algebraic solve occurs.

With `has_aux=True`, aux is evaluated and stored at the initial and accepted
states. Requested-grid aux uses normalized cubic Hermite interpolation with
endpoint tangents obtained from the Rodas5P polynomial and a JVP through the
aux map. It is not recalculated or root-solved at each requested time.

The public derivative is the derivative of the discrete Rodas5P method.
JVP/VJP propagate through Jacobian construction, stage solves, state output,
and dense output. Each factored solve uses the mathematical linear-solve rule
$dx=A^{-1}(db-dA\,x)$, so discrete LU pivot choices are not differentiated.
As with the other adaptive methods, accept/reject decisions and step-size
selection are stop-gradiented.

## Deliberate limits

- Dense exact Jacobians and dense pivoted LU only; no sparse, Krylov, or
  preconditioner interface.
- Constant identity or internally constructed block-diagonal mass matrices;
  no public general mass-matrix API.
- Semi-explicit index-1 DAEs with a square, locally nonsingular $g_z$.
- No fully implicit residual-form DAE interface or automatic index reduction.

For those broader capabilities, use SciML's
[`OrdinaryDiffEqRosenbrock`](https://docs.sciml.ai/DiffEqDocs/stable/api/ordinarydiffeq/massmatrixdae/Rosenbrock/)
and related DAE solvers.
