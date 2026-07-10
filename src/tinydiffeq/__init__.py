"""Tiny differentiable ODE/SDE/DAE/SDAE solvers for JAX.

solve_ode integrates dx/dt = f(x, t, args, p) with fixed-step (Euler, RK4)
or adaptive (Tsit5 + IController/PIController) explicit Runge-Kutta methods
inside one bounded lax.scan of exactly max_steps iterations, so shapes are
static and solves are differentiable in BOTH forward and reverse mode
(including reverse-over-forward) with O(max_steps) memory. SaveAt picks the
output: the endpoint, cubic-Hermite interpolation onto a fixed grid, or
accepted internal steps with fixed-shape padding. solve_sde is fixed-step
Euler-Maruyama with presampled
diagonal noise. solve_semi_explicit_dae handles nonstiff index-1 systems with
an implicitly differentiated algebraic root and optional aux output.
solve_semi_explicit_sdae applies fixed-step Euler-Maruyama to the reduced
index-1 stochastic system. States may be arrays or pytrees
of same-dtype real floating arrays. Stiff or fully implicit solvers, full
derivative-term PID control, events, continuous interpolation objects, and
adjoint methods are non-goals — use diffrax for those.
"""

from tinydiffeq.controllers import ConstantStepSize, IController, PIController
from tinydiffeq.dae import LMRootSolver, solve_semi_explicit_dae
from tinydiffeq.interpolation import hermite_interpolate
from tinydiffeq.ode import solve_ode
from tinydiffeq.quadrature import cumulative_trapezoid
from tinydiffeq.save_at import SaveAt
from tinydiffeq.sdae import solve_semi_explicit_sdae
from tinydiffeq.sde import solve_sde
from tinydiffeq.solution import DAESolution, Solution
from tinydiffeq.solvers import RK4, Euler, EulerMaruyama, Tsit5

__all__ = [
    "solve_ode",
    "solve_semi_explicit_dae",
    "solve_sde",
    "solve_semi_explicit_sdae",
    "Euler",
    "RK4",
    "Tsit5",
    "EulerMaruyama",
    "ConstantStepSize",
    "IController",
    "PIController",
    "SaveAt",
    "Solution",
    "DAESolution",
    "LMRootSolver",
    "hermite_interpolate",
    "cumulative_trapezoid",
]
