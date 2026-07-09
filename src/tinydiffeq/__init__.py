"""Tiny differentiable ODE/SDE solvers for JAX.

solve_ode integrates dx/dt = f(x, t, args, p) with fixed-step (Euler, RK4)
or adaptive (Tsit5 + IController) explicit Runge-Kutta methods inside one
bounded lax.scan of exactly max_steps iterations, so shapes are static and
solves are differentiable in BOTH forward and reverse mode (including
reverse-over-forward) with O(max_steps) memory. SaveAt picks the output:
the endpoint, cubic-Hermite interpolation onto a fixed grid, or the raw
padded step rows. solve_sde is fixed-step Euler-Maruyama with presampled
diagonal noise. States are arrays (scalar or vector); pytree states,
implicit/stiff solvers, PID control, events, dense output, and adjoint
methods are non-goals — use diffrax for those.
"""

from tinydiffeq.controllers import ConstantStepSize, IController
from tinydiffeq.interpolation import hermite_interpolate
from tinydiffeq.ode import solve_ode
from tinydiffeq.quadrature import cumulative_trapezoid
from tinydiffeq.saveat import SaveAt
from tinydiffeq.sde import solve_sde
from tinydiffeq.solution import Solution
from tinydiffeq.solvers import RK4, Euler, EulerMaruyama, Tsit5

__all__ = [
    "solve_ode",
    "solve_sde",
    "Euler",
    "RK4",
    "Tsit5",
    "EulerMaruyama",
    "ConstantStepSize",
    "IController",
    "SaveAt",
    "Solution",
    "hermite_interpolate",
    "cumulative_trapezoid",
]
