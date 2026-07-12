"""Tiny differentiable ODE/SDE/DAE/SDAE solvers for JAX.

solve_ode integrates dx/dt = f(x, t, args, p) with fixed-step (Euler, RK4),
adaptive explicit Tsit5, or linearly implicit Rodas5P methods
inside bounded lax.scan loops with exactly max_steps attempt slots, so shapes are
static and solves are differentiable in BOTH forward and reverse mode
(including reverse-over-forward) with O(max_steps) memory. SaveAt picks the
output: the endpoint, method-specific dense interpolation onto a fixed grid, or
accepted internal steps with fixed-shape padding. solve_sde is fixed-step
Euler-Maruyama with presampled
diagonal noise. solve_semi_explicit_dae handles index-1 systems with either
root-restored explicit methods or the stiff Rodas5P mass-matrix formulation,
plus differentiable saved differential-field aux and internal algebraic
context.
solve_semi_explicit_sdae applies fixed-step Euler-Maruyama to the reduced
index-1 stochastic system. solve_linear_ode applies dense, fixed-Krylov, or
adaptive matrix-free exponential actions to fixed homogeneous linear systems.
States may be arrays or pytrees of same-dtype real floating arrays. Fully
implicit solvers, general mass
matrices, full derivative-term PID control, events, continuous interpolation
objects, and adjoint methods are non-goals.
"""

from tinydiffeq.controllers import ConstantStepSize, IController, PIController
from tinydiffeq.dae import LMRootSolver, solve_semi_explicit_dae
from tinydiffeq.exponential import (
    AdaptiveKrylovExponential,
    DenseExponential,
    KrylovExponential,
    jvp_linear_ode,
    solve_linear_ode,
    vjp_linear_ode,
)
from tinydiffeq.interpolation import hermite_interpolate
from tinydiffeq.markov import (
    AssociativeMarkov,
    ContinuousTimeMarkovChain,
    DiscreteMarkovChain,
    MarkovDistribution,
    MatrixFreeContinuousTimeMarkovChain,
    MatrixPowerMarkov,
    SequentialMarkov,
    forecast_continuous_time_markov_chain,
    forecast_markov_chain,
    simulate_continuous_time_markov_chain,
    simulate_markov_chain,
)
from tinydiffeq.ode import solve_ode
from tinydiffeq.quadrature import cumulative_trapezoid
from tinydiffeq.save_at import SaveAt
from tinydiffeq.sdae import solve_semi_explicit_sdae
from tinydiffeq.sde import solve_sde
from tinydiffeq.solution import DAESolution, Solution
from tinydiffeq.solvers import RK4, Euler, EulerMaruyama, Rodas5P, Tsit5

__all__ = [
    "solve_ode",
    "solve_semi_explicit_dae",
    "solve_sde",
    "solve_semi_explicit_sdae",
    "solve_linear_ode",
    "jvp_linear_ode",
    "vjp_linear_ode",
    "Euler",
    "RK4",
    "Tsit5",
    "Rodas5P",
    "EulerMaruyama",
    "ConstantStepSize",
    "IController",
    "PIController",
    "DiscreteMarkovChain",
    "ContinuousTimeMarkovChain",
    "SequentialMarkov",
    "AssociativeMarkov",
    "MatrixPowerMarkov",
    "DenseExponential",
    "KrylovExponential",
    "AdaptiveKrylovExponential",
    "MarkovDistribution",
    "MatrixFreeContinuousTimeMarkovChain",
    "simulate_markov_chain",
    "simulate_continuous_time_markov_chain",
    "forecast_markov_chain",
    "forecast_continuous_time_markov_chain",
    "SaveAt",
    "Solution",
    "DAESolution",
    "LMRootSolver",
    "hermite_interpolate",
    "cumulative_trapezoid",
]
