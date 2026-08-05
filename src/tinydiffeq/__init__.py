"""Tiny differentiable ODE/SDE/DAE/SDAE solvers for JAX.

solve_ode integrates dx/dt = f(x, t, args, p) with fixed-step (Euler, RK4),
adaptive Tsit5, or linearly implicit Rodas5P methods on bounded lax.scan loops
with static shapes and composable forward/reverse AD. solve_sde integrates
diagonal-noise Ito SDEs (EulerMaruyama, Milstein, SRA1) from a PRNG key or an
explicit, differentiable noise pytree. solve_semi_explicit_dae and
solve_semi_explicit_sdae handle index-1 systems, delegating algebraic roots
and their implicit derivatives to nlls-gram. solve_linear_ode applies dense or
matrix-free Krylov exponential actions to fixed homogeneous linear systems,
and the Markov tools simulate and forecast finite-state chains. States may be
arrays or pytrees of same-dtype real floating arrays. Fully implicit solvers,
general mass matrices, events, continuous interpolation objects, and adjoint
methods are non-goals.
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
from tinydiffeq.solvers import (
    RK4,
    SRA1,
    Euler,
    EulerMaruyama,
    Milstein,
    Rodas5P,
    Tsit5,
    diagonal_brownian_increments,
)

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
    "Milstein",
    "SRA1",
    "diagonal_brownian_increments",
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
