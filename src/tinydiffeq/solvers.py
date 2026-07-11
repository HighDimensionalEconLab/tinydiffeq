from dataclasses import dataclass

import jax

from tinydiffeq._tree import add_scaled, multiply, weighted_sum

# Solvers are stateless frozen dataclasses registered as pytrees so they pass
# through jit/vmap as ordinary arguments. `step` receives `g(x, t)` -- the
# user vector field already wrapped so every evaluation goes through
# `project` -- plus `f_0`, the loop-carried value of `g` at (x, t) when the
# loop guarantees it is current (FSAL, or interpolation output requested);
# otherwise `f_0` is None and the solver evaluates its own first stage.
# The step contract is `step(g, t, x, dt, f_0, project) -> (x_1, f_1, err)`:
# `x_1` is the projected accepted candidate, `f_1` is `g(x_1, t + dt)` when the
# solver produces it for free (FSAL) and None otherwise, and `err` is the
# embedded error estimate or None. `project` is assumed idempotent (a clamp).

# Tsitouras (2011) 5(4) coefficients (FSAL: k_7 = f(x_1) is the next step's k_1).
A_21 = 0.161
A_31, A_32 = -0.008480655492356989, 0.335480655492357
A_41, A_42, A_43 = 2.8971530571054935, -6.359448489975075, 4.3622954328695815
A_51, A_52, A_53, A_54 = (
    5.325864828439257,
    -11.748883564062828,
    7.4955393428898365,
    -0.09249506636175525,
)
A_61, A_62, A_63, A_64, A_65 = (
    5.86145544294642,
    -12.92096931784711,
    8.159367898576159,
    -0.071584973281401,
    -0.028269050394068383,
)
B_1, B_2, B_3, B_4, B_5, B_6 = (
    0.09646076681806523,
    0.01,
    0.4798896504144996,
    1.379008574103742,
    -3.290069515436081,
    2.324710524099774,
)
# embedded 4th-order error coefficients (b - bhat)
E_1, E_2, E_3, E_4, E_5, E_6, E_7 = (
    -0.00178001105222577714,
    -0.0008164344596567469,
    0.007880878010261995,
    -0.1447110071732629,
    0.5823571654525552,
    -0.45808210592918697,
    1.0 / 66.0,
)
# stage times c_i (needed for non-autonomous fields)
C_2, C_3, C_4, C_5, C_6, C_7 = 0.161, 0.327, 0.9, 0.9800255409045097, 1.0, 1.0


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Euler:
    """Explicit Euler. Fixed-step only: no embedded error estimate."""

    order = 1
    fsal = False
    has_error_estimate = False

    def step(self, g, t, x, dt, f_0, project):
        k_1 = g(x, t) if f_0 is None else f_0
        x_1 = project(add_scaled(x, (dt, k_1)))
        return x_1, None, None

    def step_fixed(self, g, t, x, dt, f_0, project):
        return self.step(g, t, x, dt, f_0, project)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class RK4:
    """Classic fourth-order Runge-Kutta. Fixed-step only: no error estimate."""

    order = 4
    fsal = False
    has_error_estimate = False

    def step(self, g, t, x, dt, f_0, project):
        k_1 = g(x, t) if f_0 is None else f_0
        k_2 = g(add_scaled(x, (0.5 * dt, k_1)), t + 0.5 * dt)
        k_3 = g(add_scaled(x, (0.5 * dt, k_2)), t + 0.5 * dt)
        k_4 = g(add_scaled(x, (dt, k_3)), t + dt)
        x_1 = project(
            add_scaled(
                x,
                (dt / 6.0, k_1),
                (dt / 3.0, k_2),
                (dt / 3.0, k_3),
                (dt / 6.0, k_4),
            )
        )
        return x_1, None, None

    def step_fixed(self, g, t, x, dt, f_0, project):
        return self.step(g, t, x, dt, f_0, project)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Tsit5:
    """Tsitouras 5(4) explicit Runge-Kutta with embedded error estimate.

    FSAL: the last stage k_7 = g(x_1, t + dt) is the next step's first stage,
    so an accepted adaptive step costs six fresh evaluations. Note k_7 is
    evaluated at the *projected* accepted state, so the FSAL cache stays
    consistent with the state actually carried forward when `project` binds.
    """

    order = 5
    fsal = True
    has_error_estimate = True

    def _step(self, g, t, x, dt, f_0, project, *, need_error):
        k_1 = g(x, t) if f_0 is None else f_0
        k_2 = g(add_scaled(x, (dt * A_21, k_1)), t + C_2 * dt)
        k_3 = g(
            add_scaled(x, (dt, weighted_sum((k_1, k_2), (A_31, A_32)))),
            t + C_3 * dt,
        )
        k_4 = g(
            add_scaled(x, (dt, weighted_sum((k_1, k_2, k_3), (A_41, A_42, A_43)))),
            t + C_4 * dt,
        )
        k_5 = g(
            add_scaled(
                x,
                (dt, weighted_sum((k_1, k_2, k_3, k_4), (A_51, A_52, A_53, A_54))),
            ),
            t + C_5 * dt,
        )
        k_6 = g(
            add_scaled(
                x,
                (
                    dt,
                    weighted_sum(
                        (k_1, k_2, k_3, k_4, k_5),
                        (A_61, A_62, A_63, A_64, A_65),
                    ),
                ),
            ),
            t + C_6 * dt,
        )
        x_1 = project(
            add_scaled(
                x,
                (
                    dt,
                    weighted_sum(
                        (k_1, k_2, k_3, k_4, k_5, k_6),
                        (B_1, B_2, B_3, B_4, B_5, B_6),
                    ),
                ),
            )
        )
        k_7 = g(x_1, t + C_7 * dt)
        if need_error:
            err = jax.tree.map(
                lambda value: dt * value,
                weighted_sum(
                    (k_1, k_2, k_3, k_4, k_5, k_6, k_7),
                    (E_1, E_2, E_3, E_4, E_5, E_6, E_7),
                ),
            )
        else:
            err = None
        return x_1, k_7, err

    def step(self, g, t, x, dt, f_0, project):
        return self._step(g, t, x, dt, f_0, project, need_error=True)

    def step_fixed(self, g, t, x, dt, f_0, project):
        """Take a fixed step without constructing the unused embedded error."""
        return self._step(g, t, x, dt, f_0, project, need_error=False)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Rodas5P:
    """Fifth-order Rodas5P Rosenbrock--Wanner method.

    Rodas5P is an eight-stage, linearly implicit method with an embedded
    error estimate and a stiff-aware fourth-order continuous extension. It is
    supported by :func:`tinydiffeq.solve_ode` and
    :func:`tinydiffeq.solve_semi_explicit_dae`; both use one dense LU
    factorization per attempted step and reuse it across all stages.

    The implementation follows Steinebach (2023) and SciML's
    ``OrdinaryDiffEqRosenbrock.Rodas5P`` implementation:

    - https://doi.org/10.1007/s10543-023-00967-x
    - https://github.com/SciML/OrdinaryDiffEq.jl/tree/master/lib/OrdinaryDiffEqRosenbrock
    """

    order = 5
    fsal = False
    has_error_estimate = True


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class EulerMaruyama:
    """Euler-Maruyama for Ito SDEs with diagonal noise. Fixed-step only."""

    order = 1

    def step(self, g_drift, g_diffusion, t, x, dt, d_w, project):
        return project(
            add_scaled(x, (dt, g_drift(x, t)), (1.0, multiply(g_diffusion(x, t), d_w)))
        )
