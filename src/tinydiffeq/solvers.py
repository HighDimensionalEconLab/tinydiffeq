from dataclasses import dataclass

import jax

# Solvers are stateless frozen dataclasses registered as pytrees so they pass
# through jit/vmap as ordinary arguments. `step` receives `g(x, t)` -- the
# user vector field already wrapped so every evaluation goes through
# `project` -- plus `f0`, the loop-carried value of `g` at (x, t) when the
# loop guarantees it is current (FSAL, or interpolation output requested);
# otherwise `f0` is None and the solver evaluates its own first stage.
# The step contract is `step(g, t, x, dt, f0, project) -> (x1, f1, err)`:
# `x1` is the projected accepted candidate, `f1` is `g(x1, t + dt)` when the
# solver produces it for free (FSAL) and None otherwise, and `err` is the
# embedded error estimate or None. `project` is assumed idempotent (a clamp).

# Tsitouras (2011) 5(4) coefficients (FSAL: k7 = f(x1) is the next step's k1).
A21 = 0.161
A31, A32 = -0.008480655492356989, 0.335480655492357
A41, A42, A43 = 2.8971530571054935, -6.359448489975075, 4.3622954328695815
A51, A52, A53, A54 = (
    5.325864828439257,
    -11.748883564062828,
    7.4955393428898365,
    -0.09249506636175525,
)
A61, A62, A63, A64, A65 = (
    5.86145544294642,
    -12.92096931784711,
    8.159367898576159,
    -0.071584973281401,
    -0.028269050394068383,
)
B1, B2, B3, B4, B5, B6 = (
    0.09646076681806523,
    0.01,
    0.4798896504144996,
    1.379008574103742,
    -3.290069515436081,
    2.324710524099774,
)
# embedded 4th-order error coefficients (b - bhat)
E1, E2, E3, E4, E5, E6, E7 = (
    -0.00178001105222577714,
    -0.0008164344596567469,
    0.007880878010261995,
    -0.1447110071732629,
    0.5823571654525552,
    -0.45808210592918697,
    1.0 / 66.0,
)
# stage times c_i (needed for non-autonomous fields)
C2, C3, C4, C5, C6, C7 = 0.161, 0.327, 0.9, 0.9800255409045097, 1.0, 1.0


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Euler:
    """Explicit Euler. Fixed-step only: no embedded error estimate."""

    order = 1
    fsal = False
    has_error_estimate = False

    def step(self, g, t, x, dt, f0, project):
        k1 = g(x, t) if f0 is None else f0
        x1 = project(x + dt * k1)
        return x1, None, None


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class RK4:
    """Classic fourth-order Runge-Kutta. Fixed-step only: no error estimate."""

    order = 4
    fsal = False
    has_error_estimate = False

    def step(self, g, t, x, dt, f0, project):
        k1 = g(x, t) if f0 is None else f0
        k2 = g(x + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = g(x + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = g(x + dt * k3, t + dt)
        x1 = project(x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4))
        return x1, None, None


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Tsit5:
    """Tsitouras 5(4) explicit Runge-Kutta with embedded error estimate.

    FSAL: the last stage k7 = g(x1, t + dt) is the next step's first stage,
    so an accepted adaptive step costs six fresh evaluations. Note k7 is
    evaluated at the *projected* accepted state, so the FSAL cache stays
    consistent with the state actually carried forward when `project` binds.
    """

    order = 5
    fsal = True
    has_error_estimate = True

    def step(self, g, t, x, dt, f0, project):
        k1 = g(x, t) if f0 is None else f0
        k2 = g(x + dt * (A21 * k1), t + C2 * dt)
        k3 = g(x + dt * (A31 * k1 + A32 * k2), t + C3 * dt)
        k4 = g(x + dt * (A41 * k1 + A42 * k2 + A43 * k3), t + C4 * dt)
        k5 = g(x + dt * (A51 * k1 + A52 * k2 + A53 * k3 + A54 * k4), t + C5 * dt)
        k6 = g(
            x + dt * (A61 * k1 + A62 * k2 + A63 * k3 + A64 * k4 + A65 * k5),
            t + C6 * dt,
        )
        x1 = project(
            x + dt * (B1 * k1 + B2 * k2 + B3 * k3 + B4 * k4 + B5 * k5 + B6 * k6)
        )
        k7 = g(x1, t + C7 * dt)
        err = dt * (E1 * k1 + E2 * k2 + E3 * k3 + E4 * k4 + E5 * k5 + E6 * k6 + E7 * k7)
        return x1, k7, err


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class EulerMaruyama:
    """Euler-Maruyama for Ito SDEs with diagonal noise. Fixed-step only."""

    order = 1

    def step(self, g_drift, g_diffusion, t, x, dt, dW, project):
        return project(x + dt * g_drift(x, t) + g_diffusion(x, t) * dW)
