# Adapted copy of kernels' integrators.py, used only as the parity reference
# for tinydiffeq's RK4 and Tsit5+IController implementations. Local names use
# tinydiffeq's indexed snake_case convention.
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import config

# Differentiable ODE integrators for policy rollouts: fixed-grid RK4 and a
# free-stepping adaptive Tsit5. Both operate on array states (scalar or
# vector -- every operation is elementwise and the adaptive error norm
# reduces over components; pytree states would need tree_map plumbing). Both
# are differentiable in both forward and reverse mode through the states,
# which is what an optimizer differentiating through the rollout needs
# (nlls_gram's vjp Jacobian plus the forward-mode directional derivative of
# its geodesic acceleration). `project` (e.g. a positivity clamp) is applied
# to every accepted state.

config.update("jax_enable_x64", True)

# Tsitouras (2011) 5(4) coefficients (FSAL: k_7 = f(y_5) is the next step's k_1).
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

SAFETY, MIN_FACTOR, MAX_FACTOR = 0.9, 0.2, 5.0


def identity_project(y):
    return y


# Classic fixed-step RK4 on the dt grid; returns all n_steps + 1 states.
def rk4_grid(f, y_0, n_steps, dt, project=identity_project):
    def step(y, _):
        s_1 = f(y)
        s_2 = f(project(y + 0.5 * dt * s_1))
        s_3 = f(project(y + 0.5 * dt * s_2))
        s_4 = f(project(y + dt * s_3))
        y_next = project(y + (dt / 6.0) * (s_1 + 2.0 * s_2 + 2.0 * s_3 + s_4))
        return y_next, y

    y_final, states = jax.lax.scan(step, y_0, None, length=n_steps)
    return jnp.concatenate(
        [states, y_final[None] if jnp.ndim(y_0) else jnp.atleast_1d(y_final)]
    )


def tsit5_stages(f, y, h, k_1=None):
    if k_1 is None:
        k_1 = f(y)
    k_2 = f(y + h * (A_21 * k_1))
    k_3 = f(y + h * (A_31 * k_1 + A_32 * k_2))
    k_4 = f(y + h * (A_41 * k_1 + A_42 * k_2 + A_43 * k_3))
    k_5 = f(y + h * (A_51 * k_1 + A_52 * k_2 + A_53 * k_3 + A_54 * k_4))
    k_6 = f(y + h * (A_61 * k_1 + A_62 * k_2 + A_63 * k_3 + A_64 * k_4 + A_65 * k_5))
    y_5 = y + h * (
        B_1 * k_1 + B_2 * k_2 + B_3 * k_3 + B_4 * k_4 + B_5 * k_5 + B_6 * k_6
    )
    k_7 = f(y_5)
    err = h * (
        E_1 * k_1
        + E_2 * k_2
        + E_3 * k_3
        + E_4 * k_4
        + E_5 * k_5
        + E_6 * k_6
        + E_7 * k_7
    )
    return y_5, k_7, err


# Free-stepping adaptive Tsit5 over [0, T]: steps adapt freely with no save
# grid, clipped only so no step lands past the horizon (the final accepted
# step hits T exactly). The horizon clip doubles as the guard on step growth:
# without it, a near-flat vector field lets steps quintuple into
# quarter-horizon leaps whose Gauss-Newton linearization stalls a
# trust-region optimizer differentiating through the rollout. Runs a fixed
# budget of n_iters attempts (static shapes for jit); every iteration emits
# its (t, y) -- an accepted step advances (FSAL: its k_7 seeds the next
# attempt), a rejection or post-horizon freeze repeats the previous state,
# which downstream least-squares residuals tolerate as a harmless duplicate
# row. If the budget is exhausted before reaching T the outputs are poisoned
# to inf (the caller's residual then rejects rather than reading a
# truncated path). Returns (ts, ys) of length n_iters + 1 including the initial
# state.
#
# The step-size CONTROLLER is wrapped in stop_gradient:
# accept/reject is a non-differentiable branch either way, the gradient of
# ratio**(-1/5) blows up at the exact-zero error of a flat-start policy, and
# the dh/dtheta term only slides sample points along the visited trajectory --
# irrelevant to a residual that must vanish at every state. The states
# themselves remain fully differentiable through the RK stages.
def tsit5_free(
    f, y_0, T, n_iters, rtol=1e-6, atol=1e-6, dt_0=1.0, project=identity_project
):
    def step(carry, _):
        t, y, h, k_1, done = carry
        remaining = jnp.maximum(T - t, 0.0)
        h_eff = jnp.minimum(h, jnp.maximum(remaining, 1e-12))
        y_5, k_7, err = tsit5_stages(f, y, h_eff, k_1)
        scale = atol + rtol * jnp.maximum(jnp.abs(y), jnp.abs(y_5))
        ratio = jax.lax.stop_gradient(jnp.max(jnp.abs(err) / scale))
        accept = (ratio <= 1.0) | (h_eff <= 1e-10)
        factor = jnp.clip(
            SAFETY * jnp.maximum(ratio, 1e-12) ** (-0.2), MIN_FACTOR, MAX_FACTOR
        )
        h_next = jnp.where(done, h, jax.lax.stop_gradient(h_eff) * factor)

        advance = accept & ~done
        y_new = jnp.where(advance, project(y_5), y)
        t_new = jnp.where(advance, t + h_eff, t)
        k_1_new = jnp.where(advance, k_7, k_1)
        done_new = done | (t_new >= T - 1e-9)
        return (t_new, y_new, h_next, k_1_new, done_new), (t_new, y_new)

    y_0 = jnp.asarray(y_0)
    t_0 = jnp.asarray(0.0, dtype=jnp.result_type(y_0, float))
    carry_0 = (t_0, y_0, jnp.asarray(dt_0, t_0.dtype), f(y_0), jnp.asarray(False))
    (_, _, _, _, done), (ts, ys) = jax.lax.scan(step, carry_0, None, length=n_iters)
    ts = jnp.concatenate([t_0[None], ts])
    ys = jnp.concatenate([y_0[None] if jnp.ndim(y_0) else jnp.atleast_1d(y_0), ys])
    # budget exhausted before T -> poison (caller's residual rejects)
    return jnp.where(done, ts, jnp.inf), jnp.where(done, ys, jnp.inf)
