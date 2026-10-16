from dataclasses import dataclass

import jax
import jax.numpy as jnp

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

    An eight-stage, linearly implicit method with an embedded error estimate
    and a stiff-aware fourth-order continuous extension, supported by
    :func:`tinydiffeq.solve_ode` and
    :func:`tinydiffeq.solve_semi_explicit_dae` with one dense LU
    factorization reused across the stages of each attempted step. The
    implementation follows Steinebach (2023) and SciML's
    ``OrdinaryDiffEqRosenbrock.Rodas5P``:

    - https://doi.org/10.1007/s10543-023-00967-x
    - https://github.com/SciML/OrdinaryDiffEq.jl/tree/master/lib/OrdinaryDiffEqRosenbrock
    """

    order = 5
    fsal = False
    has_error_estimate = True


def diagonal_brownian_increments(x_0, key, n_steps, dt, dtype):
    """Draw ``n_steps`` diagonal Brownian increments ``sqrt(dt) * N(0, 1)``.

    Arrays retain the exact ``(n_steps,) + x_0.shape`` draw. Pytree states use
    one shared flat draw, partitioned into leaves in JAX's deterministic
    pytree leaf order.
    """
    leaves, treedef = jax.tree.flatten(x_0)
    if treedef == jax.tree.structure(0):
        return jnp.sqrt(dt) * jax.random.normal(
            key, (n_steps,) + x_0.shape, dtype=dtype
        )
    sizes = [leaf.size for leaf in leaves]
    flat_noise = jnp.sqrt(dt) * jax.random.normal(
        key, (n_steps, sum(sizes)), dtype=dtype
    )
    noise_leaves = []
    start = 0
    for leaf, size in zip(leaves, sizes, strict=True):
        noise_leaves.append(
            flat_noise[:, start : start + size].reshape((n_steps,) + leaf.shape)
        )
        start += size
    return jax.tree.unflatten(treedef, noise_leaves)


# SDE steppers share the contract
# `step(g_drift, g_diffusion, t, x, dt, noise, project) -> x_1` where `noise`
# is one per-step slice of the pytree produced by the solver's own
# `sample_noise(x_0, key, n_steps, dt, dtype)`, so explicit noise handed to
# `solve_sde` is validated against exactly what the solver expects.


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class EulerMaruyama:
    """Euler-Maruyama for Ito SDEs with diagonal noise. Fixed-step only.

    Strong order 0.5 for multiplicative noise. ``sample_noise`` returns the
    Brownian increments with the same pytree structure as the state and a
    leading ``n_steps`` axis.
    """

    order = 1
    strong_order = 0.5

    def sample_noise(self, x_0, key, n_steps, dt, dtype):
        return diagonal_brownian_increments(x_0, key, n_steps, dt, dtype)

    def step(self, g_drift, g_diffusion, t, x, dt, noise, project):
        return project(
            add_scaled(
                x, (dt, g_drift(x, t)), (1.0, multiply(g_diffusion(x, t), noise))
            )
        )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Milstein:
    """Milstein for Ito SDEs with diagonal noise. Fixed-step only.

    Strong order 1.0 under the diagonal commutativity condition: each
    diffusion component may depend only on its own state component. The
    correction ``(1/2) g g' (d_w^2 - dt)`` evaluates ``g g'`` as the
    forward-mode derivative of the diffusion field in the direction of its
    own value, which equals the diagonal term exactly in that case.
    ``sample_noise`` matches ``EulerMaruyama``.
    """

    order = 1
    strong_order = 1.0

    def sample_noise(self, x_0, key, n_steps, dt, dtype):
        return diagonal_brownian_increments(x_0, key, n_steps, dt, dtype)

    def step(self, g_drift, g_diffusion, t, x, dt, noise, project):
        g_value = g_diffusion(x, t)
        _, dg_g = jax.jvp(lambda state: g_diffusion(state, t), (x,), (g_value,))
        correction = jax.tree.map(lambda dg, w: 0.5 * dg * (w * w - dt), dg_g, noise)
        return project(
            add_scaled(
                x,
                (dt, g_drift(x, t)),
                (1.0, multiply(g_value, noise)),
                (1.0, correction),
            )
        )


INV_SQRT_3 = 3.0**-0.5


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SRA1:
    """Rossler SRA1 stochastic Runge-Kutta for Ito SDEs with additive
    diagonal noise. Fixed-step only.

    Strong order 1.5 when the diffusion is independent of the state (it may
    depend on time). ``sample_noise`` returns ``(d_w, d_z)``: two independent
    ``sqrt(dt) * N(0, 1)`` draws per step. The time-Wiener integral
    ``I_10 / dt`` is realized internally as ``(d_w + d_z / sqrt(3)) / 2``,
    reproducing its variance ``dt^3 / 3`` and covariance ``dt^2 / 2`` with
    the increment.
    """

    order = 2
    strong_order = 1.5

    def sample_noise(self, x_0, key, n_steps, dt, dtype):
        key_w, key_z = jax.random.split(key)
        return (
            diagonal_brownian_increments(x_0, key_w, n_steps, dt, dtype),
            diagonal_brownian_increments(x_0, key_z, n_steps, dt, dtype),
        )

    def step(self, g_drift, g_diffusion, t, x, dt, noise, project):
        d_w, d_z = noise
        g_0 = g_diffusion(x, t)
        g_1 = g_diffusion(x, t + dt)
        chi = jax.tree.map(lambda w, z: 0.5 * (w + z * INV_SQRT_3), d_w, d_z)
        k_1 = g_drift(x, t)
        stage = add_scaled(x, (0.75 * dt, k_1), (1.5, multiply(g_1, chi)))
        k_2 = g_drift(stage, t + 0.75 * dt)
        return project(
            add_scaled(
                x,
                (dt / 3.0, k_1),
                (2.0 * dt / 3.0, k_2),
                (1.0, multiply(g_1, d_w)),
                (1.0, multiply(add_scaled(g_0, (-1.0, g_1)), chi)),
            )
        )
