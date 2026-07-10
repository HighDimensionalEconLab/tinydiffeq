import jax
import jax.numpy as jnp

from tinydiffeq.ode import canonicalize_field, identity_project
from tinydiffeq.save_at import SaveAt
from tinydiffeq.solution import Solution


def solve_sde(
    drift,
    diffusion,
    solver,
    t_0,
    t_1,
    x_0,
    *,
    key,
    n_steps,
    p=None,
    args=None,
    save_at=None,
    project=None,
):
    """Integrate the Ito SDE ``dx = drift dt + diffusion d_w`` (diagonal noise)
    on the fixed grid of ``n_steps`` uniform steps from ``t_0`` to ``t_1 > t_0``.

    ``drift`` and ``diffusion`` follow the same signature convention as
    ``solve_ode`` — ``(x)``, ``(x, t)``, ``(x, t, args)``, or
    ``(x, t, args, p)``. ``n_steps`` must be a static Python int (the honest
    static-shape contract; there is no adaptive SDE stepping in v1). The
    Brownian increments are presampled from ``key`` as
    ``sqrt(dt) * normal(key, (n_steps,) + x_0.shape)``, so a fixed key gives a
    fixed noise process: reproducible across calls and differentiable with
    respect to ``x_0`` and ``p`` (not ``key``).

    ``SaveAt(ts=...)`` raises — cubic Hermite interpolation is wrong for
    rough paths; use ``t_1`` (default) or ``steps`` (here ``n_steps + 1``
    rows, all accepted).
    """
    if not isinstance(n_steps, int):
        raise TypeError("n_steps must be a static Python int")
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1")
    if save_at is None:
        save_at = SaveAt(t_1=True)
    if save_at.ts is not None:
        raise ValueError(
            "SaveAt(ts=...) is not supported for SDEs: Hermite interpolation "
            "is wrong for rough paths; use SaveAt(t_1=True) or SaveAt(steps=True)"
        )
    if project is None:
        project = identity_project
    drift = canonicalize_field(drift, name="drift")
    diffusion = canonicalize_field(diffusion, name="diffusion")

    x_0 = jnp.asarray(x_0)
    time_dtype = jnp.result_type(x_0, float)
    t_0 = jnp.asarray(t_0, time_dtype)
    t_1 = jnp.asarray(t_1, time_dtype)
    dt = (t_1 - t_0) / n_steps
    d_w = jnp.sqrt(dt) * jax.random.normal(
        key, (n_steps,) + x_0.shape, dtype=time_dtype
    )
    time_grid = jnp.linspace(t_0, t_1, n_steps + 1)

    def g_drift(x, t):
        return drift(project(x), t, args, p)

    def g_diffusion(x, t):
        return diffusion(project(x), t, args, p)

    def body(x, inputs):
        t, d_w_step = inputs
        x_1 = solver.step(g_drift, g_diffusion, t, x, dt, d_w_step, project)
        return x_1, x_1 if save_at.steps else None

    x_final, step_states = jax.lax.scan(body, x_0, (time_grid[:-1], d_w))
    num_accepted = jnp.asarray(n_steps, jnp.int32)
    ok = jnp.asarray(True)

    if save_at.t_1:
        return Solution(ts=t_1, xs=x_final, ok=ok, num_accepted=num_accepted)

    all_states = jnp.concatenate([x_0[None], step_states])
    accepted = jnp.ones((n_steps + 1,), bool)
    return Solution(
        ts=time_grid,
        xs=all_states,
        ok=ok,
        num_accepted=num_accepted,
        accepted=accepted,
    )
