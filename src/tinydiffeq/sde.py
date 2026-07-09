import jax
import jax.numpy as jnp

from tinydiffeq.ode import canonicalize_field, identity_project
from tinydiffeq.saveat import SaveAt
from tinydiffeq.solution import Solution


def solve_sde(
    drift,
    diffusion,
    solver,
    t0,
    t1,
    x0,
    *,
    key,
    n_steps,
    p=None,
    args=None,
    saveat=None,
    project=None,
):
    """Integrate the Ito SDE ``dx = drift dt + diffusion dW`` (diagonal noise)
    on the fixed grid of ``n_steps`` uniform steps from ``t0`` to ``t1 > t0``.

    ``drift`` and ``diffusion`` follow the same signature convention as
    ``solve_ode`` — ``(x)``, ``(x, t)``, ``(x, t, args)``, or
    ``(x, t, args, p)``. ``n_steps`` must be a static Python int (the honest
    static-shape contract; there is no adaptive SDE stepping in v1). The
    Brownian increments are presampled from ``key`` as
    ``sqrt(dt) * normal(key, (n_steps,) + x0.shape)``, so a fixed key gives a
    fixed noise process: reproducible across calls and differentiable with
    respect to ``x0`` and ``p`` (not ``key``).

    ``SaveAt(ts=...)`` raises — cubic Hermite interpolation is wrong for
    rough paths; use ``t1`` (default) or ``steps`` (here ``n_steps + 1``
    rows, all accepted).
    """
    if not isinstance(n_steps, int):
        raise TypeError("n_steps must be a static Python int")
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1")
    if saveat is None:
        saveat = SaveAt(t1=True)
    if saveat.ts is not None:
        raise ValueError(
            "SaveAt(ts=...) is not supported for SDEs: Hermite interpolation "
            "is wrong for rough paths; use SaveAt(t1=True) or SaveAt(steps=True)"
        )
    if project is None:
        project = identity_project
    drift = canonicalize_field(drift, name="drift")
    diffusion = canonicalize_field(diffusion, name="diffusion")

    x0 = jnp.asarray(x0)
    tdt = jnp.result_type(x0, float)
    t0 = jnp.asarray(t0, tdt)
    t1 = jnp.asarray(t1, tdt)
    dt = (t1 - t0) / n_steps
    dW = jnp.sqrt(dt) * jax.random.normal(key, (n_steps,) + x0.shape, dtype=tdt)
    ts_grid = jnp.linspace(t0, t1, n_steps + 1)

    def g_drift(x, t):
        return drift(project(x), t, args, p)

    def g_diffusion(x, t):
        return diffusion(project(x), t, args, p)

    def body(x, inputs):
        t, dW_step = inputs
        x1 = solver.step(g_drift, g_diffusion, t, x, dt, dW_step, project)
        return x1, x1 if saveat.steps else None

    x_final, xs_s = jax.lax.scan(body, x0, (ts_grid[:-1], dW))
    num_accepted = jnp.asarray(n_steps, jnp.int32)
    ok = jnp.asarray(True)

    if saveat.t1:
        return Solution(ts=t1, xs=x_final, ok=ok, num_accepted=num_accepted)

    xs_all = jnp.concatenate([x0[None], xs_s])
    accepted = jnp.ones((n_steps + 1,), bool)
    return Solution(
        ts=ts_grid, xs=xs_all, ok=ok, num_accepted=num_accepted, accepted=accepted
    )
