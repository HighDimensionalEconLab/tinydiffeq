import inspect

import jax
import jax.numpy as jnp

from tinydiffeq.controllers import ConstantStepSize
from tinydiffeq.interpolation import hermite_interpolate
from tinydiffeq.saveat import SaveAt
from tinydiffeq.solution import Solution


def identity_project(x):
    return x


def canonicalize_field(f, name="f"):
    # The vector field may take (x), (x, t), (x, t, args), or (x, t, args, p),
    # always in that order; it is wrapped into the canonical 4-arg form here
    # so the compiled code is identical for all four. Uninspectable
    # signatures (or *args) are assumed 4-arg.
    try:
        signature = inspect.signature(f)
    except (TypeError, ValueError):
        arity = 4
    else:
        arity = 0
        for parameter in signature.parameters.values():
            if parameter.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                arity += 1
            elif parameter.kind == inspect.Parameter.VAR_POSITIONAL:
                arity = 4
                break
        if arity < 1 or arity > 4:
            raise ValueError(
                f"{name} must take 1 to 4 positional arguments: "
                "(x), (x, t), (x, t, args), or (x, t, args, p)"
            )
    if arity == 1:
        return lambda x, t, args, p: f(x)
    if arity == 2:
        return lambda x, t, args, p: f(x, t)
    if arity == 3:
        return lambda x, t, args, p: f(x, t, args)
    return f


def solve_ode(
    f,
    solver,
    t0,
    t1,
    x0,
    *,
    p=None,
    args=None,
    dt0=None,
    saveat=None,
    controller=None,
    max_steps=4096,
    project=None,
):
    """Integrate ``dx/dt = f(x, t, args, p)`` from ``t0`` to ``t1 > t0``.

    The vector field may be declared ``f(x)``, ``f(x, t)``, ``f(x, t, args)``,
    or ``f(x, t, args, p)`` — always in that order. ``x`` is an array state
    (scalar or vector), ``args`` is pass-through data that is by convention
    not an AD target, and ``p`` holds differentiable parameters (any pytree);
    jvp/vjp with respect to ``p`` and ``x0`` are first-class.

    Fixed and adaptive stepping share one bounded ``lax.scan`` of exactly
    ``max_steps`` iterations, so shapes are static and curvature-dependent
    step counts never retrace. ``dt0`` is required (no auto-initial-step
    heuristic). Each attempt is clipped to the remaining horizon; the clipped
    step also feeds the controller's next-step proposal, which doubles as the
    growth guard — near-flat fields otherwise grow steps into quarter-horizon
    leaps. ``project`` (e.g. a positivity clamp, assumed idempotent) is
    applied at every point where ``f`` is evaluated and to every accepted
    state. Returns a :class:`Solution`; ``sol.ok`` is False if the budget ran
    out before ``t1`` (outputs are then the reached prefix, never poisoned).

    The time dtype follows ``jnp.result_type(x0, float)``; the library never
    sets ``jax_enable_x64`` — do that in your application.
    """
    if dt0 is None:
        raise ValueError("dt0 is required (tinydiffeq has no initial-step heuristic)")
    if saveat is None:
        saveat = SaveAt(t1=True)
    if controller is None:
        controller = ConstantStepSize()
    if project is None:
        project = identity_project
    if controller.uses_error_estimate and not solver.has_error_estimate:
        raise ValueError(
            f"{type(controller).__name__} needs an embedded error estimate, "
            f"which {type(solver).__name__} does not provide"
        )
    f = canonicalize_field(f)

    x0 = jnp.asarray(x0)
    tdt = jnp.result_type(x0, float)
    t0 = jnp.asarray(t0, tdt)
    t1 = jnp.asarray(t1, tdt)
    dt0 = jnp.asarray(dt0, tdt)
    t_eps = 4.0 * jnp.finfo(tdt).eps * jnp.maximum(1.0, jnp.abs(t1))
    # Summing ~max_steps rounded steps can leave t short of t1 by up to
    # ~max_steps * eps, so a step whose remaining horizon is within that
    # slack of the desired dt is stretched to land on t1 exactly; otherwise
    # dt0 = (t1 - t0)/n with max_steps = n would strand a one-ulp sliver.
    t_slack = max_steps * t_eps

    def g(x, t):
        return f(project(x), t, args, p)

    need_f = solver.fsal or (saveat.ts is not None)
    f_init = g(x0, t0) if need_f else jnp.zeros_like(x0)

    def body(carry, _):
        t, x, dt, f_cur, done, num_accepted = carry
        remaining = t1 - t
        h = jnp.where(remaining <= dt + t_slack, jnp.maximum(remaining, 1e-12), dt)
        x1, f1, err = solver.step(g, t, x, h, f_cur if need_f else None, project)
        if need_f and f1 is None:
            f1 = g(x1, t + h)
        accept, dt_next = controller.adapt(x, x1, err, h, dt, solver.order)
        advance = accept & ~done
        x_new = jnp.where(advance, x1, x)
        t_new = jnp.where(advance, t + h, t)
        f_new = jnp.where(advance, f1, f_cur) if need_f else f_cur
        dt_new = jnp.where(done, dt, dt_next)
        done_new = done | (t_new >= t1 - t_eps)
        num_new = num_accepted + advance.astype(jnp.int32)
        carry_new = (t_new, x_new, dt_new, f_new, done_new, num_new)
        if saveat.t1:
            out = None
        elif saveat.steps:
            out = (t_new, x_new, advance)
        else:
            out = (t_new, x_new, f_new, advance)
        return carry_new, out

    carry0 = (t0, x0, dt0, f_init, jnp.asarray(False), jnp.asarray(0, jnp.int32))
    (t_final, x_final, _, _, done, num_accepted), rows = jax.lax.scan(
        body, carry0, None, length=max_steps
    )

    if saveat.t1:
        return Solution(ts=t_final, xs=x_final, ok=done, num_accepted=num_accepted)

    if saveat.steps:
        ts_s, xs_s, adv_s = rows
    else:
        ts_s, xs_s, fs_s, adv_s = rows
    ts_all = jnp.concatenate([t0[None], ts_s])
    xs_all = jnp.concatenate([x0[None], xs_s])
    accepted = jnp.concatenate([jnp.ones((1,), bool), adv_s])

    if saveat.steps:
        if saveat.fill == "inf":
            ts_all = jnp.where(accepted, ts_all, jnp.inf)
            keep = accepted.reshape(accepted.shape + (1,) * (xs_all.ndim - 1))
            xs_all = jnp.where(keep, xs_all, jnp.inf)
        return Solution(
            ts=ts_all,
            xs=xs_all,
            ok=done,
            num_accepted=num_accepted,
            accepted=accepted,
        )

    fs_all = jnp.concatenate([f_init[None], fs_s])
    ts_query = jnp.asarray(saveat.ts, tdt)
    xs_query = hermite_interpolate(ts_query, ts_all, xs_all, fs_all)
    return Solution(ts=ts_query, xs=xs_query, ok=done, num_accepted=num_accepted)
