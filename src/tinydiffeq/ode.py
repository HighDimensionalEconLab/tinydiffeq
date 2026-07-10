import inspect

import jax
import jax.numpy as jnp

from tinydiffeq.controllers import ConstantStepSize
from tinydiffeq.interpolation import hermite_interpolate
from tinydiffeq.save_at import SaveAt
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
    t_0,
    t_1,
    x_0,
    *,
    p=None,
    args=None,
    dt_0=None,
    save_at=None,
    controller=None,
    max_steps=4096,
    project=None,
):
    """Integrate ``dx/dt = f(x, t, args, p)`` from ``t_0`` to ``t_1 > t_0``.

    The vector field may be declared ``f(x)``, ``f(x, t)``, ``f(x, t, args)``,
    or ``f(x, t, args, p)`` — always in that order. ``x`` is an array state
    (scalar or vector), ``args`` is pass-through data that is by convention
    not an AD target, and ``p`` holds differentiable parameters (any pytree);
    jvp/vjp with respect to ``p`` and ``x_0`` are first-class.

    Fixed and adaptive stepping share one bounded ``lax.scan`` of exactly
    ``max_steps`` iterations, so shapes are static and curvature-dependent
    step counts never retrace. Once ``t_1`` is reached, a scalar ``lax.cond``
    skips the expensive solver and controller work while the fixed scan emits
    its padded tail. ``dt_0`` is required (no auto-initial-step heuristic).
    Each attempt is clipped to the remaining horizon; the clipped step also
    feeds the controller's next-step proposal, which doubles as the growth
    guard — near-flat fields otherwise grow steps into quarter-horizon leaps.
    ``project`` (e.g. a positivity clamp, assumed idempotent) is applied at
    every point where ``f`` is evaluated and to every accepted state. Returns
    a :class:`Solution`; ``sol.ok`` is False if the budget ran out before
    ``t_1`` (outputs are then the reached prefix, never poisoned).

    The time dtype follows ``jnp.result_type(x_0, float)``; the library never
    sets ``jax_enable_x64`` — do that in your application.
    """
    if dt_0 is None:
        raise ValueError("dt_0 is required (tinydiffeq has no initial-step heuristic)")
    if save_at is None:
        save_at = SaveAt(t_1=True)
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

    x_0 = jnp.asarray(x_0)
    time_dtype = jnp.result_type(x_0, float)
    t_0 = jnp.asarray(t_0, time_dtype)
    t_1 = jnp.asarray(t_1, time_dtype)
    dt_0 = jnp.asarray(dt_0, time_dtype)
    positive_time_floor = jnp.asarray(jnp.finfo(time_dtype).tiny, time_dtype)
    t_eps = 4.0 * jnp.finfo(time_dtype).eps * jnp.maximum(1.0, jnp.abs(t_1))
    # Summing ~max_steps rounded steps can leave t short of t_1 by up to
    # ~max_steps * eps, so a step whose remaining horizon is within that
    # slack of the desired dt is stretched to land on t_1 exactly; otherwise
    # dt_0 = (t_1 - t_0)/n with max_steps = n would strand a one-ulp sliver.
    t_slack = max_steps * t_eps

    def g(x, t):
        return f(project(x), t, args, p)

    need_f = solver.fsal or (save_at.ts is not None)
    f_init = g(x_0, t_0) if need_f else jnp.zeros_like(x_0)
    controller_state_init = controller.init(x_0)

    def attempt_step(carry):
        t, x, dt, f_cur, done, num_accepted, controller_state = carry
        remaining = t_1 - t
        h = jnp.where(
            remaining <= dt + t_slack,
            jnp.maximum(remaining, positive_time_floor),
            dt,
        )
        x_1, f_1, err = solver.step(g, t, x, h, f_cur if need_f else None, project)
        if need_f and f_1 is None:
            f_1 = g(x_1, t + h)
        accept, dt_next, controller_state_next = controller.adapt(
            x, x_1, err, h, dt, solver.order, controller_state, t_1
        )
        advance = accept & ~done
        x_new = jnp.where(advance, x_1, x)
        t_new = jnp.where(advance, t + h, t)
        f_new = jnp.where(advance, f_1, f_cur) if need_f else f_cur
        dt_new = jnp.where(done, dt, dt_next)
        controller_state_new = jax.tree.map(
            lambda old, new: jnp.where(done, old, new),
            controller_state,
            controller_state_next,
        )
        done_new = done | (t_new >= t_1 - t_eps)
        num_new = num_accepted + advance.astype(jnp.int32)
        carry_new = (
            t_new,
            x_new,
            dt_new,
            f_new,
            done_new,
            num_new,
            controller_state_new,
        )
        if save_at.t_1:
            out = None
        elif save_at.steps:
            out = (t_new, x_new, advance)
        else:
            out = (t_new, x_new, f_new, advance)
        return carry_new, out

    def skip_step(carry):
        t, x, _, f_cur, _, _, _ = carry
        if save_at.t_1:
            out = None
        elif save_at.steps:
            out = (t, x, jnp.asarray(False))
        else:
            out = (t, x, f_cur, jnp.asarray(False))
        return carry, out

    def body(carry, _):
        return jax.lax.cond(carry[4], skip_step, attempt_step, carry)

    carry_0 = (
        t_0,
        x_0,
        dt_0,
        f_init,
        jnp.asarray(False),
        jnp.asarray(0, jnp.int32),
        controller_state_init,
    )
    (t_final, x_final, _, _, done, num_accepted, _), rows = jax.lax.scan(
        body, carry_0, None, length=max_steps
    )

    if save_at.t_1:
        return Solution(ts=t_final, xs=x_final, ok=done, num_accepted=num_accepted)

    if save_at.steps:
        ts_s, xs_s, adv_s = rows
    else:
        ts_s, xs_s, fs_s, adv_s = rows
    all_times = jnp.concatenate([t_0[None], ts_s])
    all_states = jnp.concatenate([x_0[None], xs_s])
    raw_accepted = jnp.concatenate([jnp.ones((1,), bool), adv_s])

    if save_at.steps:
        output_size = max_steps + 1
        accepted_indices = jnp.nonzero(raw_accepted, size=output_size, fill_value=0)[0]
        compact_times = all_times[accepted_indices]
        compact_states = all_states[accepted_indices]
        accepted = jnp.arange(output_size) <= num_accepted
        last_time = compact_times[num_accepted]
        last_state = compact_states[num_accepted]
        state_mask = accepted.reshape(accepted.shape + (1,) * (compact_states.ndim - 1))
        if save_at.fill == "inf":
            output_times = jnp.where(accepted, compact_times, jnp.inf)
            output_states = jnp.where(state_mask, compact_states, jnp.inf)
        else:
            output_times = jnp.where(accepted, compact_times, last_time)
            output_states = jnp.where(state_mask, compact_states, last_state)
        return Solution(
            ts=output_times,
            xs=output_states,
            ok=done,
            num_accepted=num_accepted,
            accepted=accepted,
        )

    fs_all = jnp.concatenate([f_init[None], fs_s])
    query_times = jnp.asarray(save_at.ts, time_dtype)
    query_states = hermite_interpolate(query_times, all_times, all_states, fs_all)
    return Solution(ts=query_times, xs=query_states, ok=done, num_accepted=num_accepted)
