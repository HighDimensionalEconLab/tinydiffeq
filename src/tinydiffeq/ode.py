import inspect

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from tinydiffeq._rodas5p import rodas5p_step
from tinydiffeq._tree import (
    asarray_state,
    assert_same_structure,
    fill_rows,
    prepend,
    take,
    where,
    zeros_like,
)
from tinydiffeq.controllers import ConstantStepSize
from tinydiffeq.interpolation import hermite_interpolate, rodas_interpolate
from tinydiffeq.save_at import SaveAt
from tinydiffeq.solution import Solution
from tinydiffeq.solvers import Rodas5P

ADAPTIVE_SCAN_CHUNK_SIZE = 16


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
    or ``f(x, t, args, p)`` — always in that order. ``x`` is a pytree whose
    nonempty leaves share one real floating dtype. ``args`` is pass-through
    data that is by convention not an AD target, and ``p`` holds differentiable
    parameters (any pytree);
    jvp/vjp with respect to ``p`` and ``x_0`` are first-class.

    Fixed and adaptive stepping use bounded ``lax.scan`` loops with exactly
    ``max_steps`` attempt slots, so shapes are static and curvature-dependent
    step counts never retrace. Adaptive attempts are grouped into static
    chunks so one ``lax.cond`` skips an entire padded chunk after completion.
    ``dt_0`` is required (no auto-initial-step heuristic).
    Each attempt is clipped to the remaining horizon; the clipped step also
    feeds the controller's next-step proposal, which doubles as the growth
    guard — near-flat fields otherwise grow steps into quarter-horizon leaps.
    ``project`` (e.g. a positivity clamp, assumed idempotent) is applied at
    every point where ``f`` is evaluated and to every accepted state. Returns
    a :class:`Solution`; ``sol.ok`` is False if the budget ran out before
    ``t_1`` (outputs are then the reached prefix, never poisoned).

    The time dtype follows the state dtype; the library never
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
    is_rodas = isinstance(solver, Rodas5P)
    is_fixed = isinstance(controller, ConstantStepSize)

    x_0, time_dtype = asarray_state(x_0, "x_0")
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

    def project_state(x):
        value, dtype = asarray_state(project(x), "project(x)")
        assert_same_structure(x_0, value, "project(x)")
        if dtype != time_dtype:
            raise TypeError("project(x) must preserve the state dtype")
        return value

    def g(x, t):
        value, dtype = asarray_state(f(project_state(x), t, args, p), "f(x, t)")
        assert_same_structure(x_0, value, "f(x, t)")
        if dtype != time_dtype:
            raise TypeError("f(x, t) must preserve the state dtype")
        return value

    need_f = not is_rodas and (solver.fsal or (save_at.ts is not None))
    f_init = g(x_0, t_0) if need_f else zeros_like(x_0)
    controller_state_init = controller.init(x_0)
    flat_x_0, _ = ravel_pytree(x_0)
    identity_mass = jnp.ones_like(flat_x_0)

    def attempt_step(carry):
        t, x, dt, f_cur, done, failed, num_accepted, controller_state = carry
        remaining = t_1 - t
        h = jnp.where(
            remaining <= dt + t_slack,
            jnp.maximum(remaining, positive_time_floor),
            dt,
        )
        if is_rodas:
            x_1, err, dense, step_ok = rodas5p_step(
                g, t, x, h, identity_mass, project_state
            )
            f_1 = f_cur
        else:
            step = solver.step_fixed if is_fixed else solver.step
            x_1, f_1, err = step(g, t, x, h, f_cur if need_f else None, project_state)
            if need_f and f_1 is None:
                f_1 = g(x_1, t + h)
            dense = None
            step_ok = jnp.asarray(True)
        if is_rodas:
            control_err = where(
                step_ok,
                err,
                jax.tree.map(lambda value: jnp.full_like(value, jnp.inf), err),
            )
        else:
            control_err = err
        accept, dt_next, controller_state_next = controller.adapt(
            x, x_1, control_err, h, dt, solver.order, controller_state, t_1
        )
        advance = accept & step_ok & ~done & ~failed
        x_new = where(advance, x_1, x)
        t_new = jnp.where(advance, t + h, t)
        f_new = where(advance, f_1, f_cur) if need_f else f_cur
        dt_new = jnp.where(done | failed, dt, dt_next)
        controller_state_new = jax.tree.map(
            lambda old, new: jnp.where(done | failed | ~step_ok, old, new),
            controller_state,
            controller_state_next,
        )
        done_new = done | (t_new >= t_1 - t_eps)
        failed_new = failed if controller.uses_error_estimate else failed | ~step_ok
        num_new = num_accepted + advance.astype(jnp.int32)
        carry_new = (
            t_new,
            x_new,
            dt_new,
            f_new,
            done_new,
            failed_new,
            num_new,
            controller_state_new,
        )
        if save_at.t_1:
            out = None
        elif save_at.steps:
            out = (t_new, x_new, advance)
        elif is_rodas:
            out = (t_new, x_new, dense, advance)
        else:
            out = (t_new, x_new, f_new, advance)
        return carry_new, out

    def skip_step(carry):
        t, x, _, f_cur, _, _, _, _ = carry
        if save_at.t_1:
            out = None
        elif save_at.steps:
            out = (t, x, jnp.asarray(False))
        elif is_rodas:
            out = (
                t,
                x,
                (zeros_like(x), zeros_like(x), zeros_like(x)),
                jnp.asarray(False),
            )
        else:
            out = (t, x, f_cur, jnp.asarray(False))
        return carry, out

    def body(carry, _):
        return jax.lax.cond(carry[4] | carry[5], skip_step, attempt_step, carry)

    def fixed_attempt_step(carry):
        t, x, f_cur, done, num_accepted = carry
        remaining = t_1 - t
        h = jnp.where(
            remaining <= dt_0 + t_slack,
            jnp.maximum(remaining, positive_time_floor),
            dt_0,
        )
        x_1, f_1, _ = solver.step_fixed(
            g, t, x, h, f_cur if need_f else None, project_state
        )
        if need_f and f_1 is None:
            f_1 = g(x_1, t + h)
        t_1_step = t + h
        done_1 = t_1_step >= t_1 - t_eps
        carry_1 = (t_1_step, x_1, f_1 if need_f else f_cur, done_1, num_accepted + 1)
        if save_at.t_1:
            output = None
        elif save_at.steps:
            output = (t_1_step, x_1, jnp.asarray(True))
        else:
            output = (t_1_step, x_1, f_1, jnp.asarray(True))
        return carry_1, output

    def fixed_skip_step(carry):
        t, x, f_cur, _, _ = carry
        if save_at.t_1:
            output = None
        elif save_at.steps:
            output = (t, x, jnp.asarray(False))
        else:
            output = (t, x, f_cur, jnp.asarray(False))
        return carry, output

    def fixed_body(carry, _):
        return jax.lax.cond(carry[3], fixed_skip_step, fixed_attempt_step, carry)

    def bounded_adaptive_scan(carry):
        chunk_size = min(ADAPTIVE_SCAN_CHUNK_SIZE, max_steps)
        num_chunks = (max_steps + chunk_size - 1) // chunk_size
        padded_steps = num_chunks * chunk_size
        valid = (jnp.arange(padded_steps) < max_steps).reshape(num_chunks, chunk_size)

        def repeat_output(output):
            if output is None:
                return None
            return jax.tree.map(
                lambda value: jnp.broadcast_to(value, (chunk_size,) + value.shape),
                output,
            )

        def run_chunk(chunk_carry, chunk_valid):
            def inner(inner_carry, is_valid):
                return jax.lax.cond(
                    is_valid,
                    body,
                    lambda value, _: skip_step(value),
                    inner_carry,
                    None,
                )

            return jax.lax.scan(inner, chunk_carry, chunk_valid, unroll=4)

        def skip_chunk(chunk_carry, chunk_valid):
            _, output = skip_step(chunk_carry)
            return chunk_carry, repeat_output(output)

        def outer(chunk_carry, chunk_valid):
            inactive = chunk_carry[4] | chunk_carry[5]
            return jax.lax.cond(
                inactive, skip_chunk, run_chunk, chunk_carry, chunk_valid
            )

        final_carry, chunk_rows = jax.lax.scan(outer, carry, valid)
        if chunk_rows is None:
            return final_carry, None
        rows = jax.tree.map(
            lambda value: value.reshape((padded_steps,) + value.shape[2:])[:max_steps],
            chunk_rows,
        )
        return final_carry, rows

    carry_0 = (
        t_0,
        x_0,
        dt_0,
        f_init,
        jnp.asarray(False),
        jnp.asarray(False),
        jnp.asarray(0, jnp.int32),
        controller_state_init,
    )
    if is_fixed and not is_rodas:
        fixed_carry_0 = (
            t_0,
            x_0,
            f_init,
            jnp.asarray(False),
            jnp.asarray(0, jnp.int32),
        )
        fixed_final, rows = jax.lax.scan(
            fixed_body, fixed_carry_0, None, length=max_steps
        )
        t_final, x_final, _, done, num_accepted = fixed_final
        failed = jnp.asarray(False)
    elif controller.uses_error_estimate:
        final_carry, rows = bounded_adaptive_scan(carry_0)
        (t_final, x_final, _, _, done, failed, num_accepted, _) = final_carry
    else:
        final_carry, rows = jax.lax.scan(body, carry_0, None, length=max_steps)
        (t_final, x_final, _, _, done, failed, num_accepted, _) = final_carry
    integration_ok = done & ~failed

    if save_at.t_1:
        return Solution(
            ts=t_final,
            xs=x_final,
            ok=integration_ok,
            num_accepted=num_accepted,
        )

    if save_at.steps:
        ts_s, xs_s, adv_s = rows
    elif is_rodas:
        ts_s, xs_s, dense_s, adv_s = rows
    else:
        ts_s, xs_s, fs_s, adv_s = rows
    all_times = jnp.concatenate([t_0[None], ts_s])
    all_states = prepend(x_0, xs_s)
    raw_accepted = jnp.concatenate([jnp.ones((1,), bool), adv_s])

    if save_at.steps:
        output_size = max_steps + 1
        accepted_indices = jnp.nonzero(raw_accepted, size=output_size, fill_value=0)[0]
        compact_times = all_times[accepted_indices]
        compact_states = take(all_states, accepted_indices)
        accepted = jnp.arange(output_size) <= num_accepted
        last_time = compact_times[num_accepted]
        last_state = take(compact_states, num_accepted)
        if save_at.fill == "inf":
            output_times = jnp.where(accepted, compact_times, jnp.inf)
        else:
            output_times = jnp.where(accepted, compact_times, last_time)
        output_states = fill_rows(compact_states, accepted, last_state, save_at.fill)
        return Solution(
            ts=output_times,
            xs=output_states,
            ok=integration_ok,
            num_accepted=num_accepted,
            accepted=accepted,
        )

    query_times = jnp.asarray(save_at.ts, time_dtype)
    if is_rodas:
        query_states = rodas_interpolate(query_times, all_times, all_states, dense_s)
    else:
        fs_all = prepend(f_init, fs_s)
        query_states = hermite_interpolate(query_times, all_times, all_states, fs_all)
    return Solution(
        ts=query_times,
        xs=query_states,
        ok=integration_ok,
        num_accepted=num_accepted,
    )
