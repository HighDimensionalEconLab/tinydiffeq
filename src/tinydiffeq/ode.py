import inspect

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from tinydiffeq._aux import (
    make_safe_evaluator,
    prepare_aux_reference,
    resolve_field_aux,
    split_field_output,
    zeros_from_shape,
)
from tinydiffeq._rodas5p import rodas5p_step, rodas_dense_endpoint_derivatives
from tinydiffeq._tree import (
    asarray_state,
    assert_same_structure,
    fill_rows,
    prepend,
    take,
    where,
    zero_tangent,
    zeros_like,
)
from tinydiffeq.controllers import ConstantStepSize
from tinydiffeq.interpolation import (
    hermite_interpolate,
    hermite_interval_interpolate,
    rodas_interpolate,
)
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
    has_aux=None,
    failure_ad_reference=None,
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
    ``t_1`` or a required saved output was invalid (outputs remain a finite
    reached prefix or endpoint, never poisoned).

    The field may return either ``dx`` or ``(dx, aux)``. ``has_aux=None``
    detects the form with an abstract trace; ``has_aux=False`` selects the
    minimal no-aux path without that trace. Saved aux is a nonempty pytree of
    real floating arrays. It follows ``SaveAt`` and participates in JVP/VJP.
    Requested-grid aux uses cubic Hermite interpolation with endpoint slopes
    obtained by JVP, including for Rodas5P's dense state path.

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

    has_aux, aux_shape = resolve_field_aux(
        f,
        (project_state(x_0), t_0, args, p),
        jax.tree.structure(x_0),
        has_aux,
        name="has_aux",
    )

    def field_output(x, t, p_value):
        return f(project_state(x), t, args, p_value)

    def g(x, t):
        output = field_output(x, t, p)
        value, _ = split_field_output(output, has_aux)
        value, dtype = asarray_state(value, "f(x, t)")
        assert_same_structure(x_0, value, "f(x, t)")
        if dtype != time_dtype:
            raise TypeError("f(x, t) must preserve the state dtype")
        return value

    if has_aux:
        failure_ad_reference = prepare_aux_reference(failure_ad_reference, x_0, t_0, p)

        def auxiliary(inputs):
            x_value, t_value, p_value = inputs
            output = field_output(x_value, t_value, p_value)
            return split_field_output(output, True)[1]

        evaluate_aux = make_safe_evaluator(auxiliary, aux_shape)
        zero_aux = zeros_from_shape(aux_shape)

        def aux_value_and_derivative(x, t, x_dot, active):
            (value, ok), (value_dot, _) = jax.jvp(
                lambda inputs: evaluate_aux(inputs, active, failure_ad_reference),
                ((x, t, p),),
                (((x_dot, jnp.ones_like(t), zero_tangent(p))),),
            )
            return value, ok, value_dot

    else:
        evaluate_aux = None
        zero_aux = None

    need_f = not is_rodas and (solver.fsal or (save_at.ts is not None))
    f_init = g(x_0, t_0) if need_f else zeros_like(x_0)
    track_aux = has_aux and not save_at.t_1
    if track_aux:
        aux_init, aux_init_ok = evaluate_aux(
            (x_0, t_0, p), jnp.asarray(True), failure_ad_reference
        )
        if save_at.ts is not None and not is_rodas:
            aux_init, aux_init_ok, aux_dot_init = aux_value_and_derivative(
                x_0, t_0, f_init, aux_init_ok
            )
        else:
            aux_dot_init = zero_aux
    else:
        aux_init = None
        aux_dot_init = None
        aux_init_ok = jnp.asarray(True)
    controller_state_init = controller.init(x_0)
    flat_x_0, _ = ravel_pytree(x_0)
    identity_mass = jnp.ones_like(flat_x_0)

    def attempt_step(carry):
        (
            t,
            x,
            aux,
            aux_dot,
            dt,
            f_cur,
            done,
            failed,
            num_accepted,
            controller_state,
        ) = carry
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
        provisional_advance = accept & step_ok & ~done & ~failed
        if track_aux:

            def accepted_auxiliary():
                if save_at.ts is None:
                    aux_candidate, aux_ok = evaluate_aux(
                        (x_1, t + h, p),
                        provisional_advance,
                        failure_ad_reference,
                    )
                    return aux_candidate, aux_ok, zero_aux, zero_aux
                if is_rodas:
                    left_dot, right_dot = rodas_dense_endpoint_derivatives(
                        x, x_1, dense, h
                    )
                    _, _, aux_left_dot = aux_value_and_derivative(
                        x, t, left_dot, provisional_advance
                    )
                    aux_candidate, aux_ok, aux_right_dot = aux_value_and_derivative(
                        x_1, t + h, right_dot, provisional_advance
                    )
                    return aux_candidate, aux_ok, aux_left_dot, aux_right_dot
                aux_candidate, aux_ok, aux_right_dot = aux_value_and_derivative(
                    x_1, t + h, f_1, provisional_advance
                )
                return aux_candidate, aux_ok, aux_dot, aux_right_dot

            aux_candidate, aux_ok, aux_left_dot, aux_right_dot = jax.lax.cond(
                provisional_advance,
                accepted_auxiliary,
                lambda: (aux, jnp.asarray(True), zero_aux, zero_aux),
            )
        else:
            aux_candidate = None
            aux_ok = jnp.asarray(True)
            aux_left_dot = None
            aux_right_dot = None
        advance = provisional_advance & aux_ok
        x_new = where(advance, x_1, x)
        aux_new = where(advance, aux_candidate, aux) if track_aux else None
        aux_dot_new = (
            where(advance, aux_right_dot, aux_dot)
            if track_aux and save_at.ts is not None and not is_rodas
            else aux_dot
        )
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
        failed_new = failed_new | (provisional_advance & ~aux_ok)
        num_new = num_accepted + advance.astype(jnp.int32)
        carry_new = (
            t_new,
            x_new,
            aux_new,
            aux_dot_new,
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
            out = (t_new, x_new, aux_new, advance)
        elif is_rodas:
            out = (
                t_new,
                x_new,
                aux_new,
                dense,
                aux_left_dot,
                aux_right_dot,
                advance,
            )
        else:
            out = (t_new, x_new, aux_new, f_new, aux_dot_new, advance)
        return carry_new, out

    def skip_step(carry):
        t, x, aux, aux_dot, _, f_cur, _, _, _, _ = carry
        if save_at.t_1:
            out = None
        elif save_at.steps:
            out = (t, x, aux, jnp.asarray(False))
        elif is_rodas:
            out = (
                t,
                x,
                aux,
                (zeros_like(x), zeros_like(x), zeros_like(x)),
                zero_aux,
                zero_aux,
                jnp.asarray(False),
            )
        else:
            out = (t, x, aux, f_cur, aux_dot, jnp.asarray(False))
        return carry, out

    def body(carry, _):
        return jax.lax.cond(carry[6] | carry[7], skip_step, attempt_step, carry)

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
            inactive = chunk_carry[6] | chunk_carry[7]
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
        aux_init,
        aux_dot_init,
        dt_0,
        f_init,
        jnp.asarray(False),
        ~aux_init_ok,
        jnp.asarray(0, jnp.int32),
        controller_state_init,
    )
    use_fast_fixed = is_fixed and not is_rodas and not track_aux
    if use_fast_fixed:
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
        (t_final, x_final, _, _, _, _, done, failed, num_accepted, _) = final_carry
    else:
        final_carry, rows = jax.lax.scan(body, carry_0, None, length=max_steps)
        (t_final, x_final, _, _, _, _, done, failed, num_accepted, _) = final_carry
    integration_ok = done & ~failed

    if save_at.t_1:
        if has_aux:
            aux_final, aux_ok = evaluate_aux(
                (x_final, t_final, p), jnp.asarray(True), failure_ad_reference
            )
        else:
            aux_final = None
            aux_ok = jnp.asarray(True)
        return Solution(
            ts=t_final,
            xs=x_final,
            ok=integration_ok & aux_ok,
            num_accepted=num_accepted,
            aux=aux_final,
        )

    if save_at.steps:
        if use_fast_fixed:
            ts_s, xs_s, adv_s = rows
            aux_s = None
        else:
            ts_s, xs_s, aux_s, adv_s = rows
    elif is_rodas:
        (
            ts_s,
            xs_s,
            aux_s,
            dense_s,
            aux_left_dots_s,
            aux_right_dots_s,
            adv_s,
        ) = rows
    else:
        if use_fast_fixed:
            ts_s, xs_s, fs_s, adv_s = rows
            aux_s = None
            aux_dots_s = None
        else:
            ts_s, xs_s, aux_s, fs_s, aux_dots_s, adv_s = rows
    all_times = jnp.concatenate([t_0[None], ts_s])
    all_states = prepend(x_0, xs_s)
    all_aux = prepend(aux_init, aux_s) if has_aux else None
    raw_accepted = jnp.concatenate([jnp.ones((1,), bool), adv_s])

    if save_at.steps:
        output_size = max_steps + 1
        accepted_indices = jnp.nonzero(raw_accepted, size=output_size, fill_value=0)[0]
        compact_times = all_times[accepted_indices]
        compact_states = take(all_states, accepted_indices)
        compact_aux = take(all_aux, accepted_indices) if has_aux else None
        accepted = jnp.arange(output_size) <= num_accepted
        last_time = compact_times[num_accepted]
        last_state = take(compact_states, num_accepted)
        last_aux = take(compact_aux, num_accepted) if has_aux else None
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
            aux=(
                fill_rows(compact_aux, accepted, last_aux, save_at.fill)
                if has_aux
                else None
            ),
        )

    query_times = jnp.asarray(save_at.ts, time_dtype)
    if is_rodas:
        query_states = rodas_interpolate(query_times, all_times, all_states, dense_s)
        query_aux = (
            hermite_interval_interpolate(
                query_times,
                all_times,
                all_aux,
                aux_left_dots_s,
                aux_right_dots_s,
            )
            if has_aux
            else None
        )
    else:
        fs_all = prepend(f_init, fs_s)
        query_states = hermite_interpolate(query_times, all_times, all_states, fs_all)
        aux_dots_all = prepend(aux_dot_init, aux_dots_s) if has_aux else None
        query_aux = (
            hermite_interpolate(query_times, all_times, all_aux, aux_dots_all)
            if has_aux
            else None
        )
    return Solution(
        ts=query_times,
        xs=query_states,
        ok=integration_ok,
        num_accepted=num_accepted,
        aux=query_aux,
    )
