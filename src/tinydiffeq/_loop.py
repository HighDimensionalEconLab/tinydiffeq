"""Shared actual-work loop utilities for adaptive differential equations."""

import jax
import jax.numpy as jnp


def select_step(
    t,
    t_0,
    t_1,
    dt,
    dt_0,
    step_count,
    *,
    constant,
    time_tolerance,
):
    """Choose a forward step without making it depend on the attempt budget.

    Constant-step times are formed arithmetically from the accepted-step index,
    avoiding drift from repeated addition. A nominal endpoint within a small,
    local floating-point tolerance is corrected to ``t_1``. Adaptive steps are
    never enlarged: they use ``min(dt, t_1 - t)`` and snap the bookkeeping time
    only when that exact remaining interval was integrated.

    In particular, ``time_tolerance`` must not scale with ``max_steps``. Such a
    tolerance can turn a nonbinding attempt budget into a change in the
    numerical method by stretching an earlier step to the horizon.
    """
    dtype = jnp.result_type(t, t_0, t_1, dt, dt_0)
    positive_floor = jnp.asarray(jnp.finfo(dtype).tiny, dtype)
    if constant:
        snap_tolerance = jnp.minimum(
            jnp.asarray(time_tolerance, dtype),
            0.25 * jnp.abs(jnp.asarray(dt_0, dtype)),
        )
        next_index = jnp.asarray(step_count + 1, dtype)
        nominal_next = t_0 + next_index * dt_0
        reaches_horizon = (nominal_next >= t_1) | (
            jnp.abs(nominal_next - t_1) <= snap_tolerance
        )
        t_next = jnp.where(reaches_horizon, t_1, nominal_next)
        h = jnp.maximum(t_next - t, positive_floor)
        return h, t_next, reaches_horizon

    remaining = t_1 - t
    reaches_horizon = remaining <= dt
    h = jnp.where(reaches_horizon, jnp.maximum(remaining, positive_floor), dt)
    t_next = jnp.where(reaches_horizon, t_1, t + h)
    return h, t_next, reaches_horizon


def forward_adaptive_while(
    carry,
    *,
    attempt_step,
    skip_step,
    terminated,
    max_steps,
):
    """Run adaptive attempts until termination, retaining static output shapes.

    ``attempt_step`` and ``skip_step`` share the scan-style contract
    ``(carry, output)``. Outputs receive a fixed leading ``max_steps`` buffer,
    but only actual attempts execute. Under ``vmap``, JAX's while batching rule
    advances lanes in lockstep until every lane is terminated and freezes lanes
    that completed earlier.

    Dynamic while loops support primal evaluation and forward-mode AD, but JAX
    cannot transpose them. Callers must expose that boundary in their API.
    """
    _, initial_output = skip_step(carry)
    if initial_output is None:
        initial_rows = None
    else:
        initial_rows = jax.tree.map(
            lambda value: jnp.broadcast_to(value, (max_steps,) + value.shape),
            initial_output,
        )

    def condition(loop_state):
        attempt, loop_carry, _ = loop_state
        return (attempt < max_steps) & ~terminated(loop_carry)

    def run_attempt(loop_state):
        attempt, loop_carry, loop_rows = loop_state
        next_carry, output = attempt_step(loop_carry)
        if loop_rows is not None:
            loop_rows = jax.tree.map(
                lambda rows, value: rows.at[attempt].set(value),
                loop_rows,
                output,
            )
        return attempt + 1, next_carry, loop_rows

    attempt, final_carry, rows = jax.lax.while_loop(
        condition,
        run_attempt,
        (jnp.asarray(0, jnp.int32), carry, initial_rows),
    )
    if rows is not None:
        _, final_output = skip_step(final_carry)
        attempted = jnp.arange(max_steps) < attempt

        def fill_unattempted(values, final_value):
            mask = attempted.reshape(
                attempted.shape + (1,) * (values.ndim - attempted.ndim)
            )
            return jnp.where(mask, values, final_value)

        rows = jax.tree.map(fill_unattempted, rows, final_output)
    return final_carry, rows
