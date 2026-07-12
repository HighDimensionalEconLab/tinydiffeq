import jax
import jax.numpy as jnp

from tinydiffeq._aux import (
    make_safe_evaluator,
    prepare_aux_reference,
    resolve_field_aux,
    split_field_output,
)
from tinydiffeq._tree import (
    asarray_state,
    assert_same_structure,
    fill_rows,
    prepend,
    take,
    where,
)
from tinydiffeq.ode import canonicalize_field, identity_project
from tinydiffeq.save_at import SaveAt
from tinydiffeq.solution import Solution


def _diagonal_brownian_increments(x_0, key, n_steps, dt, dtype):
    """Generate one diagonal-noise draw for an array or pytree state."""
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
    has_aux=None,
    failure_ad_reference=None,
):
    """Integrate the Ito SDE ``dx = drift dt + diffusion d_w`` (diagonal noise)
    on the fixed grid of ``n_steps`` uniform steps from ``t_0`` to ``t_1 > t_0``.

    ``drift`` and ``diffusion`` follow the same signature convention as
    ``solve_ode`` — ``(x)``, ``(x, t)``, ``(x, t, args)``, or
    ``(x, t, args, p)``. ``n_steps`` must be a static Python int (the honest
    static-shape contract; there is currently no adaptive SDE stepping). The
    Brownian increments are presampled from ``key``. Arrays retain the exact
    ``(n_steps,) + x_0.shape`` random draw. Pytree states use one shared flat
    draw, partitioned into leaves in JAX's deterministic pytree leaf order.
    Thus a fixed key gives a fixed noise process: reproducible across calls
    and differentiable with respect to ``x_0`` and ``p`` (not ``key``).

    ``SaveAt(ts=...)`` raises — cubic Hermite interpolation is wrong for
    rough paths; use ``t_1`` (default) or ``steps`` (here ``n_steps + 1``
    rows unless a saved-aux failure terminates the accepted prefix).

    ``drift`` may return either its value or ``(value, aux)``. The optional
    real-floating aux pytree is stored at the same fixed nodes as ``xs`` and
    is differentiated pathwise under the fixed random key. ``diffusion`` is
    value-only. ``has_aux=None`` auto-detects the drift form;
    ``has_aux=False`` avoids the abstract detection trace and selects the
    original no-aux scan.
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

    x_0, time_dtype = asarray_state(x_0, "x_0")
    t_0 = jnp.asarray(t_0, time_dtype)
    t_1 = jnp.asarray(t_1, time_dtype)
    dt = (t_1 - t_0) / n_steps
    d_w = _diagonal_brownian_increments(x_0, key, n_steps, dt, time_dtype)
    time_grid = jnp.linspace(t_0, t_1, n_steps + 1)

    def project_state(x):
        value, dtype = asarray_state(project(x), "project(x)")
        assert_same_structure(x_0, value, "project(x)")
        if dtype != time_dtype:
            raise TypeError("project(x) must preserve the state dtype")
        return value

    has_aux, aux_shape = resolve_field_aux(
        drift,
        (project_state(x_0), t_0, args, p),
        jax.tree.structure(x_0),
        has_aux,
        name="has_aux",
    )

    def drift_output(x, t, p_value):
        return drift(project_state(x), t, args, p_value)

    def g_drift(x, t):
        output = drift_output(x, t, p)
        value, _ = split_field_output(output, has_aux)
        value, dtype = asarray_state(value, "drift(x, t)")
        assert_same_structure(x_0, value, "drift(x, t)")
        if dtype != time_dtype:
            raise TypeError("drift(x, t) must preserve the state dtype")
        return value

    def g_diffusion(x, t):
        value, dtype = asarray_state(
            diffusion(project_state(x), t, args, p), "diffusion(x, t)"
        )
        assert_same_structure(x_0, value, "diffusion(x, t)")
        if dtype != time_dtype:
            raise TypeError("diffusion(x, t) must preserve the state dtype")
        return value

    def body(x, inputs):
        t, d_w_step = inputs
        x_1 = solver.step(g_drift, g_diffusion, t, x, dt, d_w_step, project_state)
        return x_1, x_1 if save_at.steps else None

    if save_at.t_1 or not has_aux:
        x_final, step_states = jax.lax.scan(body, x_0, (time_grid[:-1], d_w))
        num_accepted = jnp.asarray(n_steps, jnp.int32)
        ok = jnp.asarray(True)

    if save_at.t_1:
        if has_aux:
            failure_ad_reference = prepare_aux_reference(
                failure_ad_reference, x_0, t_0, p
            )

            def auxiliary(inputs):
                x_value, t_value, p_value = inputs
                return split_field_output(
                    drift_output(x_value, t_value, p_value), True
                )[1]

            evaluate_aux = make_safe_evaluator(auxiliary, aux_shape)
            aux_final, aux_ok = evaluate_aux(
                (x_final, t_1, p), jnp.asarray(True), failure_ad_reference
            )
        else:
            aux_final = None
            aux_ok = jnp.asarray(True)
        return Solution(
            ts=t_1,
            xs=x_final,
            ok=ok & aux_ok,
            num_accepted=num_accepted,
            aux=aux_final,
        )

    if has_aux:
        failure_ad_reference = prepare_aux_reference(failure_ad_reference, x_0, t_0, p)

        def auxiliary(inputs):
            x_value, t_value, p_value = inputs
            return split_field_output(drift_output(x_value, t_value, p_value), True)[1]

        evaluate_aux = make_safe_evaluator(auxiliary, aux_shape)
        aux_initial, initial_ok = evaluate_aux(
            (x_0, t_0, p), jnp.asarray(True), failure_ad_reference
        )

        def aux_attempt(carry, inputs):
            x, aux, t, failed, count = carry
            t_step, t_next, d_w_step = inputs
            x_candidate = solver.step(
                g_drift,
                g_diffusion,
                t_step,
                x,
                dt,
                d_w_step,
                project_state,
            )
            aux_candidate, aux_ok = evaluate_aux(
                (x_candidate, t_next, p),
                ~failed,
                failure_ad_reference,
            )
            advance = ~failed & aux_ok
            x_new = where(advance, x_candidate, x)
            aux_new = where(advance, aux_candidate, aux)
            t_new = jnp.where(advance, t_next, t)
            failed_new = failed | ~aux_ok
            count_new = count + advance.astype(jnp.int32)
            return (
                x_new,
                aux_new,
                t_new,
                failed_new,
                count_new,
            ), (t_new, x_new, aux_new, advance)

        def aux_skip(carry, inputs):
            x, aux, t, failed, count = carry
            return carry, (t, x, aux, jnp.asarray(False))

        def aux_body(carry, inputs):
            return jax.lax.cond(
                carry[3],
                lambda pair: aux_skip(*pair),
                lambda pair: aux_attempt(*pair),
                (carry, inputs),
            )

        carry_0 = (
            x_0,
            aux_initial,
            t_0,
            ~initial_ok,
            jnp.asarray(0, jnp.int32),
        )
        (x_final, aux_final, t_final, failed, num_accepted), rows = jax.lax.scan(
            aux_body,
            carry_0,
            (time_grid[:-1], time_grid[1:], d_w),
        )
        ts_s, xs_s, aux_s, advance_s = rows
        all_times = jnp.concatenate([t_0[None], ts_s])
        all_states = prepend(x_0, xs_s)
        all_aux = prepend(aux_initial, aux_s)
        accepted = jnp.concatenate([jnp.ones((1,), bool), advance_s])
        last_time = all_times[num_accepted]
        last_state = take(all_states, num_accepted)
        last_aux = take(all_aux, num_accepted)
        output_times = jnp.where(
            accepted,
            all_times,
            jnp.inf if save_at.fill == "inf" else last_time,
        )
        return Solution(
            ts=output_times,
            xs=fill_rows(all_states, accepted, last_state, save_at.fill),
            ok=~failed & (num_accepted == n_steps),
            num_accepted=num_accepted,
            accepted=accepted,
            aux=fill_rows(all_aux, accepted, last_aux, save_at.fill),
        )

    all_states = prepend(x_0, step_states)
    accepted = jnp.ones((n_steps + 1,), bool)
    return Solution(
        ts=time_grid,
        xs=all_states,
        ok=ok,
        num_accepted=num_accepted,
        accepted=accepted,
    )
