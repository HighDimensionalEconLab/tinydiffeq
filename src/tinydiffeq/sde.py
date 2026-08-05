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
from tinydiffeq._unvmap import unvmap_all
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
    key=None,
    n_steps,
    noise=None,
    p=None,
    args=None,
    save_at=None,
    project=None,
    has_aux=None,
    failure_ad_reference=None,
    unroll=1,
):
    """Integrate the Ito SDE ``dx = drift dt + diffusion d_w`` with diagonal
    noise on a fixed grid of ``n_steps`` uniform steps from ``t_0`` to
    ``t_1 > t_0``.

    ``drift`` and ``diffusion`` follow the same signature convention as
    ``solve_ode``. ``solver`` is ``EulerMaruyama``, ``Milstein``, or ``SRA1``,
    each declaring its per-step noise through
    ``solver.sample_noise(x_0, key, n_steps, dt, dtype)``. Exactly one of
    ``key`` and ``noise`` must be provided: a fixed ``key`` presamples a
    fixed, reproducible noise process, differentiable with respect to ``x_0``
    and ``p``; an explicit ``noise`` pytree (validated against the solver's
    spec) is additionally differentiable as data. ``SaveAt(ts=...)`` raises —
    interpolation is wrong for rough paths. ``drift`` may return
    ``(value, aux)``; ``diffusion`` is value-only. ``unroll`` (a static int)
    unrolls that many steps per iteration of the underlying ``lax.scan`` —
    identical values, fewer/larger GPU dispatches, more compile time.
    """
    if not isinstance(n_steps, int):
        raise TypeError("n_steps must be a static Python int")
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1")
    if not isinstance(unroll, int) or isinstance(unroll, bool) or unroll < 1:
        raise ValueError("unroll must be a static int of at least 1")
    if (key is None) == (noise is None):
        raise ValueError("solve_sde requires exactly one of key or noise")
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
    if noise is None:
        noise = solver.sample_noise(x_0, key, n_steps, dt, time_dtype)
    else:
        # sample_noise only reads shapes from x_0, so the abstract trace is
        # the solver's authoritative noise spec.
        reference = jax.eval_shape(
            lambda noise_key: solver.sample_noise(
                x_0, noise_key, n_steps, dt, time_dtype
            ),
            jax.random.key(0),
        )
        noise = jax.tree.map(jnp.asarray, noise)
        if jax.tree.structure(noise) != jax.tree.structure(reference):
            raise ValueError(
                "noise must match the pytree structure of "
                "solver.sample_noise(x_0, key, n_steps, dt, dtype)"
            )
        for leaf, ref in zip(
            jax.tree.leaves(noise), jax.tree.leaves(reference), strict=True
        ):
            if leaf.shape != ref.shape:
                raise ValueError(
                    f"noise leaf shape {leaf.shape} does not match the "
                    f"solver's expected {ref.shape}"
                )
            if leaf.dtype != ref.dtype:
                raise TypeError(
                    f"noise leaf dtype {leaf.dtype} must match the state "
                    f"dtype {ref.dtype}"
                )
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
        t, noise_step = inputs
        x_1 = solver.step(g_drift, g_diffusion, t, x, dt, noise_step, project_state)
        return x_1, x_1 if save_at.steps else None

    if save_at.t_1 or not has_aux:
        x_final, step_states = jax.lax.scan(
            body, x_0, (time_grid[:-1], noise), unroll=unroll
        )
        num_accepted = jnp.asarray(n_steps, jnp.int32)
        num_steps = num_accepted
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
            num_steps=num_steps,
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
            x, aux, t, failed, count, num_steps = carry
            t_step, t_next, noise_step = inputs
            x_candidate = solver.step(
                g_drift,
                g_diffusion,
                t_step,
                x,
                dt,
                noise_step,
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
            num_steps_new = num_steps + jnp.asarray(1, jnp.int32)
            return (
                x_new,
                aux_new,
                t_new,
                failed_new,
                count_new,
                num_steps_new,
            ), (t_new, x_new, aux_new, advance)

        def aux_skip(carry, inputs):
            x, aux, t, failed, count, num_steps = carry
            return carry, (t, x, aux, jnp.asarray(False))

        def aux_body(carry, inputs):
            # Scalar-predicate outer cond under vmap (see _unvmap): once every
            # lane has failed, the frozen tail skips for real.
            def live_lanes(pair):
                return jax.lax.cond(
                    pair[0][3],
                    lambda pair: aux_skip(*pair),
                    lambda pair: aux_attempt(*pair),
                    pair,
                )

            return jax.lax.cond(
                unvmap_all(carry[3]),
                lambda pair: aux_skip(*pair),
                live_lanes,
                (carry, inputs),
            )

        carry_0 = (
            x_0,
            aux_initial,
            t_0,
            ~initial_ok,
            jnp.asarray(0, jnp.int32),
            jnp.asarray(0, jnp.int32),
        )
        (
            (
                x_final,
                aux_final,
                t_final,
                failed,
                num_accepted,
                num_steps,
            ),
            rows,
        ) = jax.lax.scan(
            aux_body, carry_0, (time_grid[:-1], time_grid[1:], noise), unroll=unroll
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
            num_steps=num_steps,
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
        num_steps=num_steps,
        accepted=accepted,
    )
