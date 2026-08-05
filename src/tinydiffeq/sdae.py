"""Fixed-step stochastic integration for semi-explicit index-1 SDAEs."""

import jax
import jax.numpy as jnp

from tinydiffeq._aux import (
    make_safe_evaluator,
    resolve_algebraic_aux,
    resolve_field_aux,
    split_algebraic_output,
    split_field_output,
)
from tinydiffeq._tree import (
    add_scaled,
    asarray_state,
    assert_same_structure,
    fill_rows,
    multiply,
    prepend,
    take,
    where,
)
from tinydiffeq.dae import (
    LMRootSolver,
    _canonicalize_cached_dae_field,
    _canonicalize_dae_field,
    _get_algebraic_solver,
    _make_implicit_root_solver,
    _prepare_failure_ad_reference,
)
from tinydiffeq.save_at import SaveAt
from tinydiffeq.solution import DAESolution
from tinydiffeq.solvers import INV_SQRT_3, SRA1, EulerMaruyama


def solve_semi_explicit_sdae(
    drift,
    diffusion,
    g,
    solver,
    t_0,
    t_1,
    y_0,
    z_0,
    *,
    key,
    n_steps,
    p=None,
    args=None,
    save_at=None,
    root_solver=None,
    has_aux=None,
    has_algebraic_aux=None,
    failure_ad_reference=None,
):
    """Integrate a semi-explicit index-1 Ito SDAE with diagonal noise.

    The system is ``dy = drift(y, z, t) dt + diffusion(y, z, t) dW`` with
    ``0 = g(y, z, t)``. ``solver`` is ``EulerMaruyama`` or ``SRA1``, applied
    to the reduced SDE obtained from the locally unique root ``z = Z(y, t)``:
    the differential state advances on a fixed uniform grid of ``n_steps``
    steps, and a root solve restores consistency at every node and at SRA1's
    drift stage. SRA1's strong order 1.5 requires a diffusion that depends
    only on time. A fixed ``key`` defines one common-random-numbers path;
    JVP/VJP with respect to ``y_0`` and ``p`` are pathwise, and ``z_0`` is a
    root guess with zero tangent. Aux contracts, ``failure_ad_reference``,
    and failure behavior follow ``solve_semi_explicit_dae``.
    """
    if not isinstance(n_steps, int) or isinstance(n_steps, bool):
        raise TypeError("n_steps must be a static Python int")
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1")
    if not isinstance(solver, (EulerMaruyama, SRA1)):
        raise TypeError("semi-explicit SDAEs support EulerMaruyama and SRA1")
    if save_at is None:
        save_at = SaveAt(t_1=True)
    if save_at.ts is not None:
        raise ValueError(
            "SaveAt(ts=...) is not supported for SDAEs: interpolate neither "
            "rough paths nor algebraic outputs between stochastic steps"
        )
    if root_solver is None:
        root_solver = LMRootSolver()
    if not isinstance(root_solver, LMRootSolver):
        raise TypeError("root_solver must be an LMRootSolver")

    y_0, time_dtype = asarray_state(y_0, "y_0")
    z_0, z_dtype = asarray_state(z_0, "z_0")
    t_0 = jnp.asarray(t_0, time_dtype)
    t_1 = jnp.asarray(t_1, time_dtype)
    failure_ad_reference = _prepare_failure_ad_reference(
        failure_ad_reference, y_0, z_0, t_0, p
    )
    dt = (t_1 - t_0) / n_steps
    time_grid = jnp.linspace(t_0, t_1, n_steps + 1)
    noise = solver.sample_noise(y_0, key, n_steps, dt, time_dtype)
    is_sra1 = isinstance(solver, SRA1)

    raw_drift = drift
    raw_diffusion = diffusion
    g_field = _canonicalize_dae_field(g, "g")
    has_algebraic_aux, algebraic_aux_shape = resolve_algebraic_aux(
        g_field,
        (y_0, z_0, t_0, args, p),
        has_algebraic_aux,
    )
    if has_algebraic_aux:
        drift = _canonicalize_cached_dae_field(raw_drift, "drift")
        diffusion = _canonicalize_cached_dae_field(raw_diffusion, "diffusion")
        drift_primals = (y_0, z_0, t_0, args, p, algebraic_aux_shape)
    else:
        drift = _canonicalize_dae_field(raw_drift, "drift")
        diffusion = _canonicalize_dae_field(raw_diffusion, "diffusion")
        drift_primals = (y_0, z_0, t_0, args, p)
    has_aux, aux_shape = resolve_field_aux(
        drift,
        drift_primals,
        jax.tree.structure(y_0),
        has_aux,
        name="has_aux",
    )
    algebraic_solver = _get_algebraic_solver(g, root_solver, has_algebraic_aux)
    solve_root_ad, _, algebraic_auxiliary = _make_implicit_root_solver(
        g_field,
        algebraic_solver,
        root_solver,
        z_0,
        z_dtype,
        args,
        has_algebraic_aux,
    )
    if has_algebraic_aux:
        context_evaluator = make_safe_evaluator(
            algebraic_auxiliary, algebraic_aux_shape
        )

        def evaluate_context(y, z, t, active):
            return context_evaluator((y, z, t, p), active, failure_ad_reference)

    else:
        evaluate_context = None

    def drift_output(y, z, t, p_value, context=None):
        if has_algebraic_aux:
            return drift(y, z, t, args, p_value, context)
        return drift(y, z, t, args, p_value)

    def diffusion_output(y, z, t, p_value, context=None):
        if has_algebraic_aux:
            return diffusion(y, z, t, args, p_value, context)
        return diffusion(y, z, t, args, p_value)

    if has_aux:

        def auxiliary(inputs):
            y, z, t, p_value = inputs
            if has_algebraic_aux:
                output = g_field(y, z, t, args, p_value)
                _, context = split_algebraic_output(output, True)
            else:
                context = None
            return split_field_output(drift_output(y, z, t, p_value, context), True)[1]

        aux_evaluator = make_safe_evaluator(auxiliary, aux_shape)

        def evaluate_aux(y, z, t, active):
            return aux_evaluator((y, z, t, p), active, failure_ad_reference)

    else:
        evaluate_aux = None

    def solve_root(y, t, z_guess, active):
        return solve_root_ad(y, t, z_guess, p, active, failure_ad_reference)

    def checked_value(output, name):
        value, dtype = asarray_state(output, name)
        assert_same_structure(y_0, value, name)
        if dtype != time_dtype:
            raise TypeError(f"{name} must preserve the y dtype")
        return value

    (
        z_initial,
        initial_root_ok,
        initial_root_solves,
        initial_root_steps,
    ) = solve_root(y_0, t_0, z_0, jnp.asarray(True))
    if has_algebraic_aux:
        context_initial, initial_context_ok = evaluate_context(
            y_0, z_initial, t_0, initial_root_ok
        )
    else:
        context_initial = None
        initial_context_ok = initial_root_ok
    track_aux = has_aux and save_at.steps
    if track_aux:
        aux_initial, initial_aux_ok = evaluate_aux(
            y_0, z_initial, t_0, initial_context_ok
        )
        initial_ok = initial_context_ok & initial_aux_ok
    else:
        aux_initial = None
        initial_ok = initial_context_ok

    if has_algebraic_aux:
        y_ref, z_ref, t_ref, p_ref = failure_ad_reference
        context_reference, _ = context_evaluator(
            (y_ref, z_ref, t_ref, p_ref),
            jnp.asarray(True),
            failure_ad_reference,
        )
    else:
        context_reference = None

    def attempt_step(carry, inputs):
        (
            y,
            z,
            context,
            aux,
            t,
            failed,
            num_accepted,
            num_steps,
            num_root_solves,
            num_root_steps,
        ) = carry
        t_step, t_next, noise_step = inputs
        active = ~failed
        y_ref, z_ref, t_ref, p_ref = failure_ad_reference
        y_eval = where(active, y, y_ref)
        z_eval = where(active, z, z_ref)
        t_eval = jnp.where(active, t_step, t_ref)
        p_eval = where(active, p, p_ref)
        context_eval = (
            where(active, context, context_reference) if has_algebraic_aux else None
        )
        drift_raw, _ = split_field_output(
            drift_output(y_eval, z_eval, t_eval, p_eval, context_eval), has_aux
        )
        drift_value = checked_value(drift_raw, "drift(y, z, t)")
        diffusion_value = checked_value(
            diffusion_output(y_eval, z_eval, t_eval, p_eval, context_eval),
            "diffusion(y, z, t)",
        )
        if is_sra1:
            d_w_step, d_z_step = noise_step
            chi = jax.tree.map(
                lambda w, v: 0.5 * (w + v * INV_SQRT_3), d_w_step, d_z_step
            )
            # Additive-noise contract: the diffusion may depend only on time,
            # so its endpoint evaluation reuses the node's (y, z, context).
            t_next_eval = jnp.where(active, t_next, t_ref)
            diffusion_next = checked_value(
                diffusion_output(y_eval, z_eval, t_next_eval, p_eval, context_eval),
                "diffusion(y, z, t)",
            )
            t_stage = t_step + 0.75 * dt
            y_stage = add_scaled(
                y,
                (0.75 * dt, drift_value),
                (1.5, multiply(diffusion_next, chi)),
            )
            z_stage, stage_root_ok, stage_solves, stage_steps = solve_root(
                y_stage, t_stage, z, active
            )
            if has_algebraic_aux:
                context_stage, stage_context_ok = evaluate_context(
                    y_stage, z_stage, t_stage, stage_root_ok
                )
                stage_ok = stage_root_ok & stage_context_ok
            else:
                context_stage = None
                stage_ok = stage_root_ok
            y_stage_eval = where(stage_ok, y_stage, y_ref)
            z_stage_eval = where(stage_ok, z_stage, z_ref)
            t_stage_eval = jnp.where(stage_ok, t_stage, t_ref)
            context_stage_eval = (
                where(stage_ok, context_stage, context_reference)
                if has_algebraic_aux
                else None
            )
            stage_drift_raw, _ = split_field_output(
                drift_output(
                    y_stage_eval,
                    z_stage_eval,
                    t_stage_eval,
                    p_eval,
                    context_stage_eval,
                ),
                has_aux,
            )
            stage_drift_value = checked_value(stage_drift_raw, "drift(y, z, t)")
            y_candidate = add_scaled(
                y,
                (dt / 3.0, drift_value),
                (2.0 * dt / 3.0, stage_drift_value),
                (1.0, multiply(diffusion_next, d_w_step)),
                (
                    1.0,
                    multiply(add_scaled(diffusion_value, (-1.0, diffusion_next)), chi),
                ),
            )
            z_candidate, endpoint_ok, endpoint_solves, endpoint_steps = solve_root(
                y_candidate, t_next, z_stage, stage_ok
            )
            root_ok = stage_ok & endpoint_ok
            root_solves = stage_solves + endpoint_solves
            root_steps = stage_steps + endpoint_steps
        else:
            y_candidate = add_scaled(
                y,
                (dt, drift_value),
                (1.0, multiply(diffusion_value, noise_step)),
            )
            z_candidate, root_ok, root_solves, root_steps = solve_root(
                y_candidate, t_next, z, active
            )
        if has_algebraic_aux:
            context_candidate, context_ok = evaluate_context(
                y_candidate, z_candidate, t_next, root_ok & active
            )
        else:
            context_candidate = None
            context_ok = root_ok & active
        provisional_advance = root_ok & context_ok & active
        if track_aux:

            def accepted_aux():
                y_safe = where(provisional_advance, y_candidate, y)
                z_safe = where(provisional_advance, z_candidate, z)
                t_safe = jnp.where(provisional_advance, t_next, t)
                return evaluate_aux(
                    y_safe,
                    z_safe,
                    t_safe,
                    provisional_advance,
                )

            aux_candidate, aux_ok = jax.lax.cond(
                provisional_advance,
                accepted_aux,
                lambda: (aux, jnp.asarray(True)),
            )
        else:
            aux_candidate = None
            aux_ok = jnp.asarray(True)
        advance = provisional_advance & aux_ok
        y_new = where(advance, y_candidate, y)
        z_new = where(advance, z_candidate, z)
        context_new = (
            where(advance, context_candidate, context) if has_algebraic_aux else None
        )
        t_new = jnp.where(advance, t_next, t)
        if track_aux:
            aux_new = where(advance, aux_candidate, aux)
        else:
            aux_new = None
        failed_new = (
            failed
            | ~root_ok
            | (root_ok & ~context_ok)
            | (provisional_advance & ~aux_ok)
        )
        num_new = num_accepted + advance.astype(jnp.int32)
        carry_new = (
            y_new,
            z_new,
            context_new,
            aux_new,
            t_new,
            failed_new,
            num_new,
            num_steps + jnp.asarray(1, jnp.int32),
            num_root_solves + root_solves,
            num_root_steps + root_steps,
        )
        out = (t_new, y_new, z_new, aux_new, advance) if save_at.steps else None
        return carry_new, out

    def skip_step(carry, _):
        y, z, context, aux, t, failed, num_accepted, _, _, _ = carry
        out = (t, y, z, aux, jnp.asarray(False)) if save_at.steps else None
        return carry, out

    def body(carry, inputs):
        return jax.lax.cond(
            carry[5],
            lambda pair: skip_step(*pair),
            lambda pair: attempt_step(*pair),
            (carry, inputs),
        )

    carry_0 = (
        y_0,
        z_initial,
        context_initial,
        aux_initial,
        t_0,
        ~initial_ok,
        jnp.asarray(0, jnp.int32),
        jnp.asarray(0, jnp.int32),
        initial_root_solves,
        initial_root_steps,
    )
    (
        (
            y_final,
            z_final,
            context_final,
            aux_final,
            t_final,
            failed,
            num_accepted,
            num_steps,
            num_root_solves,
            num_root_steps,
        ),
        rows,
    ) = jax.lax.scan(
        body,
        carry_0,
        (time_grid[:-1], time_grid[1:], noise),
    )
    ok = ~failed & (num_accepted == n_steps)
    if save_at.t_1:
        if has_aux:
            aux_final, aux_ok = evaluate_aux(
                y_final, z_final, t_final, jnp.asarray(True)
            )
        else:
            aux_final = None
            aux_ok = jnp.asarray(True)
        return DAESolution(
            ts=t_final,
            ys=y_final,
            zs=z_final,
            ok=ok & aux_ok,
            num_accepted=num_accepted,
            num_steps=num_steps,
            aux=aux_final,
            num_root_solves=num_root_solves,
            num_root_steps=num_root_steps,
        )

    ts_s, ys_s, zs_s, aux_s, advance_s = rows
    all_times = jnp.concatenate([t_0[None], ts_s])
    all_ys = prepend(y_0, ys_s)
    all_zs = prepend(z_initial, zs_s)
    all_aux = prepend(aux_initial, aux_s) if track_aux else None
    accepted = jnp.concatenate([jnp.ones((1,), bool), advance_s])
    last_time = all_times[num_accepted]
    last_y = take(all_ys, num_accepted)
    last_z = take(all_zs, num_accepted)
    last_aux = take(all_aux, num_accepted) if track_aux else None
    output_times = jnp.where(
        accepted,
        all_times,
        jnp.inf if save_at.fill == "inf" else last_time,
    )
    return DAESolution(
        ts=output_times,
        ys=fill_rows(all_ys, accepted, last_y, save_at.fill),
        zs=fill_rows(all_zs, accepted, last_z, save_at.fill),
        ok=ok,
        num_accepted=num_accepted,
        num_steps=num_steps,
        accepted=accepted,
        aux=(
            fill_rows(all_aux, accepted, last_aux, save_at.fill) if track_aux else None
        ),
        num_root_solves=num_root_solves,
        num_root_steps=num_root_steps,
    )
