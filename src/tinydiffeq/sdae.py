"""Fixed-step Euler--Maruyama for semi-explicit index-1 SDAEs."""

import jax
import jax.numpy as jnp

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
    _canonicalize_dae_field,
    _get_algebraic_solver,
    _make_implicit_root_solver,
    _make_safe_aux_evaluator,
    _prepare_failure_ad_reference,
    _validate_algebraic_output,
)
from tinydiffeq.save_at import SaveAt
from tinydiffeq.sde import _diagonal_brownian_increments
from tinydiffeq.solution import DAESolution
from tinydiffeq.solvers import EulerMaruyama


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
    has_aux=False,
    failure_ad_reference=None,
):
    """Integrate a semi-explicit index-1 Ito SDAE with diagonal noise.

    The system is ``dy = drift(y, z, t) dt + diffusion(y, z, t) dW`` and
    ``0 = g(y, z, t)``. Euler--Maruyama advances the differential state on a
    fixed uniform grid, then an algebraic root solve restores consistency at
    every node. This is Euler--Maruyama applied to the reduced SDE obtained
    from the locally unique root ``z = Z(y, t)``.

    With ``has_aux=True``, ``g`` returns ``(residual, aux)`` and the floating
    aux pytree is evaluated once at each consistent node. A fixed key defines
    one common-random-number path for JVP/VJP with respect to ``y_0`` and
    ``p``. ``z_0`` is only a root guess and has zero tangent by contract.
    ``failure_ad_reference=(y, z, t, p)`` may provide a domain-safe point for
    retaining successful-lane derivatives when other ``vmap`` lanes fail.
    A nonfinite aux leaf fails the solve; at the initial point no stochastic
    time-step work is attempted after that failure.
    """
    if not isinstance(n_steps, int) or isinstance(n_steps, bool):
        raise TypeError("n_steps must be a static Python int")
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1")
    if not isinstance(has_aux, bool):
        raise TypeError("has_aux must be a static Python bool")
    if not isinstance(solver, EulerMaruyama):
        raise TypeError("semi-explicit SDAEs currently support EulerMaruyama")
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
    d_w = _diagonal_brownian_increments(y_0, key, n_steps, dt, time_dtype)

    drift = _canonicalize_dae_field(drift, "drift")
    diffusion = _canonicalize_dae_field(diffusion, "diffusion")
    g_field = _canonicalize_dae_field(g, "g")
    zero_aux = _validate_algebraic_output(g_field, y_0, z_0, t_0, args, p, has_aux)
    algebraic_solver = _get_algebraic_solver(g, root_solver, has_aux)
    solve_root_ad, _, auxiliary = _make_implicit_root_solver(
        g_field,
        algebraic_solver,
        root_solver,
        z_0,
        z_dtype,
        args,
        has_aux,
    )
    evaluate_aux = _make_safe_aux_evaluator(auxiliary, zero_aux) if has_aux else None

    def solve_root(y, t, z_guess):
        return solve_root_ad(y, t, z_guess, p, failure_ad_reference)

    def checked_field(field, name, y, z, t):
        value, dtype = asarray_state(field(y, z, t, args, p), name)
        assert_same_structure(y_0, value, name)
        if dtype != time_dtype:
            raise TypeError(f"{name} must preserve the y dtype")
        return value

    z_initial, initial_root_ok = solve_root(y_0, t_0, z_0)
    if has_aux:
        aux_initial, initial_aux_ok = evaluate_aux(
            y_0,
            z_initial,
            t_0,
            p,
            initial_root_ok,
            failure_ad_reference,
        )
        initial_ok = initial_root_ok & initial_aux_ok
    else:
        aux_initial = None
        initial_ok = initial_root_ok

    def attempt_step(carry, inputs):
        y, z, aux, t, failed, num_accepted = carry
        t_step, t_next, d_w_step = inputs
        drift_value = checked_field(drift, "drift(y, z, t)", y, z, t_step)
        diffusion_value = checked_field(diffusion, "diffusion(y, z, t)", y, z, t_step)
        y_candidate = add_scaled(
            y,
            (dt, drift_value),
            (1.0, multiply(diffusion_value, d_w_step)),
        )
        z_candidate, root_ok = solve_root(y_candidate, t_next, z)
        provisional_advance = root_ok & ~failed
        if has_aux:

            def accepted_aux():
                y_safe = where(provisional_advance, y_candidate, y)
                z_safe = where(provisional_advance, z_candidate, z)
                t_safe = jnp.where(provisional_advance, t_next, t)
                return evaluate_aux(
                    y_safe,
                    z_safe,
                    t_safe,
                    p,
                    provisional_advance,
                    failure_ad_reference,
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
        t_new = jnp.where(advance, t_next, t)
        if has_aux:
            aux_new = where(advance, aux_candidate, aux)
        else:
            aux_new = None
        failed_new = failed | ~root_ok | (provisional_advance & ~aux_ok)
        num_new = num_accepted + advance.astype(jnp.int32)
        carry_new = (y_new, z_new, aux_new, t_new, failed_new, num_new)
        out = (t_new, y_new, z_new, aux_new, advance) if save_at.steps else None
        return carry_new, out

    def skip_step(carry, _):
        y, z, aux, t, failed, num_accepted = carry
        out = (t, y, z, aux, jnp.asarray(False)) if save_at.steps else None
        return (y, z, aux, t, failed, num_accepted), out

    def body(carry, inputs):
        return jax.lax.cond(
            carry[4],
            lambda pair: skip_step(*pair),
            lambda pair: attempt_step(*pair),
            (carry, inputs),
        )

    carry_0 = (
        y_0,
        z_initial,
        aux_initial,
        t_0,
        ~initial_ok,
        jnp.asarray(0, jnp.int32),
    )
    (y_final, z_final, aux_final, t_final, failed, num_accepted), rows = jax.lax.scan(
        body,
        carry_0,
        (time_grid[:-1], time_grid[1:], d_w),
    )
    ok = ~failed & (num_accepted == n_steps)
    if save_at.t_1:
        return DAESolution(
            ts=t_final,
            ys=y_final,
            zs=z_final,
            ok=ok,
            num_accepted=num_accepted,
            aux=aux_final,
        )

    ts_s, ys_s, zs_s, aux_s, advance_s = rows
    all_times = jnp.concatenate([t_0[None], ts_s])
    all_ys = prepend(y_0, ys_s)
    all_zs = prepend(z_initial, zs_s)
    all_aux = prepend(aux_initial, aux_s) if has_aux else None
    accepted = jnp.concatenate([jnp.ones((1,), bool), advance_s])
    last_time = all_times[num_accepted]
    last_y = take(all_ys, num_accepted)
    last_z = take(all_zs, num_accepted)
    last_aux = take(all_aux, num_accepted) if has_aux else None
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
        accepted=accepted,
        aux=(fill_rows(all_aux, accepted, last_aux, save_at.fill) if has_aux else None),
    )
