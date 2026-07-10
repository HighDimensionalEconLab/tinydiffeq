import functools
import inspect
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from nlls_gram import LMStatus, SquareLevenbergMarquardt

from tinydiffeq._tree import (
    add_scaled,
    asarray_state,
    assert_same_structure,
    fill_rows,
    full_like,
    prepend,
    take,
    weighted_sum,
    where,
    zeros_like,
)
from tinydiffeq.controllers import ConstantStepSize
from tinydiffeq.interpolation import hermite_interpolate
from tinydiffeq.save_at import SaveAt
from tinydiffeq.solution import DAESolution
from tinydiffeq.solvers import (
    A_21,
    A_31,
    A_32,
    A_41,
    A_42,
    A_43,
    A_51,
    A_52,
    A_53,
    A_54,
    A_61,
    A_62,
    A_63,
    A_64,
    A_65,
    B_1,
    B_2,
    B_3,
    B_4,
    B_5,
    B_6,
    C_2,
    C_3,
    C_4,
    C_5,
    C_6,
    E_1,
    E_2,
    E_3,
    E_4,
    E_5,
    E_6,
    E_7,
    RK4,
    Tsit5,
)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LMRootSolver:
    """Configuration for algebraic solves in a semi-explicit DAE.

    The implementation is :class:`nlls_gram.SquareLevenbergMarquardt`.
    ``max_steps`` counts nonlinear iterations for one algebraic root and is
    independent of the integration's time-step ``max_steps``. ``atol=None``
    selects nlls-gram's precision-aware default: ``1e-6`` in float32 and
    ``1e-10`` in float64.
    """

    max_steps: int = field(default=8, metadata=dict(static=True))
    atol: float | None = field(default=None, metadata=dict(static=True))
    init_damping: float = field(default=1e-3, metadata=dict(static=True))
    damping_decrease: float = field(default=0.5, metadata=dict(static=True))
    damping_increase: float = field(default=4.0, metadata=dict(static=True))

    def __post_init__(self):
        if not isinstance(self.max_steps, int) or isinstance(self.max_steps, bool):
            raise ValueError("LMRootSolver.max_steps must be a positive int")
        if self.max_steps <= 0:
            raise ValueError("LMRootSolver.max_steps must be a positive int")
        if self.atol is not None and self.atol < 0:
            raise ValueError("LMRootSolver.atol must be nonnegative or None")
        if self.init_damping <= 0:
            raise ValueError("LMRootSolver.init_damping must be positive")
        if self.damping_decrease <= 0:
            raise ValueError("LMRootSolver.damping_decrease must be positive")
        if self.damping_increase <= 0:
            raise ValueError("LMRootSolver.damping_increase must be positive")


def _canonicalize_dae_field(fn, name):
    # DAE fields take (y, z), (y, z, t), (y, z, t, args), or
    # (y, z, t, args, p), always in that order.
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        arity = 5
    else:
        arity = 0
        for parameter in signature.parameters.values():
            if parameter.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                arity += 1
            elif parameter.kind == inspect.Parameter.VAR_POSITIONAL:
                arity = 5
                break
        if arity < 2 or arity > 5:
            raise ValueError(
                f"{name} must take 2 to 5 positional arguments: "
                "(y, z), (y, z, t), (y, z, t, args), or "
                "(y, z, t, args, p)"
            )
    if arity == 2:
        return lambda y, z, t, args, p: fn(y, z)
    if arity == 3:
        return lambda y, z, t, args, p: fn(y, z, t)
    if arity == 4:
        return lambda y, z, t, args, p: fn(y, z, t, args)
    return fn


def _build_algebraic_solver(g, config):
    g = _canonicalize_dae_field(g, "g")

    def residual(z, args, root_p):
        y, t, p = root_p
        return g(y, z, t, args, p)

    return SquareLevenbergMarquardt(
        residual,
        init_damping=config.init_damping,
        damping_decrease=config.damping_decrease,
        damping_increase=config.damping_increase,
    )


@functools.cache
def _cached_algebraic_solver(g, config):
    # nlls-gram's compiled loop keys on solver identity. Cache by the user's
    # stable function identity so repeated eager solves do not retrace.
    return _build_algebraic_solver(g, config)


def _get_algebraic_solver(g, config):
    try:
        return _cached_algebraic_solver(g, config)
    except TypeError:
        # Unhashable callable objects are uncommon, but remain supported.
        return _build_algebraic_solver(g, config)


def solve_semi_explicit_dae(
    f,
    g,
    solver,
    t_0,
    t_1,
    y_0,
    z_0,
    *,
    p=None,
    args=None,
    dt_0=None,
    save_at=None,
    controller=None,
    root_solver=None,
    max_steps=4096,
):
    """Integrate a nonstiff semi-explicit index-1 DAE.

    The system is ``dy/dt = f(y, z, t, args, p)`` and
    ``0 = g(y, z, t, args, p)``, with a square nonsingular algebraic Jacobian
    ``dg/dz``. ``z_0`` is a root-finding guess: the initial algebraic state is
    made consistent automatically, and its derivative is determined by the
    constraint rather than by the guess.

    RK4 with a fixed controller and Tsit5 with either fixed or adaptive
    control are supported. The outer ``max_steps`` is the bounded number of
    attempted time steps; :class:`LMRootSolver.max_steps` separately bounds
    each algebraic solve. Adaptive stage-root failures reject the attempted
    time step and shrink it. A fixed-step failure terminates with ``ok=False``.

    ``args`` is fixed data by convention. All differentiated model parameters
    belong in ``p``; internally each root differentiates implicitly with
    respect to ``(y, t, p)`` using nlls-gram's custom JVP.
    """
    if dt_0 is None:
        raise ValueError("dt_0 is required (tinydiffeq has no initial-step heuristic)")
    if save_at is None:
        save_at = SaveAt(t_1=True)
    if controller is None:
        controller = ConstantStepSize()
    if root_solver is None:
        root_solver = LMRootSolver()
    if not isinstance(root_solver, LMRootSolver):
        raise TypeError("root_solver must be an LMRootSolver")
    if not isinstance(solver, (RK4, Tsit5)):
        raise TypeError("semi-explicit DAEs currently support RK4 and Tsit5")
    if controller.uses_error_estimate and not solver.has_error_estimate:
        raise ValueError(
            f"{type(controller).__name__} needs an embedded error estimate, "
            f"which {type(solver).__name__} does not provide"
        )

    f = _canonicalize_dae_field(f, "f")
    algebraic_solver = _get_algebraic_solver(g, root_solver)

    y_0, time_dtype = asarray_state(y_0, "y_0")
    z_0, z_dtype = asarray_state(z_0, "z_0")
    t_0 = jnp.asarray(t_0, time_dtype)
    t_1 = jnp.asarray(t_1, time_dtype)
    dt_0 = jnp.asarray(dt_0, time_dtype)
    positive_time_floor = jnp.asarray(jnp.finfo(time_dtype).tiny, time_dtype)
    t_eps = 4.0 * jnp.finfo(time_dtype).eps * jnp.maximum(1.0, jnp.abs(t_1))
    t_slack = max_steps * t_eps

    def solve_root(y, t, z_guess):
        result = algebraic_solver.solve(
            z_guess,
            args,
            p=(y, t, p),
            max_steps=root_solver.max_steps,
            atol=root_solver.atol,
        )
        ok = result.status == jnp.asarray(LMStatus.CONVERGED, result.status.dtype)
        value, dtype = asarray_state(result.x, "algebraic root")
        assert_same_structure(z_0, value, "algebraic root")
        if dtype != z_dtype:
            raise TypeError("the algebraic root must preserve the z dtype")
        return value, ok

    def differential(y, z, t):
        value, dtype = asarray_state(f(y, z, t, args, p), "f(y, z, t)")
        assert_same_structure(y_0, value, "f(y, z, t)")
        if dtype != time_dtype:
            raise TypeError("f(y, z, t) must preserve the y dtype")
        return value

    z_initial, initial_root_ok = solve_root(y_0, t_0, z_0)
    need_f = solver.fsal or (save_at.ts is not None)
    f_initial = jax.lax.cond(
        initial_root_ok,
        lambda: differential(y_0, z_initial, t_0),
        lambda: zeros_like(y_0),
    )
    controller_state_initial = controller.init(y_0)

    def stage(y_stage, t_stage, z_guess, active, need_derivative=True):
        def evaluate():
            z_stage, root_ok = solve_root(y_stage, t_stage, z_guess)
            if need_derivative:
                k_stage = jax.lax.cond(
                    root_ok,
                    lambda: differential(y_stage, z_stage, t_stage),
                    lambda: zeros_like(y_stage),
                )
            else:
                k_stage = zeros_like(y_stage)
            return z_stage, k_stage, root_ok

        def skip():
            return z_guess, zeros_like(y_stage), jnp.asarray(False)

        return jax.lax.cond(active, evaluate, skip)

    def rk4_step(t, y, z, h, f_cur):
        k_1 = differential(y, z, t) if f_cur is None else f_cur
        z_2, k_2, ok_2 = stage(add_scaled(y, (0.5 * h, k_1)), t + 0.5 * h, z, True)
        z_3, k_3, ok_3 = stage(add_scaled(y, (0.5 * h, k_2)), t + 0.5 * h, z_2, ok_2)
        z_4, k_4, ok_4 = stage(add_scaled(y, (h, k_3)), t + h, z_3, ok_2 & ok_3)
        y_1 = add_scaled(y, (h / 6.0, weighted_sum((k_1, k_2, k_3, k_4), (1, 2, 2, 1))))
        stage_ok = ok_2 & ok_3 & ok_4
        z_1, f_1, endpoint_ok = stage(y_1, t + h, z_4, stage_ok, need_derivative=need_f)
        return y_1, z_1, f_1, None, stage_ok & endpoint_ok

    def tsit5_step(t, y, z, h, f_cur):
        k_1 = differential(y, z, t) if f_cur is None else f_cur
        ks = [k_1]
        z_stage = z
        stages_ok = jnp.asarray(True)
        rows = (
            ((A_21,), C_2),
            ((A_31, A_32), C_3),
            ((A_41, A_42, A_43), C_4),
            ((A_51, A_52, A_53, A_54), C_5),
            ((A_61, A_62, A_63, A_64, A_65), C_6),
        )
        for coefficients, stage_time in rows:
            y_stage = add_scaled(y, (h, weighted_sum(ks, coefficients)))
            z_stage, k_stage, root_ok = stage(
                y_stage, t + stage_time * h, z_stage, stages_ok
            )
            stages_ok = stages_ok & root_ok
            ks.append(k_stage)
        y_1 = add_scaled(y, (h, weighted_sum(ks, (B_1, B_2, B_3, B_4, B_5, B_6))))
        z_1, k_7, endpoint_ok = stage(y_1, t + h, z_stage, stages_ok)
        ks.append(k_7)
        root_ok = stages_ok & endpoint_ok
        err = jax.tree.map(
            lambda value: h * value,
            weighted_sum(ks, (E_1, E_2, E_3, E_4, E_5, E_6, E_7)),
        )
        return y_1, z_1, k_7, err, root_ok

    def attempt_step(carry):
        t, y, z, dt, f_cur, reached, failed, num_accepted, controller_state = carry
        remaining = t_1 - t
        h = jnp.where(
            remaining <= dt + t_slack,
            jnp.maximum(remaining, positive_time_floor),
            dt,
        )
        if isinstance(solver, RK4):
            y_1, z_1, f_1, err, root_ok = rk4_step(
                t, y, z, h, f_cur if need_f else None
            )
        else:
            y_1, z_1, f_1, err, root_ok = tsit5_step(
                t, y, z, h, f_cur if need_f else None
            )

        if controller.uses_error_estimate:
            control_err = where(root_ok, err, full_like(err, jnp.inf))
        else:
            control_err = err
        controller_accept, dt_next, controller_state_next = controller.adapt(
            y, y_1, control_err, h, dt, solver.order, controller_state, t_1
        )
        accept = controller_accept & root_ok
        advance = accept & ~reached & ~failed
        y_new = where(advance, y_1, y)
        z_new = where(advance, z_1, z)
        t_new = jnp.where(advance, t + h, t)
        f_new = where(advance, f_1, f_cur) if need_f else f_cur
        dt_new = jnp.where(reached | failed, dt, dt_next)
        controller_state_next = jax.tree.map(
            lambda old, new: jnp.where(root_ok, new, old),
            controller_state,
            controller_state_next,
        )
        if controller.uses_error_estimate:
            failed_new = failed
        else:
            failed_new = failed | ~root_ok
        reached_new = reached | (t_new >= t_1 - t_eps)
        num_new = num_accepted + advance.astype(jnp.int32)
        carry_new = (
            t_new,
            y_new,
            z_new,
            dt_new,
            f_new,
            reached_new,
            failed_new,
            num_new,
            controller_state_next,
        )
        if save_at.t_1:
            out = None
        elif save_at.steps:
            out = (t_new, y_new, z_new, advance)
        else:
            out = (t_new, y_new, z_new, f_new, advance)
        return carry_new, out

    def skip_step(carry):
        t, y, z, _, f_cur, _, _, _, _ = carry
        if save_at.t_1:
            out = None
        elif save_at.steps:
            out = (t, y, z, jnp.asarray(False))
        else:
            out = (t, y, z, f_cur, jnp.asarray(False))
        return carry, out

    def body(carry, _):
        terminated = carry[5] | carry[6]
        return jax.lax.cond(terminated, skip_step, attempt_step, carry)

    carry_0 = (
        t_0,
        y_0,
        z_initial,
        dt_0,
        f_initial,
        jnp.asarray(False),
        ~initial_root_ok,
        jnp.asarray(0, jnp.int32),
        controller_state_initial,
    )
    (t_final, y_final, z_final, _, _, reached, failed, num_accepted, _), rows = (
        jax.lax.scan(body, carry_0, None, length=max_steps)
    )
    integration_ok = reached & ~failed

    if save_at.t_1:
        return DAESolution(
            ts=t_final,
            ys=y_final,
            zs=z_final,
            ok=integration_ok,
            num_accepted=num_accepted,
        )

    if save_at.steps:
        ts_s, ys_s, zs_s, adv_s = rows
    else:
        ts_s, ys_s, zs_s, fs_s, adv_s = rows
    all_times = jnp.concatenate([t_0[None], ts_s])
    all_ys = prepend(y_0, ys_s)
    all_zs = prepend(z_initial, zs_s)
    raw_accepted = jnp.concatenate([jnp.ones((1,), bool), adv_s])

    if save_at.steps:
        output_size = max_steps + 1
        accepted_indices = jnp.nonzero(raw_accepted, size=output_size, fill_value=0)[0]
        compact_times = all_times[accepted_indices]
        compact_ys = take(all_ys, accepted_indices)
        compact_zs = take(all_zs, accepted_indices)
        accepted = jnp.arange(output_size) <= num_accepted
        last_time = compact_times[num_accepted]
        last_y = take(compact_ys, num_accepted)
        last_z = take(compact_zs, num_accepted)

        output_times = jnp.where(
            accepted,
            compact_times,
            jnp.inf if save_at.fill == "inf" else last_time,
        )
        return DAESolution(
            ts=output_times,
            ys=fill_rows(compact_ys, accepted, last_y, save_at.fill),
            zs=fill_rows(compact_zs, accepted, last_z, save_at.fill),
            ok=integration_ok,
            num_accepted=num_accepted,
            accepted=accepted,
        )

    fs_all = prepend(f_initial, fs_s)
    query_times = jnp.asarray(save_at.ts, time_dtype)
    query_ys = hermite_interpolate(query_times, all_times, all_ys, fs_all)
    guess_indices = jnp.clip(
        jnp.searchsorted(all_times, query_times, side="right") - 1,
        0,
        all_times.shape[0] - 1,
    )
    query_guesses = take(all_zs, guess_indices)
    query_zs, query_root_ok = jax.vmap(solve_root)(query_ys, query_times, query_guesses)
    return DAESolution(
        ts=query_times,
        ys=query_ys,
        zs=query_zs,
        ok=integration_ok & jnp.all(query_root_ok),
        num_accepted=num_accepted,
    )
