import functools
import inspect
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from nlls_gram import LevenbergMarquardt, LMStatus

from tinydiffeq._aux import (
    make_safe_evaluator,
    resolve_algebraic_aux,
    resolve_field_aux,
    shape_tree,
    split_algebraic_output,
    split_field_output,
)
from tinydiffeq._rodas5p import rodas5p_step, rodas_dense_endpoint_derivatives
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
from tinydiffeq.interpolation import (
    hermite_interpolate,
    hermite_interval_interpolate,
    rodas_interpolate,
)
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
    Rodas5P,
    Tsit5,
)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LMRootSolver:
    """Configuration for algebraic solves in a semi-explicit DAE.

    The implementation is :class:`nlls_gram.LevenbergMarquardt`. Its
    shape-adaptive dense default uses ``linear_solver="auto"`` (the normal
    Cholesky form for a square DAE constraint) and its implicit derivative
    uses ``ad_solver="auto"``, which resolves every square constraint to the
    general nonsymmetric direct solve.
    ``max_steps`` counts nonlinear iterations for one algebraic root and is
    independent of the integration's time-step ``max_steps``. ``atol=None``
    selects ``1e-6`` in float32 and ``1e-10`` in float64. ``gtol`` and
    ``xtol`` default to zero (disabled); root tolerances are deliberately
    independent of the outer integration tolerances.

    The remaining fields pass directly to ``LevenbergMarquardt``. Algebraic
    residuals do not expose nlls aux, Jacobian caching is disabled because
    roots change at every DAE stage, and geodesic acceleration is disabled.
    """

    max_steps: int = field(default=8, metadata=dict(static=True))
    atol: float | None = field(default=None, metadata=dict(static=True))
    gtol: float = field(default=0.0, metadata=dict(static=True))
    xtol: float = field(default=0.0, metadata=dict(static=True))
    init_damping: float = field(default=1e-3, metadata=dict(static=True))
    damping_decrease: float = field(default=0.5, metadata=dict(static=True))
    damping_increase: float = field(default=4.0, metadata=dict(static=True))
    max_damping: float | None = field(default=None, metadata=dict(static=True))
    linear_solver: str = field(default="auto", metadata=dict(static=True))
    jacobian_mode: str = field(default="auto", metadata=dict(static=True))
    iterative_tol: float = field(default=0.0, metadata=dict(static=True))
    iterative_atol: float = field(default=0.0, metadata=dict(static=True))
    iterative_maxiter: int | None = field(default=8, metadata=dict(static=True))
    dual_preconditioner: Any = field(default=None, metadata=dict(static=True))
    preconditioner_factory: Any = field(default=None, metadata=dict(static=True))
    normal_preconditioner: Any = field(default=None, metadata=dict(static=True))
    whitened_preconditioner: Any = field(default=None, metadata=dict(static=True))
    ad_solver: str = field(default="auto", metadata=dict(static=True))
    ad_solver_tol: float | None = field(default=None, metadata=dict(static=True))
    ad_solver_atol: float = field(default=0.0, metadata=dict(static=True))
    ad_solver_maxiter: int | None = field(default=None, metadata=dict(static=True))
    ad_solver_preconditioner: Any = field(default=None, metadata=dict(static=True))
    ad_solver_penalty: float | None = field(default=None, metadata=dict(static=True))
    linear_solve_dtype: Any = field(default=None, metadata=dict(static=True))
    metric_solve_dtype: Any = field(default=None, metadata=dict(static=True))
    metric: Any = field(default=None, metadata=dict(static=True))
    metric_factory: Any = field(default=None, metadata=dict(static=True))
    recycle: Any = field(default=None, metadata=dict(static=True))

    def __post_init__(self):
        if not isinstance(self.max_steps, int) or isinstance(self.max_steps, bool):
            raise ValueError("LMRootSolver.max_steps must be a positive int")
        if self.max_steps <= 0:
            raise ValueError("LMRootSolver.max_steps must be a positive int")
        if self.atol is not None and self.atol < 0:
            raise ValueError("LMRootSolver.atol must be nonnegative or None")
        if self.gtol < 0:
            raise ValueError("LMRootSolver.gtol must be nonnegative")
        if self.xtol < 0:
            raise ValueError("LMRootSolver.xtol must be nonnegative")
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


def _canonicalize_cached_dae_field(fn, name):
    """Require the full DAE signature when algebraic aux is consumed."""
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn
    arity = 0
    for parameter in signature.parameters.values():
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            arity += 1
        elif parameter.kind == inspect.Parameter.VAR_POSITIONAL:
            return fn
    if arity != 6:
        raise ValueError(
            f"{name} must take (y, z, t, args, p, algebraic_aux) when "
            "has_algebraic_aux=True"
        )
    return fn


def _asarray_residual(value):
    """Convert the algebraic residual while rejecting residual pytrees."""
    if jax.tree.structure(value) != jax.tree.structure(0):
        raise TypeError("g(y, z, t) residual must be a single array")
    try:
        value = jnp.asarray(value)
    except (TypeError, ValueError) as error:
        raise TypeError("g(y, z, t) residual must be array-like") from error
    if not jnp.issubdtype(value.dtype, jnp.floating):
        raise TypeError("g(y, z, t) residual must have a real floating dtype")
    if value.size == 0:
        raise ValueError("g(y, z, t) residual must not be empty")
    return value


def _build_algebraic_solver(g, config, has_algebraic_aux):
    g = _canonicalize_dae_field(g, "g")

    def residual(z, args, root_p):
        y, t, p = root_p
        value = g(y, z, t, args, p)
        value = value[0] if has_algebraic_aux else value
        return _asarray_residual(value)

    return LevenbergMarquardt(
        residual,
        init_damping=config.init_damping,
        damping_decrease=config.damping_decrease,
        damping_increase=config.damping_increase,
        max_damping=config.max_damping,
        linear_solver=config.linear_solver,
        jacobian_mode=config.jacobian_mode,
        iterative_tol=config.iterative_tol,
        iterative_atol=config.iterative_atol,
        iterative_maxiter=config.iterative_maxiter,
        dual_preconditioner=config.dual_preconditioner,
        preconditioner_factory=config.preconditioner_factory,
        normal_preconditioner=config.normal_preconditioner,
        whitened_preconditioner=config.whitened_preconditioner,
        ad_solver=config.ad_solver,
        ad_solver_tol=config.ad_solver_tol,
        ad_solver_atol=config.ad_solver_atol,
        ad_solver_maxiter=config.ad_solver_maxiter,
        ad_solver_preconditioner=config.ad_solver_preconditioner,
        ad_solver_penalty=config.ad_solver_penalty,
        linear_solve_dtype=config.linear_solve_dtype,
        metric_solve_dtype=config.metric_solve_dtype,
        metric=config.metric,
        metric_factory=config.metric_factory,
        geodesic_acceleration=False,
        cache_jacobian=False,
        recycle=config.recycle,
    )


@functools.cache
def _cached_algebraic_solver(g, config, has_algebraic_aux):
    # nlls-gram's compiled loop keys on solver identity. Cache by the user's
    # stable function identity so repeated eager solves do not retrace.
    return _build_algebraic_solver(g, config, has_algebraic_aux)


def _get_algebraic_solver(g, config, has_algebraic_aux=False):
    try:
        return _cached_algebraic_solver(g, config, has_algebraic_aux)
    except TypeError:
        # Unhashable callable objects are uncommon, but remain supported.
        return _build_algebraic_solver(g, config, has_algebraic_aux)


def _prepare_failure_ad_reference(reference, y, z, t, p):
    """Validate a derivative-only reference point for inactive vmap lanes."""

    def ones(tree):
        return jax.tree.map(
            lambda value: jax.lax.stop_gradient(jnp.ones_like(value)),
            tree,
        )

    if reference is None:
        return ones(y), ones(z), jnp.ones_like(t), ones(p)
    if not isinstance(reference, tuple) or len(reference) != 4:
        raise TypeError("failure_ad_reference must be a (y, z, t, p) tuple")

    def cast_like(candidate, model, name):
        if jax.tree.structure(candidate) != jax.tree.structure(model):
            raise ValueError(f"failure_ad_reference {name} has the wrong pytree")

        def cast(candidate_leaf, model_leaf):
            model_leaf = jnp.asarray(model_leaf)
            if not isinstance(candidate_leaf, jax.core.Tracer):
                concrete = np.asarray(candidate_leaf, dtype=np.dtype(model_leaf.dtype))
                if np.issubdtype(concrete.dtype, np.inexact) and not np.all(
                    np.isfinite(concrete)
                ):
                    raise ValueError(
                        f"failure_ad_reference {name} leaves must be finite"
                    )
            candidate_leaf = jnp.asarray(candidate_leaf, model_leaf.dtype)
            if candidate_leaf.shape != model_leaf.shape:
                raise ValueError(
                    f"failure_ad_reference {name} leaves must match model shapes"
                )
            return jax.lax.stop_gradient(candidate_leaf)

        return jax.tree.map(cast, candidate, model)

    y_ref, z_ref, t_ref, p_ref = reference
    y_ref = cast_like(y_ref, y, "y")
    z_ref = cast_like(z_ref, z, "z")
    if not isinstance(t_ref, jax.core.Tracer):
        concrete_t = np.asarray(t_ref, dtype=np.dtype(t.dtype))
        if not np.all(np.isfinite(concrete_t)):
            raise ValueError("failure_ad_reference t must be finite")
    t_ref = jax.lax.stop_gradient(jnp.asarray(t_ref, t.dtype))
    if t_ref.shape != t.shape:
        raise ValueError("failure_ad_reference t must match the time shape")
    p_ref = cast_like(p_ref, p, "p")
    return y_ref, z_ref, t_ref, p_ref


def _make_implicit_root_solver(
    g,
    algebraic_solver,
    root_solver,
    z_reference,
    z_dtype,
    args,
    has_algebraic_aux,
):
    """Delegate root solving and status-safe implicit AD to nlls-gram."""

    root_atol = root_solver.atol
    if root_atol is None:
        root_atol = 1e-10 if jnp.finfo(z_dtype).bits > 32 else 1e-6

    def algebraic_output(y, z, t, p):
        return g(y, z, t, args, p)

    def residual(y, z, t, p):
        value = algebraic_output(y, z, t, p)
        value = value[0] if has_algebraic_aux else value
        return _asarray_residual(value)

    def algebraic_auxiliary(inputs):
        y, z, t, p = inputs
        return split_algebraic_output(algebraic_output(y, z, t, p), True)[1]

    def solve_root(y, t, z_guess, p, failure_ad_reference):
        y_ref, z_ref, t_ref, p_ref = failure_ad_reference
        result = algebraic_solver.solve(
            z_guess,
            args,
            p=(y, t, p),
            max_steps=root_solver.max_steps,
            atol=root_atol,
            gtol=root_solver.gtol,
            xtol=root_solver.xtol,
            failure_ad_reference=(z_ref, args, (y_ref, t_ref, p_ref)),
        )
        ok = result.status == jnp.asarray(LMStatus.CONVERGED, result.status.dtype)
        value, dtype = asarray_state(result.x, "algebraic root")
        assert_same_structure(z_reference, value, "algebraic root")
        if dtype != z_dtype:
            raise TypeError("the algebraic root must preserve the z dtype")
        # A failed root is not a solution. Returning the finite warm start
        # keeps the static failure prefix usable while `ok` carries validity.
        # The guess is never a differentiable algebraic state, including on
        # failure; successful tangents come entirely from nlls implicit AD.
        return where(ok, value, jax.lax.stop_gradient(z_guess)), ok

    return solve_root, residual, algebraic_auxiliary


def _solve_rodas5p_dae(
    combined_values,
    evaluate_aux,
    t_0,
    t_1,
    y_0,
    z_initial,
    aux_initial,
    initial_ok,
    p,
    failure_ad_reference,
    dt_0,
    save_at,
    controller,
    max_steps,
    time_dtype,
    z_dtype,
    has_aux,
):
    """Integrate a semi-explicit index-1 DAE with native Rodas5P stages."""
    positive_time_floor = jnp.asarray(jnp.finfo(time_dtype).tiny, time_dtype)
    t_eps = 4.0 * jnp.finfo(time_dtype).eps * jnp.maximum(1.0, jnp.abs(t_1))
    t_slack = max_steps * t_eps
    y_flat, _ = ravel_pytree(y_0)
    z_flat, _ = ravel_pytree(z_initial)
    mass_diagonal = jnp.concatenate([jnp.ones_like(y_flat), jnp.zeros_like(z_flat)])
    controller_state_initial = controller.init((y_0, z_initial))
    track_aux = has_aux and not save_at.t_1

    def combined_field(state, t, active):
        y, z = state
        differential_value, residual_raw, field_ok = combined_values(y, z, t, active)
        residual_value, dtype = asarray_state(residual_raw, "g(y, z, t)")
        if dtype != z_dtype:
            raise TypeError("g(y, z, t) must preserve the z dtype")
        residual_flat, _ = ravel_pytree(residual_value)
        if residual_flat.size != z_flat.size:
            raise ValueError("g(y, z, t) must have the same flattened size as z")
        invalid = jnp.asarray(jnp.nan, time_dtype)
        differential_value = jax.tree.map(
            lambda value: jnp.where(field_ok, value, jnp.full_like(value, invalid)),
            differential_value,
        )
        residual_value = jax.tree.map(
            lambda value: jnp.where(field_ok, value, jnp.full_like(value, invalid)),
            residual_value,
        )
        return differential_value, residual_value

    def identity(state):
        return state

    zero_dense = (
        (zeros_like(y_0), zeros_like(z_initial)),
        (zeros_like(y_0), zeros_like(z_initial)),
        (zeros_like(y_0), zeros_like(z_initial)),
    )
    zero_aux_dot = zeros_like(aux_initial) if track_aux else None

    def auxiliary_value_and_derivative(y, z, t, state_dot, active):
        y_dot, z_dot = state_dot
        (value, ok), (value_dot, _) = jax.jvp(
            lambda y_value, z_value, t_value: evaluate_aux(
                y_value,
                z_value,
                t_value,
                p,
                active,
                failure_ad_reference,
            ),
            (y, z, t),
            (y_dot, z_dot, jnp.ones_like(t)),
        )
        return value, ok, value_dot

    def attempt_step(carry):
        (
            t,
            y,
            z,
            aux,
            dt,
            reached,
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
        state = (y, z)
        field_active = ~reached & ~failed
        candidate, err, dense, step_ok = rodas5p_step(
            lambda state_value, time_value: combined_field(
                state_value, time_value, field_active
            ),
            t,
            state,
            h,
            mass_diagonal,
            identity,
        )
        y_1, z_1 = candidate
        control_err = where(step_ok, err, full_like(err, jnp.inf))
        controller_accept, dt_next, controller_state_next = controller.adapt(
            state,
            candidate,
            control_err,
            h,
            dt,
            5,
            controller_state,
            t_1,
        )
        provisional_advance = controller_accept & step_ok & ~reached & ~failed

        if track_aux:

            def accepted_auxiliary():
                if save_at.ts is None:
                    aux_candidate, aux_ok = evaluate_aux(
                        y_1,
                        z_1,
                        t + h,
                        p,
                        provisional_advance,
                        failure_ad_reference,
                    )
                    return aux_candidate, aux_ok, zero_aux_dot, zero_aux_dot
                left_dot, right_dot = rodas_dense_endpoint_derivatives(
                    state, candidate, dense, h
                )
                _, _, aux_left_dot = auxiliary_value_and_derivative(
                    y, z, t, left_dot, provisional_advance
                )
                aux_candidate, aux_ok, aux_right_dot = auxiliary_value_and_derivative(
                    y_1, z_1, t + h, right_dot, provisional_advance
                )
                return aux_candidate, aux_ok, aux_left_dot, aux_right_dot

            aux_candidate, aux_ok, aux_left_dot, aux_right_dot = jax.lax.cond(
                provisional_advance,
                accepted_auxiliary,
                lambda: (aux, jnp.asarray(True), zero_aux_dot, zero_aux_dot),
            )
        else:
            aux_candidate = None
            aux_ok = jnp.asarray(True)
            aux_left_dot = None
            aux_right_dot = None

        advance = provisional_advance & aux_ok
        y_new = where(advance, y_1, y)
        z_new = where(advance, z_1, z)
        aux_new = where(advance, aux_candidate, aux) if track_aux else None
        t_new = jnp.where(advance, t + h, t)
        dt_new = jnp.where(reached | failed, dt, dt_next)
        controller_state_new = jax.tree.map(
            lambda old, new: jnp.where(step_ok, new, old),
            controller_state,
            controller_state_next,
        )
        if controller.uses_error_estimate:
            failed_new = failed
        else:
            failed_new = failed | ~step_ok
        failed_new = failed_new | (provisional_advance & ~aux_ok)
        reached_new = reached | (t_new >= t_1 - t_eps)
        num_new = num_accepted + advance.astype(jnp.int32)
        carry_new = (
            t_new,
            y_new,
            z_new,
            aux_new,
            dt_new,
            reached_new,
            failed_new,
            num_new,
            controller_state_new,
        )
        if save_at.t_1:
            out = None
        elif save_at.steps:
            out = (t_new, y_new, z_new, aux_new, advance)
        else:
            out = (
                t_new,
                y_new,
                z_new,
                aux_new,
                dense,
                aux_left_dot,
                aux_right_dot,
                advance,
            )
        return carry_new, out

    def skip_step(carry):
        t, y, z, aux, _, _, _, _, _ = carry
        if save_at.t_1:
            out = None
        elif save_at.steps:
            out = (t, y, z, aux, jnp.asarray(False))
        else:
            out = (
                t,
                y,
                z,
                aux,
                zero_dense,
                zero_aux_dot,
                zero_aux_dot,
                jnp.asarray(False),
            )
        return carry, out

    def body(carry, _):
        return jax.lax.cond(carry[5] | carry[6], skip_step, attempt_step, carry)

    carry_0 = (
        t_0,
        y_0,
        z_initial,
        aux_initial,
        dt_0,
        jnp.asarray(False),
        ~initial_ok,
        jnp.asarray(0, jnp.int32),
        controller_state_initial,
    )
    (
        (
            t_final,
            y_final,
            z_final,
            aux_final,
            _,
            reached,
            failed,
            num_accepted,
            _,
        ),
        rows,
    ) = jax.lax.scan(body, carry_0, None, length=max_steps)
    integration_ok = reached & ~failed

    if save_at.t_1:
        if has_aux:
            aux_final, aux_ok = evaluate_aux(
                y_final,
                z_final,
                t_final,
                p,
                jnp.asarray(True),
                failure_ad_reference,
            )
        else:
            aux_final = None
            aux_ok = jnp.asarray(True)
        return DAESolution(
            ts=t_final,
            ys=y_final,
            zs=z_final,
            ok=integration_ok & aux_ok,
            num_accepted=num_accepted,
            aux=aux_final,
        )

    if save_at.steps:
        ts_s, ys_s, zs_s, aux_s, advance_s = rows
    else:
        (
            ts_s,
            ys_s,
            zs_s,
            aux_s,
            dense_s,
            aux_left_dots_s,
            aux_right_dots_s,
            advance_s,
        ) = rows
    all_times = jnp.concatenate([t_0[None], ts_s])
    all_ys = prepend(y_0, ys_s)
    all_zs = prepend(z_initial, zs_s)
    all_aux = prepend(aux_initial, aux_s) if track_aux else None
    raw_accepted = jnp.concatenate([jnp.ones((1,), bool), advance_s])

    if save_at.steps:
        output_size = max_steps + 1
        accepted_indices = jnp.nonzero(raw_accepted, size=output_size, fill_value=0)[0]
        compact_times = all_times[accepted_indices]
        compact_ys = take(all_ys, accepted_indices)
        compact_zs = take(all_zs, accepted_indices)
        compact_aux = take(all_aux, accepted_indices) if track_aux else None
        accepted = jnp.arange(output_size) <= num_accepted
        last_time = compact_times[num_accepted]
        last_y = take(compact_ys, num_accepted)
        last_z = take(compact_zs, num_accepted)
        last_aux = take(compact_aux, num_accepted) if track_aux else None
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
            aux=(
                fill_rows(compact_aux, accepted, last_aux, save_at.fill)
                if track_aux
                else None
            ),
        )

    query_times = jnp.asarray(save_at.ts, time_dtype)
    query_ys, query_zs = rodas_interpolate(
        query_times,
        all_times,
        (all_ys, all_zs),
        dense_s,
    )
    query_aux = (
        hermite_interval_interpolate(
            query_times,
            all_times,
            all_aux,
            aux_left_dots_s,
            aux_right_dots_s,
        )
        if track_aux
        else None
    )
    return DAESolution(
        ts=query_times,
        ys=query_ys,
        zs=query_zs,
        ok=integration_ok,
        num_accepted=num_accepted,
        aux=query_aux,
    )


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
    has_aux=None,
    has_algebraic_aux=None,
    failure_ad_reference=None,
    max_steps=4096,
):
    """Integrate a semi-explicit index-1 DAE.

    The system is ``dy/dt = f(y, z, t, args, p)`` and
    ``0 = g(y, z, t, args, p)``, with a square nonsingular algebraic Jacobian
    ``dg/dz``. ``z_0`` is a root-finding guess: the initial algebraic state is
    made consistent automatically, and its derivative is determined by the
    constraint rather than by the guess.

    RK4 with fixed control, Tsit5 with fixed or adaptive control, and the
    linearly implicit Rodas5P method with fixed or adaptive control are
    supported. The outer ``max_steps`` bounds attempted time steps;
    :class:`LMRootSolver.max_steps` separately bounds algebraic solves. RK4
    and Tsit5 restore ``g=0`` at every stage. Rodas5P uses the root solver only
    for initial consistency, then advances the block mass-matrix system with
    reused linear solves; later ``z`` values satisfy the constraint to the
    integration accuracy rather than the root tolerance.

    ``args`` is fixed data by convention. All differentiated model parameters
    belong in ``p``. Initial consistency and explicit-method roots
    differentiate implicitly with respect to ``(y, t, p)``; Rodas5P then
    differentiates its discrete Jacobians and uses implicit derivatives for
    its linear solves.

    ``f`` may return ``dy`` or ``(dy, saved_aux)``. If ``g`` returns
    ``(residual, algebraic_aux)``, that second value is internal context and
    ``f`` must take the full six-argument form
    ``f(y, z, t, args, p, algebraic_aux)``. Only ``saved_aux`` is returned as
    ``sol.aux``. It is stored at accepted nodes and interpolated on requested
    deterministic grids; algebraic aux is never stored or interpolated.
    ``has_aux`` and ``has_algebraic_aux`` default to abstract auto-detection;
    explicit ``False`` selects the minimal paths without those traces.

    ``failure_ad_reference=(y, z, t, p)`` may provide a domain-safe point for
    retaining successful-lane derivatives when other ``vmap`` lanes fail.
    A nonfinite inexact algebraic-aux leaf at initialization prevents all
    time-step work. Saved aux is checked at the initial and accepted nodes in
    prefix/grid modes; endpoint mode checks it only after integration and
    retains the endpoint state with zero aux if that check fails.
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
    if not isinstance(solver, (RK4, Tsit5, Rodas5P)):
        raise TypeError("semi-explicit DAEs currently support RK4, Tsit5, and Rodas5P")
    if controller.uses_error_estimate and not solver.has_error_estimate:
        raise ValueError(
            f"{type(controller).__name__} needs an embedded error estimate, "
            f"which {type(solver).__name__} does not provide"
        )

    y_0, time_dtype = asarray_state(y_0, "y_0")
    z_0, z_dtype = asarray_state(z_0, "z_0")
    t_0 = jnp.asarray(t_0, time_dtype)
    t_1 = jnp.asarray(t_1, time_dtype)
    dt_0 = jnp.asarray(dt_0, time_dtype)
    failure_ad_reference = _prepare_failure_ad_reference(
        failure_ad_reference, y_0, z_0, t_0, p
    )
    positive_time_floor = jnp.asarray(jnp.finfo(time_dtype).tiny, time_dtype)
    t_eps = 4.0 * jnp.finfo(time_dtype).eps * jnp.maximum(1.0, jnp.abs(t_1))
    t_slack = max_steps * t_eps

    raw_f = f
    g_field = _canonicalize_dae_field(g, "g")
    has_algebraic_aux, algebraic_aux_shape = resolve_algebraic_aux(
        g_field,
        (y_0, z_0, t_0, args, p),
        has_algebraic_aux,
    )
    if has_algebraic_aux:
        f = _canonicalize_cached_dae_field(raw_f, "f")
        f_primals = (y_0, z_0, t_0, args, p, algebraic_aux_shape)
    else:
        f = _canonicalize_dae_field(raw_f, "f")
        f_primals = (y_0, z_0, t_0, args, p)
    has_aux, aux_shape = resolve_field_aux(
        f,
        f_primals,
        jax.tree.structure(y_0),
        has_aux,
        name="has_aux",
    )
    algebraic_solver = _get_algebraic_solver(g, root_solver, has_algebraic_aux)
    solve_root_ad, residual, algebraic_auxiliary = _make_implicit_root_solver(
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
        y_ref, z_ref, t_ref, p_ref = failure_ad_reference
        context_reference, _ = context_evaluator(
            (y_ref, z_ref, t_ref, p_ref),
            jnp.asarray(True),
            failure_ad_reference,
        )

        def evaluate_context(y, z, t, p_value, active):
            return context_evaluator((y, z, t, p_value), active, failure_ad_reference)

    else:
        evaluate_context = None

    def differential_output(y, z, t, p_value, algebraic_aux=None):
        if has_algebraic_aux:
            return f(y, z, t, args, p_value, algebraic_aux)
        return f(y, z, t, args, p_value)

    if has_aux:

        def auxiliary(inputs):
            y, z, t, p_value = inputs
            if has_algebraic_aux:
                output = g_field(y, z, t, args, p_value)
                _, context = split_algebraic_output(output, True)
            else:
                context = None
            output = differential_output(y, z, t, p_value, context)
            return split_field_output(output, True)[1]

        aux_evaluator = make_safe_evaluator(auxiliary, aux_shape)

        def evaluate_aux(y, z, t, p_value, active, failure_reference):
            return aux_evaluator((y, z, t, p_value), active, failure_reference)

    else:
        evaluate_aux = None

    def solve_root(y, t, z_guess):
        return solve_root_ad(y, t, z_guess, p, failure_ad_reference)

    def differential(y, z, t, active=None):
        if active is None:
            active = jnp.asarray(True)
        if has_algebraic_aux:
            context, context_ok = evaluate_context(y, z, t, p, active)
            context_eval = where(context_ok, context, context_reference)
        else:
            context = None
            context_ok = active
            context_eval = None
        y_ref, z_ref, t_ref, p_ref = failure_ad_reference
        y_eval = where(context_ok, y, y_ref)
        z_eval = where(context_ok, z, z_ref)
        t_eval = jnp.where(context_ok, t, t_ref)
        p_eval = where(context_ok, p, p_ref)
        output = differential_output(y_eval, z_eval, t_eval, p_eval, context_eval)
        value, _ = split_field_output(output, has_aux)
        value, dtype = asarray_state(value, "f(y, z, t)")
        assert_same_structure(y_0, value, "f(y, z, t)")
        if dtype != time_dtype:
            raise TypeError("f(y, z, t) must preserve the y dtype")
        return where(context_ok, value, zeros_like(y_0)), context_ok

    def algebraic_time_derivatives(y, z, t, y_dot, active):
        """IFT time derivatives for Hermite dense output at a root."""
        y_ref, z_ref, t_ref, p_ref = failure_ad_reference
        y_eval = where(active, y, y_ref)
        z_eval = where(active, z, z_ref)
        t_eval = jnp.where(active, t, t_ref)
        p_eval = where(active, p, p_ref)
        theta, unravel = ravel_pytree(z_eval)

        def residual_theta(theta_value):
            return jnp.ravel(residual(y_eval, unravel(theta_value), t_eval, p_eval))

        jacobian = jax.jacfwd(residual_theta)(theta)

        def residual_y_t(y_value, t_value):
            return jnp.ravel(residual(y_value, z_eval, t_value, p_eval))

        rhs = jax.jvp(
            residual_y_t,
            (y_eval, t_eval),
            (y_dot, jnp.ones_like(t)),
        )[1]
        # Under vmap, a scalar cond becomes selection and this solve runs on
        # inactive lanes. Replace a failed/singular system before solving so
        # neither the primal nor its transpose sees NaNs from that lane.
        identity = jnp.eye(theta.size, dtype=theta.dtype)
        jacobian_safe = jnp.where(active, jacobian, identity)
        rhs_safe = jnp.where(active, rhs, jnp.zeros_like(rhs))
        z_dot = unravel(jnp.linalg.solve(jacobian_safe, -rhs_safe))
        z_dot = where(active, z_dot, zeros_like(z_dot))
        if not has_aux:
            return z_dot, None

        aux_dot = jax.jvp(
            lambda y_value, z_value, t_value: evaluate_aux(
                y_value,
                z_value,
                t_value,
                p,
                active,
                failure_ad_reference,
            )[0],
            (y, z, t),
            (y_dot, z_dot, jnp.ones_like(t)),
        )[1]
        aux_dot = where(active, aux_dot, zeros_like(aux_dot))
        return z_dot, aux_dot

    z_initial, initial_root_ok = solve_root(y_0, t_0, z_0)
    if has_algebraic_aux:
        _, initial_context_ok = evaluate_context(
            y_0, z_initial, t_0, p, initial_root_ok
        )
    else:
        initial_context_ok = initial_root_ok
    if has_aux and not save_at.t_1:
        aux_initial, initial_aux_ok = evaluate_aux(
            y_0,
            z_initial,
            t_0,
            p,
            initial_context_ok,
            failure_ad_reference,
        )
        initial_ok = initial_context_ok & initial_aux_ok
    else:
        aux_initial = None
        initial_ok = initial_context_ok
    track_aux = has_aux and not save_at.t_1

    if isinstance(solver, Rodas5P):
        if has_algebraic_aux:

            def combined_output(inputs):
                y, z, t, p_value = inputs
                algebraic_output = g_field(y, z, t, args, p_value)
                residual_value, context = split_algebraic_output(algebraic_output, True)
                residual_value = _asarray_residual(residual_value)
                field_output = differential_output(y, z, t, p_value, context)
                differential_value, _ = split_field_output(field_output, has_aux)
                differential_value, dtype = asarray_state(
                    differential_value, "f(y, z, t)"
                )
                assert_same_structure(y_0, differential_value, "f(y, z, t)")
                if dtype != time_dtype:
                    raise TypeError("f(y, z, t) must preserve the y dtype")
                return differential_value, residual_value, context

            combined_shape = shape_tree(
                jax.eval_shape(combined_output, (y_0, z_0, t_0, p))
            )
            combined_evaluator = make_safe_evaluator(combined_output, combined_shape)

            def combined_values(y, z, t, active):
                (differential_value, residual_value, _), ok = combined_evaluator(
                    (y, z, t, p), active, failure_ad_reference
                )
                return differential_value, residual_value, ok

        else:

            def combined_values(y, z, t, active):
                y_ref, z_ref, t_ref, p_ref = failure_ad_reference
                y_eval = where(active, y, y_ref)
                z_eval = where(active, z, z_ref)
                t_eval = jnp.where(active, t, t_ref)
                p_eval = where(active, p, p_ref)
                residual_value = _asarray_residual(
                    g_field(y_eval, z_eval, t_eval, args, p_eval)
                )
                field_output = differential_output(y_eval, z_eval, t_eval, p_eval)
                differential_value, _ = split_field_output(field_output, has_aux)
                differential_value, dtype = asarray_state(
                    differential_value, "f(y, z, t)"
                )
                assert_same_structure(y_0, differential_value, "f(y, z, t)")
                if dtype != time_dtype:
                    raise TypeError("f(y, z, t) must preserve the y dtype")
                return differential_value, residual_value, active

        return _solve_rodas5p_dae(
            combined_values,
            evaluate_aux,
            t_0,
            t_1,
            y_0,
            z_initial,
            aux_initial,
            initial_ok,
            p,
            failure_ad_reference,
            dt_0,
            save_at,
            controller,
            max_steps,
            time_dtype,
            z_dtype,
            has_aux,
        )

    need_f = solver.fsal or (save_at.ts is not None)
    f_initial = jax.lax.cond(
        initial_ok,
        lambda: differential(y_0, z_initial, t_0)[0],
        lambda: zeros_like(y_0),
    )
    if save_at.ts is not None:
        z_dot_initial, aux_dot_initial = algebraic_time_derivatives(
            y_0, z_initial, t_0, f_initial, initial_ok
        )
    else:
        z_dot_initial, aux_dot_initial = None, None
    controller_state_initial = controller.init(y_0)

    def stage(y_stage, t_stage, z_guess, active, need_derivative=True):
        def evaluate():
            z_stage, root_ok = solve_root(y_stage, t_stage, z_guess)
            if need_derivative:
                k_stage, field_ok = jax.lax.cond(
                    root_ok,
                    lambda: differential(y_stage, z_stage, t_stage, root_ok),
                    lambda: (zeros_like(y_stage), jnp.asarray(False)),
                )
            else:
                k_stage = zeros_like(y_stage)
                if has_algebraic_aux:
                    _, field_ok = evaluate_context(
                        y_stage, z_stage, t_stage, p, root_ok
                    )
                else:
                    field_ok = root_ok
            return z_stage, k_stage, root_ok & field_ok

        def skip():
            return z_guess, zeros_like(y_stage), jnp.asarray(False)

        return jax.lax.cond(active, evaluate, skip)

    def rk4_step(t, y, z, h, f_cur):
        k_1 = differential(y, z, t)[0] if f_cur is None else f_cur
        z_2, k_2, ok_2 = stage(add_scaled(y, (0.5 * h, k_1)), t + 0.5 * h, z, True)
        z_3, k_3, ok_3 = stage(add_scaled(y, (0.5 * h, k_2)), t + 0.5 * h, z_2, ok_2)
        z_4, k_4, ok_4 = stage(add_scaled(y, (h, k_3)), t + h, z_3, ok_2 & ok_3)
        y_1 = add_scaled(y, (h / 6.0, weighted_sum((k_1, k_2, k_3, k_4), (1, 2, 2, 1))))
        stage_ok = ok_2 & ok_3 & ok_4
        z_1, f_1, endpoint_ok = stage(y_1, t + h, z_4, stage_ok, need_derivative=need_f)
        return y_1, z_1, f_1, None, stage_ok & endpoint_ok

    def tsit5_step(t, y, z, h, f_cur):
        k_1 = differential(y, z, t)[0] if f_cur is None else f_cur
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
        (
            t,
            y,
            z,
            aux,
            dt,
            f_cur,
            z_dot,
            aux_dot,
            reached,
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
        provisional_advance = accept & ~reached & ~failed
        if track_aux:

            def accepted_aux():
                y_safe = where(provisional_advance, y_1, y)
                z_safe = where(provisional_advance, z_1, z)
                t_safe = jnp.where(provisional_advance, t + h, t)
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
        y_new = where(advance, y_1, y)
        z_new = where(advance, z_1, z)
        t_new = jnp.where(advance, t + h, t)
        f_new = where(advance, f_1, f_cur) if need_f else f_cur
        if track_aux:
            aux_new = where(advance, aux_candidate, aux)
        else:
            aux_new = None
        if save_at.ts is not None:

            def accepted_derivatives():
                y_safe = where(advance, y_1, y)
                z_safe = where(advance, z_1, z)
                t_safe = jnp.where(advance, t + h, t)
                f_safe = where(advance, f_1, f_cur)
                return algebraic_time_derivatives(
                    y_safe, z_safe, t_safe, f_safe, advance
                )

            z_dot_new, aux_dot_new = jax.lax.cond(
                advance,
                accepted_derivatives,
                lambda: (z_dot, aux_dot),
            )
        else:
            z_dot_new, aux_dot_new = None, None
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
        failed_new = failed_new | (provisional_advance & ~aux_ok)
        reached_new = reached | (t_new >= t_1 - t_eps)
        num_new = num_accepted + advance.astype(jnp.int32)
        carry_new = (
            t_new,
            y_new,
            z_new,
            aux_new,
            dt_new,
            f_new,
            z_dot_new,
            aux_dot_new,
            reached_new,
            failed_new,
            num_new,
            controller_state_next,
        )
        if save_at.t_1:
            out = None
        elif save_at.steps:
            out = (t_new, y_new, z_new, aux_new, advance)
        else:
            out = (
                t_new,
                y_new,
                z_new,
                aux_new,
                f_new,
                z_dot_new,
                aux_dot_new,
                advance,
            )
        return carry_new, out

    def skip_step(carry):
        t, y, z, aux, _, f_cur, z_dot, aux_dot, _, _, _, _ = carry
        if save_at.t_1:
            out = None
        elif save_at.steps:
            out = (t, y, z, aux, jnp.asarray(False))
        else:
            out = (
                t,
                y,
                z,
                aux,
                f_cur,
                z_dot,
                aux_dot,
                jnp.asarray(False),
            )
        return carry, out

    def body(carry, _):
        terminated = carry[8] | carry[9]
        return jax.lax.cond(terminated, skip_step, attempt_step, carry)

    carry_0 = (
        t_0,
        y_0,
        z_initial,
        aux_initial,
        dt_0,
        f_initial,
        z_dot_initial,
        aux_dot_initial,
        jnp.asarray(False),
        ~initial_ok,
        jnp.asarray(0, jnp.int32),
        controller_state_initial,
    )
    (
        (
            t_final,
            y_final,
            z_final,
            aux_final,
            _,
            _,
            _,
            _,
            reached,
            failed,
            num_accepted,
            _,
        ),
        rows,
    ) = jax.lax.scan(body, carry_0, None, length=max_steps)
    integration_ok = reached & ~failed

    if save_at.t_1:
        if has_aux:
            aux_final, aux_ok = evaluate_aux(
                y_final,
                z_final,
                t_final,
                p,
                jnp.asarray(True),
                failure_ad_reference,
            )
        else:
            aux_final = None
            aux_ok = jnp.asarray(True)
        return DAESolution(
            ts=t_final,
            ys=y_final,
            zs=z_final,
            ok=integration_ok & aux_ok,
            num_accepted=num_accepted,
            aux=aux_final,
        )

    if save_at.steps:
        ts_s, ys_s, zs_s, aux_s, adv_s = rows
    else:
        ts_s, ys_s, zs_s, aux_s, fs_s, z_dots_s, aux_dots_s, adv_s = rows
    all_times = jnp.concatenate([t_0[None], ts_s])
    all_ys = prepend(y_0, ys_s)
    all_zs = prepend(z_initial, zs_s)
    all_aux = prepend(aux_initial, aux_s) if track_aux else None
    raw_accepted = jnp.concatenate([jnp.ones((1,), bool), adv_s])

    if save_at.steps:
        output_size = max_steps + 1
        accepted_indices = jnp.nonzero(raw_accepted, size=output_size, fill_value=0)[0]
        compact_times = all_times[accepted_indices]
        compact_ys = take(all_ys, accepted_indices)
        compact_zs = take(all_zs, accepted_indices)
        compact_aux = take(all_aux, accepted_indices) if track_aux else None
        accepted = jnp.arange(output_size) <= num_accepted
        last_time = compact_times[num_accepted]
        last_y = take(compact_ys, num_accepted)
        last_z = take(compact_zs, num_accepted)
        last_aux = take(compact_aux, num_accepted) if track_aux else None

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
            aux=(
                fill_rows(compact_aux, accepted, last_aux, save_at.fill)
                if track_aux
                else None
            ),
        )

    fs_all = prepend(f_initial, fs_s)
    z_dots_all = prepend(z_dot_initial, z_dots_s)
    aux_dots_all = prepend(aux_dot_initial, aux_dots_s) if track_aux else None
    query_times = jnp.asarray(save_at.ts, time_dtype)
    values = (all_ys, all_zs, all_aux) if track_aux else (all_ys, all_zs)
    derivatives = (
        (fs_all, z_dots_all, aux_dots_all) if track_aux else (fs_all, z_dots_all)
    )
    interpolated = hermite_interpolate(query_times, all_times, values, derivatives)
    if track_aux:
        query_ys, query_zs, query_aux = interpolated
    else:
        query_ys, query_zs = interpolated
        query_aux = None
    return DAESolution(
        ts=query_times,
        ys=query_ys,
        zs=query_zs,
        ok=integration_ok,
        num_accepted=num_accepted,
        aux=query_aux,
    )
