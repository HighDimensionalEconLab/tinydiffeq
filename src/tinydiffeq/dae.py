import functools
import inspect
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from nlls_gram import LMStatus, SquareLevenbergMarquardt

from tinydiffeq._tree import (
    add_scaled,
    asarray_aux,
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


def _build_algebraic_solver(g, config, has_aux):
    g = _canonicalize_dae_field(g, "g")

    def residual(z, args, root_p):
        y, t, p = root_p
        value = g(y, z, t, args, p)
        return value[0] if has_aux else value

    return SquareLevenbergMarquardt(
        residual,
        init_damping=config.init_damping,
        damping_decrease=config.damping_decrease,
        damping_increase=config.damping_increase,
    )


@functools.cache
def _cached_algebraic_solver(g, config, has_aux):
    # nlls-gram's compiled loop keys on solver identity. Cache by the user's
    # stable function identity so repeated eager solves do not retrace.
    return _build_algebraic_solver(g, config, has_aux)


def _get_algebraic_solver(g, config, has_aux=False):
    try:
        return _cached_algebraic_solver(g, config, has_aux)
    except TypeError:
        # Unhashable callable objects are uncommon, but remain supported.
        return _build_algebraic_solver(g, config, has_aux)


def _validate_algebraic_output(g, y, z, t, args, p, has_aux):
    output = jax.eval_shape(lambda a, b, c: g(a, b, c, args, p), y, z, t)
    if has_aux:
        if not isinstance(output, tuple) or len(output) != 2:
            raise TypeError("with has_aux=True, g must return (residual, aux)")
        aux = output[1]
        leaves = jax.tree.leaves(aux)
        if not leaves:
            raise ValueError("aux must contain at least one array leaf")
        for leaf in leaves:
            if not jnp.issubdtype(leaf.dtype, jnp.floating):
                raise TypeError("aux leaves must have a real floating dtype")
            if leaf.size == 0:
                raise ValueError("aux leaves must not be empty")
        return aux
    elif isinstance(output, tuple):
        raise TypeError("g returned a tuple; pass has_aux=True for (residual, aux)")
    return None


def _zero_discrete_tangent(value):
    return jnp.zeros(value.shape, dtype=jax.dtypes.float0)


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
    has_aux,
):
    """Wrap nlls roots in a status-safe implicit derivative."""

    def algebraic_output(y, z, t, p):
        return g(y, z, t, args, p)

    def residual(y, z, t, p):
        value = algebraic_output(y, z, t, p)
        return value[0] if has_aux else value

    def auxiliary(y, z, t, p):
        return asarray_aux(algebraic_output(y, z, t, p)[1])

    @jax.custom_jvp
    def solve_root(y, t, z_guess, p, failure_ad_reference):
        result = algebraic_solver.solve(
            z_guess,
            args,
            p=(y, t, p),
            max_steps=root_solver.max_steps,
            atol=root_solver.atol,
        )
        ok = result.status == jnp.asarray(LMStatus.CONVERGED, result.status.dtype)
        value, dtype = asarray_state(result.x, "algebraic root")
        assert_same_structure(z_reference, value, "algebraic root")
        if dtype != z_dtype:
            raise TypeError("the algebraic root must preserve the z dtype")
        # A failed root is not a solution. Returning the finite warm start
        # keeps the static failure prefix usable while `ok` carries validity.
        return where(ok, value, z_guess), ok

    @solve_root.defjvp
    def solve_root_jvp(primals, tangents):
        y, t, z_guess, p, failure_ad_reference = primals
        y_dot, t_dot, _, p_dot, _ = tangents
        z, ok = solve_root(y, t, z_guess, p, failure_ad_reference)
        y_ref, z_ref, t_ref, p_ref = failure_ad_reference
        y_eval = where(ok, y, y_ref)
        z_eval = where(ok, z, z_ref)
        t_eval = jnp.where(ok, t, t_ref)
        p_eval = where(ok, p, p_ref)
        theta, unravel = ravel_pytree(z_eval)

        def residual_theta(theta_value):
            return jnp.ravel(residual(y_eval, unravel(theta_value), t_eval, p_eval))

        jacobian = jax.jacfwd(residual_theta)(theta)

        def residual_inputs(y_value, t_value, p_value):
            return jnp.ravel(residual(y_value, z_eval, t_value, p_value))

        rhs = jax.jvp(
            residual_inputs,
            (y_eval, t_eval, p_eval),
            (y_dot, t_dot, p_dot),
        )[1]
        identity = jnp.eye(theta.size, dtype=theta.dtype)
        jacobian_safe = jnp.where(ok, jacobian, identity)
        rhs_safe = jnp.where(ok, rhs, jnp.zeros_like(rhs))
        z_dot = unravel(jnp.linalg.solve(jacobian_safe, -rhs_safe))
        return (z, ok), (z_dot, _zero_discrete_tangent(ok))

    return solve_root, residual, auxiliary


def _ravel_tangent_like(primal, tangent, dtype):
    """Flatten tangents, replacing discrete float0 leaves by exact zeros."""
    primal_leaves, primal_tree = jax.tree.flatten(primal)
    tangent_leaves, tangent_tree = jax.tree.flatten(tangent)
    if primal_tree != tangent_tree:
        raise TypeError("primal and tangent pytrees must have the same structure")
    pieces = []
    for primal_leaf, tangent_leaf in zip(primal_leaves, tangent_leaves, strict=True):
        primal_leaf = jnp.asarray(primal_leaf)
        tangent_dtype = getattr(tangent_leaf, "dtype", None)
        if tangent_dtype == jax.dtypes.float0:
            pieces.append(jnp.zeros(primal_leaf.size, dtype))
        else:
            pieces.append(jnp.asarray(tangent_leaf, dtype).reshape(-1))
    return jnp.concatenate(pieces) if pieces else jnp.zeros((0,), dtype)


def _make_safe_aux_evaluator(auxiliary, aux_structure):
    """Evaluate aux only on valid roots and give failed lanes zero tangents."""

    def zeros():
        return jax.tree.map(
            lambda leaf: jnp.zeros(leaf.shape, leaf.dtype),
            aux_structure,
        )

    def all_finite(value):
        finite = jnp.asarray(True)
        for leaf in jax.tree.leaves(value):
            finite = finite & jnp.all(jnp.isfinite(leaf))
        return finite

    @jax.custom_jvp
    def evaluate_aux(y, z, t, p, active, failure_ad_reference):
        raw_value = jax.lax.cond(
            active,
            lambda: auxiliary(y, z, t, p),
            zeros,
        )
        ok = active & all_finite(raw_value)
        return where(ok, raw_value, zeros()), ok

    @evaluate_aux.defjvp
    def evaluate_aux_jvp(primals, tangents):
        y, z, t, p, active, failure_ad_reference = primals
        y_dot, z_dot, t_dot, p_dot, _, _ = tangents
        value, ok = evaluate_aux(y, z, t, p, active, failure_ad_reference)
        joint = (y, z, t, p)
        joint_dot = (y_dot, z_dot, t_dot, p_dot)
        theta, unravel = ravel_pytree(joint)
        theta_dot = _ravel_tangent_like(joint, joint_dot, theta.dtype)
        theta_ref, _ = ravel_pytree(failure_ad_reference)
        theta_eval = jnp.where(ok, theta, theta_ref)

        def aux_theta(theta_value):
            y_value, z_value, t_value, p_value = unravel(theta_value)
            return auxiliary(y_value, z_value, t_value, p_value)

        value_dot = jax.jvp(aux_theta, (theta_eval,), (theta_dot,))[1]
        value_dot = where(ok, value_dot, zeros_like(value_dot))
        return (value, ok), (value_dot, _zero_discrete_tangent(ok))

    return evaluate_aux


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
    has_aux=False,
    failure_ad_reference=None,
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
    belong in ``p``; each converged root differentiates implicitly with
    respect to ``(y, t, p)``, while a failed root has a zero tangent. With
    ``has_aux=True``, ``g`` returns ``(residual, aux)``; the floating aux
    pytree is stored at accepted nodes and interpolated on requested grids.
    ``failure_ad_reference=(y, z, t, p)`` may provide a domain-safe point for
    retaining successful-lane derivatives when other ``vmap`` lanes fail.
    A nonfinite aux leaf fails the solve; at the initial point no time-step
    work is attempted after that failure.
    """
    if dt_0 is None:
        raise ValueError("dt_0 is required (tinydiffeq has no initial-step heuristic)")
    if not isinstance(has_aux, bool):
        raise TypeError("has_aux must be a static Python bool")
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

    f = _canonicalize_dae_field(f, "f")
    g_field = _canonicalize_dae_field(g, "g")
    zero_aux = _validate_algebraic_output(g_field, y_0, z_0, t_0, args, p, has_aux)
    algebraic_solver = _get_algebraic_solver(g, root_solver, has_aux)
    solve_root_ad, residual, auxiliary = _make_implicit_root_solver(
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

    def differential(y, z, t):
        value, dtype = asarray_state(f(y, z, t, args, p), "f(y, z, t)")
        assert_same_structure(y_0, value, "f(y, z, t)")
        if dtype != time_dtype:
            raise TypeError("f(y, z, t) must preserve the y dtype")
        return value

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
    need_f = solver.fsal or (save_at.ts is not None)
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
    f_initial = jax.lax.cond(
        initial_ok,
        lambda: differential(y_0, z_initial, t_0),
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
        if has_aux:

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
        if has_aux:
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
        return DAESolution(
            ts=t_final,
            ys=y_final,
            zs=z_final,
            ok=integration_ok,
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
    all_aux = prepend(aux_initial, aux_s) if has_aux else None
    raw_accepted = jnp.concatenate([jnp.ones((1,), bool), adv_s])

    if save_at.steps:
        output_size = max_steps + 1
        accepted_indices = jnp.nonzero(raw_accepted, size=output_size, fill_value=0)[0]
        compact_times = all_times[accepted_indices]
        compact_ys = take(all_ys, accepted_indices)
        compact_zs = take(all_zs, accepted_indices)
        compact_aux = take(all_aux, accepted_indices) if has_aux else None
        accepted = jnp.arange(output_size) <= num_accepted
        last_time = compact_times[num_accepted]
        last_y = take(compact_ys, num_accepted)
        last_z = take(compact_zs, num_accepted)
        last_aux = take(compact_aux, num_accepted) if has_aux else None

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
                if has_aux
                else None
            ),
        )

    fs_all = prepend(f_initial, fs_s)
    z_dots_all = prepend(z_dot_initial, z_dots_s)
    aux_dots_all = prepend(aux_dot_initial, aux_dots_s) if has_aux else None
    query_times = jnp.asarray(save_at.ts, time_dtype)
    values = (all_ys, all_zs, all_aux) if has_aux else (all_ys, all_zs)
    derivatives = (
        (fs_all, z_dots_all, aux_dots_all) if has_aux else (fs_all, z_dots_all)
    )
    interpolated = hermite_interpolate(query_times, all_times, values, derivatives)
    if has_aux:
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
