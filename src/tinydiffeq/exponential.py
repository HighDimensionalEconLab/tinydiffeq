"""Exponential actions for autonomous homogeneous linear ODEs."""

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
from jax.flatten_util import ravel_pytree

from tinydiffeq._tree import asarray_state, assert_same_structure
from tinydiffeq.save_at import SaveAt
from tinydiffeq.solution import Solution


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class DenseExponential:
    """Dense scaling-and-squaring exponential for a fixed linear operator."""


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class KrylovExponential:
    """Static Arnoldi exponential action for an array or callable operator.

    ``krylov_dim``, ``num_substeps``, and ``reorthogonalization_passes`` are
    static compilation controls. Two-pass classical Gram--Schmidt is the
    stable default; one pass reduces the dominant basis memory traffic when
    the operator has been validated for it. The precision-dependent default
    error tolerances are ``1e-5``/``1e-7`` for float32 and
    ``1e-10``/``1e-12`` for float64.
    """

    krylov_dim: int = field(default=30, metadata=dict(static=True))
    num_substeps: int = field(default=1, metadata=dict(static=True))
    reorthogonalization_passes: int = field(default=2, metadata=dict(static=True))
    rtol: float | None = None
    atol: float | None = None

    def __post_init__(self):
        if not isinstance(self.krylov_dim, int) or isinstance(self.krylov_dim, bool):
            raise TypeError("KrylovExponential.krylov_dim must be a positive int")
        if self.krylov_dim < 1:
            raise ValueError("KrylovExponential.krylov_dim must be a positive int")
        if not isinstance(self.num_substeps, int) or isinstance(
            self.num_substeps, bool
        ):
            raise TypeError("KrylovExponential.num_substeps must be a positive int")
        if self.num_substeps < 1:
            raise ValueError("KrylovExponential.num_substeps must be a positive int")
        if (
            not isinstance(self.reorthogonalization_passes, int)
            or isinstance(self.reorthogonalization_passes, bool)
            or self.reorthogonalization_passes not in (1, 2)
        ):
            raise ValueError(
                "KrylovExponential.reorthogonalization_passes must be 1 or 2"
            )
        for name, value in (("rtol", self.rtol), ("atol", self.atol)):
            if value is not None and value < 0:
                raise ValueError(f"KrylovExponential.{name} must be nonnegative")


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class AdaptiveKrylovExponential:
    """Adaptive matrix-free Arnoldi exponential action.

    The Krylov dimension remains static while accepted internal time slices
    adapt to the leading-term Arnoldi residual. ``max_steps`` bounds all
    accepted and rejected attempts, keeping compiled shapes static. The
    precision-dependent tolerance defaults match ``KrylovExponential``.
    """

    krylov_dim: int = field(default=30, metadata=dict(static=True))
    max_steps: int = field(default=128, metadata=dict(static=True))
    reorthogonalization_passes: int = field(default=2, metadata=dict(static=True))
    initial_step: float | None = None
    safety: float = 0.9
    min_factor: float = 0.2
    max_factor: float = 5.0
    rtol: float | None = None
    atol: float | None = None

    def __post_init__(self):
        for name, value in (
            ("krylov_dim", self.krylov_dim),
            ("max_steps", self.max_steps),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(
                    f"AdaptiveKrylovExponential.{name} must be a positive int"
                )
            if value < 1:
                raise ValueError(
                    f"AdaptiveKrylovExponential.{name} must be a positive int"
                )
        if (
            not isinstance(self.reorthogonalization_passes, int)
            or isinstance(self.reorthogonalization_passes, bool)
            or self.reorthogonalization_passes not in (1, 2)
        ):
            raise ValueError(
                "AdaptiveKrylovExponential.reorthogonalization_passes must be 1 or 2"
            )
        if self.initial_step is not None and self.initial_step <= 0:
            raise ValueError("AdaptiveKrylovExponential.initial_step must be positive")
        if not 0 < self.safety <= 1:
            raise ValueError("AdaptiveKrylovExponential.safety must be in (0, 1]")
        if not 0 < self.min_factor < 1:
            raise ValueError("AdaptiveKrylovExponential.min_factor must be in (0, 1)")
        if self.max_factor <= 1:
            raise ValueError(
                "AdaptiveKrylovExponential.max_factor must be greater than 1"
            )
        for name, value in (("rtol", self.rtol), ("atol", self.atol)):
            if value is not None and value < 0:
                raise ValueError(
                    f"AdaptiveKrylovExponential.{name} must be nonnegative"
                )


@jax.custom_jvp
def _dense_exponential_action(matrix, vector, time):
    return jsp_linalg.expm(time * matrix) @ vector


def _dense_exponential_action_jvp(primals, tangents):
    matrix, vector, time = primals
    matrix_tangent, vector_tangent, time_tangent = tangents
    scaled_matrix = time * matrix
    matrix_active = not isinstance(matrix_tangent, jax.custom_derivatives.SymbolicZero)
    time_active = not isinstance(time_tangent, jax.custom_derivatives.SymbolicZero)
    vector_active = not isinstance(vector_tangent, jax.custom_derivatives.SymbolicZero)
    if matrix_active or time_active:
        scaled_tangent = (
            time * matrix_tangent if matrix_active else jnp.zeros_like(matrix)
        ) + (time_tangent * matrix if time_active else jnp.zeros_like(matrix))
        exponential, frechet = jsp_linalg.expm_frechet(
            scaled_matrix,
            scaled_tangent,
            compute_expm=True,
        )
    else:
        exponential = jsp_linalg.expm(scaled_matrix)
        frechet = jnp.zeros_like(matrix)
    value = exponential @ vector
    tangent = frechet @ vector
    if vector_active:
        tangent = tangent + exponential @ vector_tangent
    return value, tangent


_dense_exponential_action.defjvp(_dense_exponential_action_jvp, symbolic_zeros=True)


def _krylov_tolerances(method, dtype):
    if jnp.finfo(dtype).bits <= 32:
        default_rtol, default_atol = 1e-5, 1e-7
    else:
        default_rtol, default_atol = 1e-10, 1e-12
    rtol = default_rtol if method.rtol is None else method.rtol
    atol = default_atol if method.atol is None else method.atol
    return jnp.asarray(rtol, dtype), jnp.asarray(atol, dtype)


def _arnoldi_exponential_action(
    action, vector, time, krylov_dim, reorthogonalization_passes
):
    dimension = vector.shape[0]
    subspace_dim = min(krylov_dim, dimension)
    beta = jnp.linalg.norm(vector)
    beta_safe = jnp.where(beta > 0, beta, 1)
    # Each Krylov vector is a contiguous row. This orientation is materially
    # faster with the row-major layouts used by JAX/XLA than the conventional
    # mathematical n-by-m column layout.
    basis = jnp.zeros((subspace_dim + 1, dimension), vector.dtype)
    basis = basis.at[0].set(vector / beta_safe)
    hessenberg = jnp.zeros((subspace_dim + 1, subspace_dim), vector.dtype)
    epsilon = jnp.asarray(jnp.finfo(vector.dtype).eps, vector.dtype)

    def arnoldi_step(index, carry):
        current_basis, current_hessenberg = carry
        candidate = action(current_basis[index])
        action_norm = jnp.linalg.norm(candidate)
        coefficients = current_basis @ candidate
        candidate = candidate - coefficients @ current_basis
        if reorthogonalization_passes == 2:
            correction = current_basis @ candidate
            candidate = candidate - correction @ current_basis
            coefficients = coefficients + correction
        next_norm = jnp.linalg.norm(candidate)
        breakdown_floor = (
            100 * epsilon * jnp.maximum(jnp.asarray(1.0, vector.dtype), action_norm)
        )
        continues = next_norm > breakdown_floor
        safe_norm = jnp.where(continues, next_norm, 1)
        next_vector = jnp.where(continues, candidate / safe_norm, 0)
        current_basis = current_basis.at[index + 1].set(next_vector)
        current_hessenberg = current_hessenberg.at[:, index].set(coefficients)
        current_hessenberg = current_hessenberg.at[index + 1, index].set(
            jnp.where(continues, next_norm, 0)
        )
        return current_basis, current_hessenberg

    basis, hessenberg = jax.lax.fori_loop(
        0, subspace_dim, arnoldi_step, (basis, hessenberg)
    )
    augmented = jnp.zeros((subspace_dim + 1, subspace_dim + 1), vector.dtype)
    augmented = augmented.at[:, :subspace_dim].set(hessenberg)
    projected_exponential = jsp_linalg.expm(time * augmented)
    result = beta * (projected_exponential[:subspace_dim, 0] @ basis[:subspace_dim])
    # The final component of exp(t H_aug)e_1 is the standard leading-term
    # Arnoldi error estimate used by static expv implementations.
    residual = beta * jnp.abs(projected_exponential[subspace_dim, 0])
    result = jnp.where(beta > 0, result, vector)
    return result, residual


def _krylov_propagate(action, vector, time, method):
    substep_time = time / method.num_substeps

    def substep(_, carry):
        current, maximum_residual = carry
        next_value, residual = _arnoldi_exponential_action(
            action,
            current,
            substep_time,
            method.krylov_dim,
            method.reorthogonalization_passes,
        )
        return next_value, jnp.maximum(maximum_residual, residual)

    result, residual = jax.lax.fori_loop(
        0,
        method.num_substeps,
        substep,
        (vector, jnp.asarray(0.0, vector.dtype)),
    )
    rtol, atol = _krylov_tolerances(method, vector.dtype)
    accurate = residual <= atol + rtol * jnp.linalg.norm(result)
    return result, accurate


def _adaptive_krylov_propagate(action, vector, time, method):
    """Propagate with residual-controlled time slices and a bounded scan."""
    dtype = vector.dtype
    zero = jnp.asarray(0.0, dtype)
    one = jnp.asarray(1.0, dtype)
    total = jnp.maximum(jnp.asarray(time, dtype), zero)
    rtol, atol = _krylov_tolerances(method, dtype)
    epsilon = jnp.asarray(jnp.finfo(dtype).eps, dtype)
    time_scale = jnp.maximum(one, total)
    time_tolerance = 8 * epsilon * time_scale
    minimum_step = 16 * epsilon * time_scale
    if method.initial_step is None:
        first_step = total
    else:
        first_step = jnp.minimum(jnp.asarray(method.initial_step, dtype), total)
    first_step = jnp.where(total > zero, jnp.maximum(first_step, minimum_step), one)

    def attempt(carry, _):
        current, elapsed, step, done, failed, accepted, rejected = carry

        def advance(active_carry):
            current, elapsed, step, done, failed, accepted, rejected = active_carry
            remaining = jnp.maximum(total - elapsed, zero)
            step_used = jnp.minimum(step, remaining)
            candidate, residual = _arnoldi_exponential_action(
                action,
                current,
                step_used,
                method.krylov_dim,
                method.reorthogonalization_passes,
            )
            global_scale = atol + rtol * jnp.maximum(
                jnp.linalg.norm(current), jnp.linalg.norm(candidate)
            )
            # Budget local error in proportion to the fraction of the full
            # interval advanced. Summed accepted residual estimates then obey
            # the requested endpoint scale rather than an N-times-looser one.
            interval_fraction = step_used / jnp.maximum(total, minimum_step)
            scale = jnp.maximum(global_scale * interval_fraction, jnp.finfo(dtype).tiny)
            error_ratio = residual / scale
            finite = jnp.all(jnp.isfinite(candidate)) & jnp.isfinite(error_ratio)
            accept = finite & (error_ratio <= one)

            # An epsilon floor is enough to select the maximum growth factor
            # and avoids enormous derivatives of ratio**exponent at an exact
            # happy breakdown (notably in float32 reverse mode).
            safe_ratio = jnp.maximum(error_ratio, epsilon)
            # Division by the step fraction changes an O(h^m) local residual
            # ratio into O(h^(m-1)).
            exponent = -one / jnp.asarray(max(method.krylov_dim - 1, 1), dtype)
            raw_factor = jnp.asarray(method.safety, dtype) * safe_ratio**exponent
            raw_factor = jnp.where(
                jnp.isfinite(raw_factor), raw_factor, method.min_factor
            )
            accepted_factor = jnp.clip(
                raw_factor,
                jnp.asarray(method.min_factor, dtype),
                jnp.asarray(method.max_factor, dtype),
            )
            rejected_factor = jnp.clip(
                raw_factor,
                jnp.asarray(method.min_factor, dtype),
                jnp.asarray(0.9, dtype),
            )
            factor = jnp.where(accept, accepted_factor, rejected_factor)

            next_elapsed = jnp.where(accept, elapsed + step_used, elapsed)
            reached_end = accept & (total - next_elapsed <= time_tolerance)
            next_elapsed = jnp.where(reached_end, total, next_elapsed)
            next_current = jnp.where(accept, candidate, current)
            next_remaining = jnp.maximum(total - next_elapsed, zero)
            proposed_step = step_used * factor
            next_step = jnp.minimum(proposed_step, next_remaining)
            next_step = jnp.where(
                reached_end, step, jnp.maximum(next_step, minimum_step)
            )
            stalled = (~accept) & (step_used <= minimum_step)
            return (
                next_current,
                next_elapsed,
                next_step,
                reached_end,
                failed | stalled,
                accepted + accept.astype(jnp.int32),
                rejected + (~accept).astype(jnp.int32),
            )

        inactive = done | failed
        next_carry = jax.lax.cond(inactive, lambda value: value, advance, carry)
        return next_carry, None

    initial = (
        vector,
        zero,
        first_step,
        total == zero,
        jnp.asarray(False),
        jnp.asarray(0, jnp.int32),
        jnp.asarray(0, jnp.int32),
    )
    final, _ = jax.lax.scan(attempt, initial, xs=None, length=method.max_steps)
    result, _, _, done, failed, accepted, rejected = final
    return result, done & ~failed, accepted, rejected


_EXPONENTIAL_METHODS = (
    DenseExponential,
    KrylovExponential,
    AdaptiveKrylovExponential,
)


def _propagate_krylov(action, vector, time, method):
    if isinstance(method, AdaptiveKrylovExponential):
        value, ok, accepted, _ = _adaptive_krylov_propagate(
            action, vector, time, method
        )
        return value, ok, accepted
    value, ok = _krylov_propagate(action, vector, time, method)
    accepted = jnp.where(time > 0, method.num_substeps, 0).astype(jnp.int32)
    return value, ok, accepted


def _prepare_linear_operator(operator, x_0):
    x_0, dtype = asarray_state(x_0, "x_0")
    flat_initial, unravel = ravel_pytree(x_0)
    reference_structure = jax.tree.structure(x_0)

    if callable(operator):

        def flat_action(vector):
            state = unravel(vector)
            result, result_dtype = asarray_state(operator(state), "operator(x)")
            assert_same_structure(x_0, result, "operator(x)")
            if result_dtype != dtype:
                raise TypeError("operator(x) must preserve the state dtype")
            if jax.tree.structure(result) != reference_structure:
                raise ValueError("operator(x) must preserve the state pytree")
            return ravel_pytree(result)[0]

        dense_operator = None
    else:
        dense_operator = jnp.asarray(operator)
        if dense_operator.dtype != dtype:
            raise TypeError("operator matrix and x_0 must have the same dtype")
        dimension = flat_initial.size
        if dense_operator.shape != (dimension, dimension):
            raise ValueError("operator matrix must be square with size ravel(x_0).size")

        def flat_action(vector):
            return dense_operator @ vector

    return x_0, dtype, flat_initial, unravel, flat_action, dense_operator


def solve_linear_ode(operator, method, t_0, t_1, x_0, *, save_at=None):
    """Solve ``dx/dt = A(x)`` for a fixed homogeneous linear operator.

    ``operator`` is either a square matrix using the column convention
    ``A @ x`` or a callable that maps the state pytree to an identically
    structured pytree. ``DenseExponential`` materializes a callable operator
    with forward-mode Jacobian columns before applying a dense matrix
    exponential. ``KrylovExponential`` and ``AdaptiveKrylovExponential`` only
    evaluate operator actions and are therefore suitable for structured
    matrix-free pytrees.

    Endpoint output is the default. ``SaveAt(ts=...)`` evaluates independent
    exponential actions at the requested times. The solve supports ordinary
    JVPs and VJPs through the initial state and through differentiable arrays
    used by the operator. The operator must be autonomous, homogeneous, and
    linear; affine and nonlinear exponential methods are a separate problem.
    """
    if not isinstance(method, _EXPONENTIAL_METHODS):
        raise TypeError(
            "method must be DenseExponential, KrylovExponential, or "
            "AdaptiveKrylovExponential"
        )
    if save_at is None:
        save_at = SaveAt(t_1=True)
    if save_at.steps:
        raise ValueError("linear exponential solves require endpoint or SaveAt.ts")

    (
        x_0,
        dtype,
        flat_initial,
        unravel,
        flat_action,
        dense_operator,
    ) = _prepare_linear_operator(operator, x_0)
    t_0 = jnp.asarray(t_0, dtype)
    t_1 = jnp.asarray(t_1, dtype)

    if isinstance(method, DenseExponential) and dense_operator is None:
        dense_operator = jax.jacfwd(flat_action)(jnp.zeros_like(flat_initial))

    def evaluate(time):
        elapsed = time - t_0
        if isinstance(method, DenseExponential):
            value = _dense_exponential_action(dense_operator, flat_initial, elapsed)
            count = jnp.where(elapsed > 0, 1, 0).astype(jnp.int32)
            return value, jnp.asarray(True), count
        return _propagate_krylov(flat_action, flat_initial, elapsed, method)

    if save_at.t_1:
        times = t_1
        flat_states, method_ok, num_accepted = evaluate(t_1)
        states = unravel(flat_states)
        times_ok = t_1 >= t_0
    else:
        times = jnp.asarray(save_at.ts, dtype)
        if times.ndim != 1:
            raise TypeError("SaveAt.ts must be one-dimensional")
        times_ok = jnp.all((times >= t_0) & (times <= t_1))
        flat_states, method_ok, counts = jax.vmap(evaluate)(times)
        num_accepted = jnp.max(counts, initial=jnp.asarray(0, jnp.int32))
        states = jax.vmap(unravel)(flat_states)
    finite = jnp.all(jnp.isfinite(flat_states))
    return Solution(
        ts=times,
        xs=states,
        ok=times_ok & jnp.all(method_ok) & finite,
        num_accepted=num_accepted,
    )


def _flatten_directions(directions, reference, unravel, *, batched, name):
    directions, dtype = asarray_state(directions, name)
    assert_same_structure(reference, directions, name)
    reference_leaves = jax.tree.leaves(reference)
    direction_leaves = jax.tree.leaves(directions)
    if batched:
        batch_size = direction_leaves[0].shape[0]
        for reference_leaf, direction_leaf in zip(
            reference_leaves, direction_leaves, strict=True
        ):
            if direction_leaf.shape != (batch_size,) + reference_leaf.shape:
                raise ValueError(
                    f"batched {name} leaves must have shape "
                    "(batch_size,) + corresponding state shape"
                )
        flat = jnp.concatenate(
            [leaf.reshape(batch_size, -1) for leaf in direction_leaves], axis=1
        )
        restore = jax.vmap(unravel)
    else:
        for reference_leaf, direction_leaf in zip(
            reference_leaves, direction_leaves, strict=True
        ):
            if direction_leaf.shape != reference_leaf.shape:
                raise ValueError(f"{name} leaves must match the state leaf shapes")
        flat = ravel_pytree(directions)[0]
        restore = unravel
    return flat, restore, dtype


def _terminal_value(method, dense_operator, action, initial, elapsed):
    if isinstance(method, DenseExponential):
        exponential = jsp_linalg.expm(elapsed * dense_operator)
        count = jnp.where(elapsed > 0, 1, 0).astype(jnp.int32)
        return exponential @ initial, jnp.asarray(True), exponential, count
    value, ok, count = _propagate_krylov(action, initial, elapsed, method)
    return value, ok, None, count


def jvp_linear_ode(
    operator,
    method,
    t_0,
    t_1,
    x_0,
    x_0_tangent,
    *,
    batched=False,
):
    """Return a terminal linear solve and hand-coded initial-state JVP.

    With ``batched=True``, every tangent leaf has a leading direction axis.
    Dense mode forms one exponential and applies it to every direction.
    Matrix-free Krylov mode vectorizes independent exponential actions. The
    operator is fixed: use ordinary ``jax.jvp`` when differentiating operator
    entries or arrays captured by a callable.
    """
    if not isinstance(method, _EXPONENTIAL_METHODS):
        raise TypeError(
            "method must be DenseExponential, KrylovExponential, or "
            "AdaptiveKrylovExponential"
        )
    (
        x_0,
        dtype,
        flat_initial,
        unravel,
        flat_action,
        dense_operator,
    ) = _prepare_linear_operator(operator, x_0)
    tangent, restore_tangent, tangent_dtype = _flatten_directions(
        x_0_tangent,
        x_0,
        unravel,
        batched=batched,
        name="x_0_tangent",
    )
    if tangent_dtype != dtype:
        raise TypeError("x_0_tangent must have the same dtype as x_0")
    if isinstance(method, DenseExponential) and dense_operator is None:
        dense_operator = jax.jacfwd(flat_action)(jnp.zeros_like(flat_initial))
    t_0 = jnp.asarray(t_0, dtype)
    t_1 = jnp.asarray(t_1, dtype)
    elapsed = t_1 - t_0
    flat_value, primal_ok, exponential, num_accepted = _terminal_value(
        method, dense_operator, flat_action, flat_initial, elapsed
    )
    if isinstance(method, DenseExponential):
        flat_tangent = tangent @ exponential.T if batched else exponential @ tangent
        tangent_ok = jnp.asarray(True)
    elif batched:
        flat_tangent, tangent_ok, _ = jax.vmap(
            lambda direction: _propagate_krylov(flat_action, direction, elapsed, method)
        )(tangent)
    else:
        flat_tangent, tangent_ok, _ = _propagate_krylov(
            flat_action, tangent, elapsed, method
        )
    finite = jnp.all(jnp.isfinite(flat_value)) & jnp.all(jnp.isfinite(flat_tangent))
    solution = Solution(
        ts=t_1,
        xs=unravel(flat_value),
        ok=(t_1 >= t_0) & primal_ok & jnp.all(tangent_ok) & finite,
        num_accepted=num_accepted,
    )
    return solution, restore_tangent(flat_tangent)


def vjp_linear_ode(
    operator,
    method,
    t_0,
    t_1,
    x_0,
    cotangent,
    *,
    batched=False,
):
    """Return a terminal linear solve and hand-coded initial-state VJP.

    The pullback is another exponential action with the transposed operator.
    ``batched=True`` accepts multiple terminal cotangents on a leading axis.
    Dense mode reuses the primal exponential for every cotangent. Callable
    transpose actions are generated with ``jax.linear_transpose`` and therefore
    require the declared operator to be linear in its state argument.
    """
    if not isinstance(method, _EXPONENTIAL_METHODS):
        raise TypeError(
            "method must be DenseExponential, KrylovExponential, or "
            "AdaptiveKrylovExponential"
        )
    (
        x_0,
        dtype,
        flat_initial,
        unravel,
        flat_action,
        dense_operator,
    ) = _prepare_linear_operator(operator, x_0)
    flat_cotangent, restore_cotangent, cotangent_dtype = _flatten_directions(
        cotangent,
        x_0,
        unravel,
        batched=batched,
        name="cotangent",
    )
    if cotangent_dtype != dtype:
        raise TypeError("cotangent must have the same dtype as x_0")
    if isinstance(method, DenseExponential) and dense_operator is None:
        dense_operator = jax.jacfwd(flat_action)(jnp.zeros_like(flat_initial))
    t_0 = jnp.asarray(t_0, dtype)
    t_1 = jnp.asarray(t_1, dtype)
    elapsed = t_1 - t_0
    flat_value, primal_ok, exponential, num_accepted = _terminal_value(
        method, dense_operator, flat_action, flat_initial, elapsed
    )
    if isinstance(method, DenseExponential):
        flat_gradient = (
            flat_cotangent @ exponential if batched else exponential.T @ flat_cotangent
        )
        gradient_ok = jnp.asarray(True)
    else:
        zero = jnp.zeros_like(flat_initial)

        def transpose_action(vector):
            return jax.linear_transpose(flat_action, zero)(vector)[0]

        if batched:
            flat_gradient, gradient_ok, _ = jax.vmap(
                lambda direction: _propagate_krylov(
                    transpose_action, direction, elapsed, method
                )
            )(flat_cotangent)
        else:
            flat_gradient, gradient_ok, _ = _propagate_krylov(
                transpose_action, flat_cotangent, elapsed, method
            )
    finite = jnp.all(jnp.isfinite(flat_value)) & jnp.all(jnp.isfinite(flat_gradient))
    solution = Solution(
        ts=t_1,
        xs=unravel(flat_value),
        ok=(t_1 >= t_0) & primal_ok & jnp.all(gradient_ok) & finite,
        num_accepted=num_accepted,
    )
    return solution, restore_cotangent(flat_gradient)
