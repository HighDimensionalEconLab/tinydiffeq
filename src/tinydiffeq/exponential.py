"""Exponential actions for autonomous homogeneous linear ODEs."""

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
from jax._src import ad_util
from jax._src.interpreters import ad, batching
from jax.experimental import hijax
from jax.flatten_util import ravel_pytree

from tinydiffeq._tree import asarray_state, assert_same_structure
from tinydiffeq._unvmap import unvmap_all
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


class _TimeShiftAction(hijax.HiPrim):
    lin, linearized = hijax.linearize_from_jvp

    def __init__(self, mat_shape, vec_shape, dt_shape, dtype):
        self.in_avals = (
            hijax.ShapedArray(mat_shape, dtype),
            hijax.ShapedArray(vec_shape, dtype),
            hijax.ShapedArray(dt_shape, dtype),
        )
        self.out_aval = hijax.ShapedArray(vec_shape, dtype)
        self.params = {
            "mat_shape": mat_shape,
            "vec_shape": vec_shape,
            "dt_shape": dt_shape,
            "dtype": dtype,
        }
        super().__init__()

    def expand(self, matrix, value, dt):
        return value

    def jvp(self, primals, tangents):
        matrix, value, dt = primals
        dM, dv, d_dt = tangents
        out = self(matrix, value, dt)
        dot = jnp.zeros_like(out)
        if type(dv) is not ad_util.Zero:
            dot = dot + self(matrix, dv, dt)
        if type(dM) is not ad_util.Zero:
            mv = jnp.einsum("...ij,...j->...i", dM, value)
            dot = dot + jnp.expand_dims(dt, -1) * self(matrix, mv, dt)
        if type(d_dt) is not ad_util.Zero:
            mv = jnp.einsum("...ij,...j->...i", matrix, value)
            dot = dot + jnp.expand_dims(d_dt, -1) * self(matrix, mv, dt)
        return out, dot

    def vjp_fwd(self, nzs_in, matrix, value, dt):
        return self(matrix, value, dt), (matrix, value, dt)

    def vjp_bwd_retval(self, res, ct):
        matrix, value, dt = res
        shifted = self(jnp.swapaxes(matrix, -1, -2), ct, dt)
        grad_M = jnp.expand_dims(dt, (-1, -2)) * jnp.einsum(
            "...i,...j->...ij", shifted, value
        )
        grad_dt = jnp.sum(
            shifted * jnp.einsum("...ij,...j->...i", matrix, value), axis=-1
        )
        if grad_M.ndim > matrix.ndim:
            grad_M = jnp.sum(grad_M, axis=tuple(range(grad_M.ndim - matrix.ndim)))
        if grad_dt.ndim > dt.ndim:
            grad_dt = jnp.sum(grad_dt, axis=tuple(range(grad_dt.ndim - dt.ndim)))
        return grad_M, shifted, grad_dt

    def transpose(self, out_ct, matrix_acc, value_acc, dt_acc):
        if type(out_ct) is ad_util.Zero:
            return None
        if isinstance(value_acc, ad.GradAccum):
            value_acc.accum(self(jnp.swapaxes(matrix_acc, -1, -2), out_ct, dt_acc))
        return None

    def batch(self, axis_data, args, dims):
        matrix, value, dt = args
        m_d, v_d, t_d = dims
        if m_d is None and v_d is None and t_d is None:
            return self(matrix, value, dt), None
        if m_d is not None:
            matrix = batching.moveaxis(matrix, m_d, 0)
        if v_d is not None:
            value = batching.moveaxis(value, v_d, 0)
        else:
            value = jnp.broadcast_to(value, (axis_data.size,) + value.shape)
        if t_d is not None:
            dt = batching.moveaxis(dt, t_d, 0)
        prim = _TimeShiftAction(matrix.shape, value.shape, dt.shape, value.dtype)
        return prim(matrix, value, dt), 0


def _dense_exponential_action(matrix, vector, time):
    time = jnp.asarray(time, dtype=vector.dtype)
    t_sg = jax.lax.stop_gradient(time)
    dt = time - t_sg
    value = jsp_linalg.expm(t_sg * matrix) @ vector
    prim = _TimeShiftAction(matrix.shape, value.shape, dt.shape, value.dtype)
    return prim(matrix, value, dt)


def _krylov_tolerances(method, dtype):
    if jnp.finfo(dtype).bits <= 32:
        default_rtol, default_atol = 1e-5, 1e-7
    else:
        default_rtol, default_atol = 1e-10, 1e-12
    rtol = default_rtol if method.rtol is None else method.rtol
    atol = default_atol if method.atol is None else method.atol
    return jnp.asarray(rtol, dtype), jnp.asarray(atol, dtype)


def _arnoldi_decomposition(action, vector, subspace_dim, reorthogonalization_passes):
    dimension = vector.shape[0]
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
        action_norm = jnp.linalg.norm(jax.lax.stop_gradient(candidate))
        coefficients = current_basis @ candidate
        candidate = candidate - coefficients @ current_basis
        if reorthogonalization_passes == 2:
            correction = current_basis @ candidate
            candidate = candidate - correction @ current_basis
            coefficients = coefficients + correction
        breakdown_floor = (
            100 * epsilon * jnp.maximum(jnp.asarray(1.0, vector.dtype), action_norm)
        )
        # ``norm`` has an undefined derivative at the exact zero residual of a
        # happy breakdown. Masking ``norm(candidate)`` afterwards is too late:
        # reverse mode still encounters ``0 / 0`` in the norm pullback. Branch
        # on the squared norm first, so the inactive branch reaches ``sqrt`` as
        # a constant and therefore has an exact-zero, finite tangent.
        squared_norm = jnp.real(jnp.vdot(candidate, candidate))
        continues = squared_norm > breakdown_floor**2
        safe_norm = jnp.sqrt(jnp.where(continues, squared_norm, 1))
        next_vector = jnp.where(continues, candidate / safe_norm, 0)
        current_basis = current_basis.at[index + 1].set(next_vector)
        current_hessenberg = current_hessenberg.at[:, index].set(coefficients)
        current_hessenberg = current_hessenberg.at[index + 1, index].set(
            jnp.where(continues, safe_norm, 0)
        )
        return current_basis, current_hessenberg

    basis, hessenberg = jax.lax.fori_loop(
        0, subspace_dim, arnoldi_step, (basis, hessenberg)
    )
    augmented = jnp.zeros((subspace_dim + 1, subspace_dim + 1), vector.dtype)
    augmented = augmented.at[:, :subspace_dim].set(hessenberg)
    return basis, augmented, beta


def _arnoldi_eval_from_decomposition(
    basis, augmented, beta, vector, time, subspace_dim
):
    e_0 = jnp.zeros(subspace_dim + 1, vector.dtype).at[0].set(1)
    first_col = _dense_exponential_action(augmented, e_0, time)
    result = beta * (first_col[:subspace_dim] @ basis[:subspace_dim])
    # The final component of exp(t H_aug)e_1 is the standard leading-term
    # Arnoldi error estimate used by static expv implementations.
    residual = jax.lax.stop_gradient(beta * jnp.abs(first_col[subspace_dim]))
    result = jnp.where(beta > 0, result, vector)
    return result, residual


def _arnoldi_exponential_action(
    action,
    vector,
    time,
    krylov_dim,
    reorthogonalization_passes,
    dense_operator=None,
):
    dimension = vector.shape[0]
    subspace_dim = min(krylov_dim, dimension)
    if subspace_dim == dimension:
        # A full-dimensional Krylov space is the whole linear state space, so
        # the action is exact. Materializing the operator also removes the
        # coordinate singularity of an Arnoldi basis at an early happy
        # breakdown (for example, an eigenvector initial state). In that case
        # the exponential action is smooth even though the normalized Krylov
        # basis is not; differentiating the dense action gives its unique
        # JVP/VJP. This does not change the matrix-free scaling regime, where
        # ``krylov_dim < dimension``.
        if dense_operator is None:
            dense_operator = jax.jacfwd(action)(jnp.zeros_like(vector))
        result = _dense_exponential_action(dense_operator, vector, time)
        return result, jnp.asarray(0.0, vector.dtype)
    basis, augmented, beta = _arnoldi_decomposition(
        action, vector, subspace_dim, reorthogonalization_passes
    )
    return _arnoldi_eval_from_decomposition(
        basis, augmented, beta, vector, time, subspace_dim
    )


def _krylov_propagate(action, vector, time, method, dense_operator=None):
    substep_time = time / method.num_substeps

    def substep(_, carry):
        current, maximum_residual = carry
        next_value, residual = _arnoldi_exponential_action(
            action,
            current,
            substep_time,
            method.krylov_dim,
            method.reorthogonalization_passes,
            dense_operator=dense_operator,
        )
        return next_value, jnp.maximum(maximum_residual, residual)

    result, residual = jax.lax.fori_loop(
        0,
        method.num_substeps,
        substep,
        (vector, jnp.asarray(0.0, vector.dtype)),
    )
    rtol, atol = _krylov_tolerances(method, vector.dtype)
    accurate = residual <= atol + rtol * jnp.linalg.norm(jax.lax.stop_gradient(result))
    return result, accurate


def _adaptive_krylov_propagate(action, vector, time, method, dense_operator=None):
    """Propagate with residual-controlled time slices and a bounded scan."""
    dtype = vector.dtype
    zero = jnp.asarray(0.0, dtype)
    one = jnp.asarray(1.0, dtype)
    total = jnp.maximum(jnp.asarray(time, dtype), zero)
    total_sg = jax.lax.stop_gradient(total)
    rtol, atol = _krylov_tolerances(method, dtype)
    epsilon = jnp.asarray(jnp.finfo(dtype).eps, dtype)
    time_scale = jnp.maximum(one, total_sg)
    time_tolerance = 8 * epsilon * time_scale
    minimum_step = 16 * epsilon * time_scale
    if method.initial_step is None:
        first_step = total_sg
    else:
        first_step = jnp.minimum(jnp.asarray(method.initial_step, dtype), total_sg)
    first_step = jnp.where(total_sg > zero, jnp.maximum(first_step, minimum_step), one)

    def attempt(carry, _):
        current, elapsed, step, done, failed, accepted, rejected = carry
        inactive = done | failed

        def advance(active_carry):
            current, elapsed, step, done, failed, accepted, rejected = active_carry
            remaining = jnp.maximum(total - elapsed, zero)
            step_used = jnp.where(
                step < jax.lax.stop_gradient(remaining), step, remaining
            )
            candidate, residual = _arnoldi_exponential_action(
                action,
                current,
                step_used,
                method.krylov_dim,
                method.reorthogonalization_passes,
                dense_operator=dense_operator,
            )
            curr_sg = jax.lax.stop_gradient(current)
            cand_sg = jax.lax.stop_gradient(candidate)
            res_sg = jax.lax.stop_gradient(residual)
            step_sg = jax.lax.stop_gradient(step_used)
            global_scale = atol + rtol * jnp.maximum(
                jnp.linalg.norm(curr_sg), jnp.linalg.norm(cand_sg)
            )
            # Budget local error in proportion to the fraction of the full
            # interval advanced. Summed accepted residual estimates then obey
            # the requested endpoint scale rather than an N-times-looser one.
            interval_fraction = step_sg / jnp.maximum(total_sg, minimum_step)
            scale = jnp.maximum(global_scale * interval_fraction, jnp.finfo(dtype).tiny)
            error_ratio = res_sg / scale
            finite = jnp.all(jnp.isfinite(cand_sg)) & jnp.isfinite(error_ratio)
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
                jnp.isfinite(raw_factor),
                raw_factor,
                jnp.asarray(method.min_factor, dtype),
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

            next_elapsed = jnp.where(accept, elapsed + step_sg, elapsed)
            reached_end = accept & (total_sg - next_elapsed <= time_tolerance)
            next_elapsed = jnp.where(reached_end, total_sg, next_elapsed)
            next_current = jnp.where(accept, candidate, current)
            next_remaining = jnp.maximum(total_sg - next_elapsed, zero)
            proposed_step = step_sg * factor
            next_step = jnp.minimum(proposed_step, next_remaining)
            next_step = jnp.where(
                reached_end, step, jnp.maximum(next_step, minimum_step)
            )
            stalled = (~accept) & (step_sg <= minimum_step)
            updated = (
                next_current,
                next_elapsed,
                next_step,
                reached_end,
                failed | stalled,
                accepted + accept.astype(jnp.int32),
                rejected + (~accept).astype(jnp.int32),
            )
            return jax.tree.map(
                lambda old, new: jnp.where(inactive, old, new),
                active_carry,
                updated,
            )

        next_carry = jax.lax.cond(
            unvmap_all(inactive), lambda value: value, advance, carry
        )
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


def _propagate_krylov(action, vector, time, method, dense_operator=None):
    if isinstance(method, AdaptiveKrylovExponential):
        value, ok, accepted, rejected = _adaptive_krylov_propagate(
            action, vector, time, method, dense_operator=dense_operator
        )
        return value, ok, accepted, accepted + rejected
    value, ok = _krylov_propagate(
        action, vector, time, method, dense_operator=dense_operator
    )
    accepted = jnp.where(time > 0, method.num_substeps, 0).astype(jnp.int32)
    return value, ok, accepted, accepted


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
    ``A @ x`` or a callable mapping the state pytree to an identically
    structured pytree. ``DenseExponential`` materializes a callable operator
    before a dense matrix exponential; the Krylov methods evaluate only
    operator actions. Endpoint output is the default, and ``SaveAt(ts=...)``
    evaluates independent exponential actions at the requested times. JVPs
    and VJPs flow through the initial state and differentiable operator
    arrays. The operator must be autonomous, homogeneous, and linear.
    """
    if not isinstance(method, _EXPONENTIAL_METHODS):
        raise TypeError(
            "method must be DenseExponential, KrylovExponential, or "
            "AdaptiveKrylovExponential"
        )
    if save_at is None:
        save_at = SaveAt(t_1=True)
    if save_at.exact:
        raise ValueError("SaveAt exact=True is only supported by solve_ode")
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

    dimension = flat_initial.shape[0]
    full_krylov = not isinstance(method, DenseExponential) and (
        min(method.krylov_dim, dimension) == dimension
    )
    use_dense = isinstance(method, DenseExponential) or (
        isinstance(method, KrylovExponential)
        and method.num_substeps == 1
        and full_krylov
    )
    if (use_dense or full_krylov) and dense_operator is None:
        dense_operator = jax.jacfwd(flat_action)(jnp.zeros_like(flat_initial))

    single_krylov = (
        not use_dense
        and not save_at.t_1
        and isinstance(method, KrylovExponential)
        and method.num_substeps == 1
    )
    if single_krylov:
        subspace_dim = min(method.krylov_dim, dimension)
        basis, augmented, beta = _arnoldi_decomposition(
            flat_action,
            flat_initial,
            subspace_dim,
            method.reorthogonalization_passes,
        )
        rtol, atol = _krylov_tolerances(method, dtype)

    def evaluate(time):
        elapsed = time - t_0
        if use_dense:
            value = _dense_exponential_action(dense_operator, flat_initial, elapsed)
            count = jnp.where(elapsed > 0, 1, 0).astype(jnp.int32)
            return value, jnp.asarray(True), count, count
        if single_krylov:
            value, residual = _arnoldi_eval_from_decomposition(
                basis, augmented, beta, flat_initial, elapsed, subspace_dim
            )
            accurate = residual <= atol + rtol * jnp.linalg.norm(
                jax.lax.stop_gradient(value)
            )
            count = jnp.where(elapsed > 0, 1, 0).astype(jnp.int32)
            return value, accurate, count, count
        return _propagate_krylov(
            flat_action,
            flat_initial,
            elapsed,
            method,
            dense_operator=dense_operator,
        )

    if save_at.t_1:
        times = t_1
        flat_states, method_ok, num_accepted, num_steps = evaluate(t_1)
        states = unravel(flat_states)
        times_ok = t_1 >= t_0
    else:
        times = jnp.asarray(save_at.ts, dtype)
        if times.ndim != 1:
            raise TypeError("SaveAt.ts must be one-dimensional")
        times_ok = jnp.all((times >= t_0) & (times <= t_1))
        flat_states, method_ok, accepted_counts, step_counts = jax.vmap(evaluate)(times)
        num_accepted = jnp.max(accepted_counts, initial=jnp.asarray(0, jnp.int32))
        num_steps = jnp.max(step_counts, initial=jnp.asarray(0, jnp.int32))
        states = jax.vmap(unravel)(flat_states)
    finite = jnp.all(jnp.isfinite(flat_states))
    return Solution(
        ts=times,
        xs=states,
        ok=times_ok & jnp.all(method_ok) & finite,
        num_accepted=num_accepted,
        num_steps=num_steps,
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
        return exponential @ initial, jnp.asarray(True), exponential, count, count
    value, ok, accepted, steps = _propagate_krylov(
        action, initial, elapsed, method, dense_operator=dense_operator
    )
    return value, ok, None, accepted, steps


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
    dimension = flat_initial.shape[0]
    full_krylov = not isinstance(method, DenseExponential) and (
        min(method.krylov_dim, dimension) == dimension
    )
    if (isinstance(method, DenseExponential) or full_krylov) and dense_operator is None:
        dense_operator = jax.jacfwd(flat_action)(jnp.zeros_like(flat_initial))
    t_0 = jnp.asarray(t_0, dtype)
    t_1 = jnp.asarray(t_1, dtype)
    elapsed = t_1 - t_0
    flat_value, primal_ok, exponential, num_accepted, num_steps = _terminal_value(
        method, dense_operator, flat_action, flat_initial, elapsed
    )
    if isinstance(method, DenseExponential):
        flat_tangent = tangent @ exponential.T if batched else exponential @ tangent
        tangent_ok = jnp.asarray(True)
    elif batched:
        flat_tangent, tangent_ok, _, _ = jax.vmap(
            lambda direction: _propagate_krylov(
                flat_action,
                direction,
                elapsed,
                method,
                dense_operator=dense_operator,
            )
        )(tangent)
    else:
        flat_tangent, tangent_ok, _, _ = _propagate_krylov(
            flat_action, tangent, elapsed, method, dense_operator=dense_operator
        )
    finite = jnp.all(jnp.isfinite(flat_value)) & jnp.all(jnp.isfinite(flat_tangent))
    solution = Solution(
        ts=t_1,
        xs=unravel(flat_value),
        ok=(t_1 >= t_0) & primal_ok & jnp.all(tangent_ok) & finite,
        num_accepted=num_accepted,
        num_steps=num_steps,
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
    dimension = flat_initial.shape[0]
    full_krylov = not isinstance(method, DenseExponential) and (
        min(method.krylov_dim, dimension) == dimension
    )
    if (isinstance(method, DenseExponential) or full_krylov) and dense_operator is None:
        dense_operator = jax.jacfwd(flat_action)(jnp.zeros_like(flat_initial))
    t_0 = jnp.asarray(t_0, dtype)
    t_1 = jnp.asarray(t_1, dtype)
    elapsed = t_1 - t_0
    flat_value, primal_ok, exponential, num_accepted, num_steps = _terminal_value(
        method, dense_operator, flat_action, flat_initial, elapsed
    )
    if isinstance(method, DenseExponential):
        flat_gradient = (
            flat_cotangent @ exponential if batched else exponential.T @ flat_cotangent
        )
        gradient_ok = jnp.asarray(True)
    else:
        zero = jnp.zeros_like(flat_initial)
        transpose_dense = None if dense_operator is None else dense_operator.T

        def transpose_action(vector):
            return jax.linear_transpose(flat_action, zero)(vector)[0]

        if batched:
            flat_gradient, gradient_ok, _, _ = jax.vmap(
                lambda direction: _propagate_krylov(
                    transpose_action,
                    direction,
                    elapsed,
                    method,
                    dense_operator=transpose_dense,
                )
            )(flat_cotangent)
        else:
            flat_gradient, gradient_ok, _, _ = _propagate_krylov(
                transpose_action,
                flat_cotangent,
                elapsed,
                method,
                dense_operator=transpose_dense,
            )
    finite = jnp.all(jnp.isfinite(flat_value)) & jnp.all(jnp.isfinite(flat_gradient))
    solution = Solution(
        ts=t_1,
        xs=unravel(flat_value),
        ok=(t_1 >= t_0) & primal_ok & jnp.all(gradient_ok) & finite,
        num_accepted=num_accepted,
        num_steps=num_steps,
    )
    return solution, restore_cotangent(flat_gradient)
