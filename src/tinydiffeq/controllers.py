from dataclasses import dataclass

import jax
import jax.numpy as jnp

# Controllers decide accept/reject and the next step size from the embedded
# error estimate. Contract: init(x_0) -> state, then
# adapt(x_0, x_1, err, dt_used, dt_prev, order, state) ->
# (accept, dt_next, state_next), where dt_used is the horizon-clipped step the
# solver actually took and dt_prev is the unclipped carried step size. All
# present numeric fields and controller state are pytree data leaves, so
# changing explicit tolerance values or coefficients never recompiles. An
# omitted tolerance or dt_min is None and therefore a different pytree
# structure.

SAFETY, MIN_FACTOR, MAX_FACTOR = 0.9, 0.2, 5.0
FLOAT32_RTOL, FLOAT32_ATOL = 1e-4, 1e-6
FLOAT64_RTOL, FLOAT64_ATOL = 1e-7, 1e-9
DT_MIN_EPS_FACTOR = 10.0


def _resolve_tolerances(rtol, atol, dtype):
    if jnp.finfo(dtype).bits <= 32:
        default_rtol, default_atol = FLOAT32_RTOL, FLOAT32_ATOL
    else:
        default_rtol, default_atol = FLOAT64_RTOL, FLOAT64_ATOL
    resolved_rtol = default_rtol if rtol is None else rtol
    resolved_atol = default_atol if atol is None else atol
    return jnp.asarray(resolved_rtol, dtype), jnp.asarray(resolved_atol, dtype)


def _resolve_dt_min(dt_min, dtype, time_scale):
    if dt_min is None:
        resolved = (
            DT_MIN_EPS_FACTOR
            * jnp.finfo(dtype).eps
            * jnp.maximum(1.0, jnp.abs(time_scale))
        )
    else:
        resolved = dt_min
    return jnp.asarray(resolved, dtype)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ConstantStepSize:
    """Accept every step and keep the carried step size unchanged."""

    uses_error_estimate = False

    def init(self, x_0):
        return ()

    def adapt(self, x_0, x_1, err, dt_used, dt_prev, order, state, time_scale=1.0):
        return jnp.asarray(True), dt_prev, state


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class IController:
    """Integral step-size controller with max-norm error.

    Accept iff ``E = max(|err| / (atol + rtol * max(|x_0|, |x_1|))) <= 1``
    (forced accept once the step reaches ``dt_min``), and propose
    ``dt_next = dt_used * clip(safety * E**(-1/order), factor_min, factor_max)``
    clipped to ``[dt_min, dt_max]``. This is the classic integral controller —
    equal to diffrax's ``PIDController`` at its default coefficients
    (p_coeff=0, d_coeff=0); there is no proportional term, hence the name.

    If omitted, ``rtol``/``atol`` default to ``1e-4``/``1e-6`` for float32
    states and ``1e-7``/``1e-9`` for float64 states. Explicit tolerances are
    cast to the state dtype. ``dt_min`` defaults to ten machine epsilons in the
    time dtype, scaled by ``max(1, |t_1|)``.
    """

    rtol: float | None = None
    atol: float | None = None
    dt_min: float | None = None
    dt_max: float = float("inf")
    safety: float = SAFETY
    factor_min: float = MIN_FACTOR
    factor_max: float = MAX_FACTOR

    uses_error_estimate = True

    def init(self, x_0):
        return ()

    def adapt(self, x_0, x_1, err, dt_used, dt_prev, order, state, time_scale=1.0):
        dtype = jnp.result_type(x_0, float)
        rtol, atol = _resolve_tolerances(self.rtol, self.atol, dtype)
        dt_min = _resolve_dt_min(
            self.dt_min, jnp.result_type(dt_used), jnp.asarray(time_scale)
        )
        dt_max = jnp.asarray(self.dt_max, jnp.result_type(dt_used))
        safety = jnp.asarray(self.safety, dtype)
        factor_min = jnp.asarray(self.factor_min, dtype)
        factor_max = jnp.asarray(self.factor_max, dtype)
        scale = atol + rtol * jnp.maximum(jnp.abs(x_0), jnp.abs(x_1))
        # The controller is wrapped in stop_gradient: accept/reject is a
        # non-differentiable branch either way, the gradient of E**(-1/order)
        # blows up at the exact-zero error of a flat-start policy, and the
        # d(dt)/dtheta term only slides sample points along the visited
        # trajectory -- irrelevant to a residual that must vanish at every
        # state. The states themselves remain fully differentiable through
        # the RK stages.
        error_ratio = jax.lax.stop_gradient(jnp.max(jnp.abs(err) / scale))
        accept = (error_ratio <= 1.0) | (dt_used <= dt_min)
        error_floor = jnp.asarray(jnp.finfo(dtype).eps, dtype)
        factor = jnp.clip(
            safety * jnp.maximum(error_ratio, error_floor) ** (-1.0 / order),
            factor_min,
            factor_max,
        )
        dt_next = jnp.clip(jax.lax.stop_gradient(dt_used) * factor, dt_min, dt_max)
        return accept, dt_next, state


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PIController:
    """Proportional-integral step-size controller with max-norm error.

    In addition to the current scaled error ``E``, this controller carries the
    previous accepted step's error ``E_prev`` and proposes

    ``dt_next = dt_used * clip(safety * E**(-(p_coeff+i_coeff)/order)``
    ``* E_prev**(p_coeff/order), factor_min, factor_max)``.

    ``E_prev`` starts at one and changes only after an accepted step. The
    defaults ``p_coeff=0.4`` and ``i_coeff=0.3`` damp step-size oscillations on
    harder problems; ``p_coeff=0, i_coeff=1`` reproduces :class:`IController`.
    The error ratios and step-size update are stop-gradiented, while states
    remain fully differentiable through the solver stages. If omitted,
    ``rtol``/``atol`` default to ``1e-4``/``1e-6`` for float32 states and
    ``1e-7``/``1e-9`` for float64 states. ``dt_min`` defaults to ten machine
    epsilons in the time dtype, scaled by ``max(1, |t_1|)``.
    """

    rtol: float | None = None
    atol: float | None = None
    p_coeff: float = 0.4
    i_coeff: float = 0.3
    dt_min: float | None = None
    dt_max: float = float("inf")
    safety: float = SAFETY
    factor_min: float = MIN_FACTOR
    factor_max: float = MAX_FACTOR

    uses_error_estimate = True

    def init(self, x_0):
        return jnp.asarray(1.0, jnp.result_type(x_0, float))

    def adapt(self, x_0, x_1, err, dt_used, dt_prev, order, state, time_scale=1.0):
        dtype = jnp.result_type(x_0, float)
        rtol, atol = _resolve_tolerances(self.rtol, self.atol, dtype)
        dt_min = _resolve_dt_min(
            self.dt_min, jnp.result_type(dt_used), jnp.asarray(time_scale)
        )
        dt_max = jnp.asarray(self.dt_max, jnp.result_type(dt_used))
        safety = jnp.asarray(self.safety, dtype)
        factor_min = jnp.asarray(self.factor_min, dtype)
        factor_max = jnp.asarray(self.factor_max, dtype)
        p_coeff = jnp.asarray(self.p_coeff, dtype)
        i_coeff = jnp.asarray(self.i_coeff, dtype)
        scale = atol + rtol * jnp.maximum(jnp.abs(x_0), jnp.abs(x_1))
        error_ratio = jax.lax.stop_gradient(jnp.max(jnp.abs(err) / scale))
        accept = (error_ratio <= 1.0) | (dt_used <= dt_min)
        error_floor = jnp.asarray(jnp.finfo(dtype).eps, dtype)
        safe_error_ratio = jnp.maximum(error_ratio, error_floor)
        safe_previous_error_ratio = jnp.maximum(state, error_floor)
        factor = jnp.clip(
            safety
            * safe_error_ratio ** (-(p_coeff + i_coeff) / order)
            * safe_previous_error_ratio ** (p_coeff / order),
            factor_min,
            factor_max,
        )
        dt_next = jnp.clip(jax.lax.stop_gradient(dt_used) * factor, dt_min, dt_max)
        state_next = jnp.where(accept, error_ratio, state)
        return accept, dt_next, state_next
