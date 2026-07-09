from dataclasses import dataclass

import jax
import jax.numpy as jnp

# Controllers decide accept/reject and the next step size from the embedded
# error estimate. Contract: adapt(x0, x1, err, dt_used, dt_prev, order) ->
# (accept, dt_next), where dt_used is the horizon-clipped step the solver
# actually took and dt_prev is the unclipped carried step size. All numeric
# fields are pytree data leaves, so changing tolerances never recompiles.

SAFETY, MIN_FACTOR, MAX_FACTOR = 0.9, 0.2, 5.0


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ConstantStepSize:
    """Accept every step and keep the carried step size unchanged."""

    uses_error_estimate = False

    def adapt(self, x0, x1, err, dt_used, dt_prev, order):
        return jnp.asarray(True), dt_prev


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class IController:
    """Integral step-size controller with max-norm error.

    Accept iff ``E = max(|err| / (atol + rtol * max(|x0|, |x1|))) <= 1``
    (forced accept once the step reaches ``dtmin``), and propose
    ``dt_next = dt_used * clip(safety * E**(-1/order), factormin, factormax)``
    clipped to ``[dtmin, dtmax]``. This is the classic integral controller —
    equal to diffrax's ``PIDController`` at its default coefficients
    (pcoeff=0, dcoeff=0); there is no proportional term, hence the name.
    """

    rtol: float
    atol: float
    dtmin: float = 1e-10
    dtmax: float = float("inf")
    safety: float = SAFETY
    factormin: float = MIN_FACTOR
    factormax: float = MAX_FACTOR

    uses_error_estimate = True

    def adapt(self, x0, x1, err, dt_used, dt_prev, order):
        dtype = jnp.result_type(x0, float)
        rtol = jnp.asarray(self.rtol, dtype)
        atol = jnp.asarray(self.atol, dtype)
        dtmin = jnp.asarray(self.dtmin, jnp.result_type(dt_used))
        dtmax = jnp.asarray(self.dtmax, jnp.result_type(dt_used))
        safety = jnp.asarray(self.safety, dtype)
        factormin = jnp.asarray(self.factormin, dtype)
        factormax = jnp.asarray(self.factormax, dtype)
        scale = atol + rtol * jnp.maximum(jnp.abs(x0), jnp.abs(x1))
        # The controller is wrapped in stop_gradient: accept/reject is a
        # non-differentiable branch either way, the gradient of E**(-1/order)
        # blows up at the exact-zero error of a flat-start policy, and the
        # d(dt)/dtheta term only slides sample points along the visited
        # trajectory -- irrelevant to a residual that must vanish at every
        # state. The states themselves remain fully differentiable through
        # the RK stages.
        E = jax.lax.stop_gradient(jnp.max(jnp.abs(err) / scale))
        accept = (E <= 1.0) | (dt_used <= dtmin)
        factor = jnp.clip(
            safety * jnp.maximum(E, 1e-12) ** (-1.0 / order), factormin, factormax
        )
        dt_next = jnp.clip(jax.lax.stop_gradient(dt_used) * factor, dtmin, dtmax)
        return accept, dt_next
