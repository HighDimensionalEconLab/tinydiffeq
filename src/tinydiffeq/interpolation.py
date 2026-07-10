import jax
import jax.numpy as jnp

# Cubic Hermite interpolation over the raw attempt rows of the bounded scan.
# The knot times are nondecreasing with exact duplicates (rejected attempts
# and the post-horizon frozen tail repeat the previous row), so no compaction
# is needed: searchsorted(side="right") - 1 lands on the LAST duplicate at or
# before each query, giving a positive-width bracket everywhere except the
# frozen tail. Degenerate (zero-width) brackets return the left knot -- flat
# extrapolation beyond the reached time when the budget was exhausted. The
# bracketing indices are integer and non-differentiable, consistent with the
# stop-gradiented step controller (the sliding-knot d(dt)/dtheta term is
# deliberately excluded); values differentiate fully through the left/right
# states and derivatives.


def hermite_interpolate(ts_query, knot_ts, knot_xs, knot_fs):
    """Cubic Hermite interpolation of ``(knot_ts, knot_xs, knot_fs)`` at
    ``ts_query``, where ``knot_fs`` holds the time derivatives at the knots.

    ``knot_ts`` must be nondecreasing; duplicate knots are allowed and
    queries falling on a zero-width bracket return the left knot value.
    Queries outside the knot span clamp to the boundary knot values (flat
    extrapolation) rather than evaluating the cubic outside its bracket.
    """
    n = knot_ts.shape[0]
    idx = jnp.clip(jnp.searchsorted(knot_ts, ts_query, side="right") - 1, 0, n - 2)
    t_left, t_right = knot_ts[idx], knot_ts[idx + 1]
    width = t_right - t_left
    degenerate = width <= 0.0
    # double-where: divide by the safe width BEFORE branching so neither the
    # primal nor its jvp/vjp ever sees a 0/0.
    width_safe = jnp.where(degenerate, 1.0, width)
    s = jnp.clip((ts_query - t_left) / width_safe, 0.0, 1.0)

    def interpolate_leaf(xs, fs):
        x_left, x_right = xs[idx], xs[idx + 1]
        f_left, f_right = fs[idx], fs[idx + 1]
        extra = xs.ndim - 1

        def bc(a):
            return a.reshape(a.shape + (1,) * extra)

        s_, w_, deg_ = bc(s), bc(width_safe), bc(degenerate)
        h_00 = (1.0 + 2.0 * s_) * (1.0 - s_) ** 2
        h_10 = s_ * (1.0 - s_) ** 2
        h_01 = s_**2 * (3.0 - 2.0 * s_)
        h_11 = s_**2 * (s_ - 1.0)
        value = (
            h_00 * x_left + h_10 * w_ * f_left + h_01 * x_right + h_11 * w_ * f_right
        )
        return jnp.where(deg_, x_left, value)

    return jax.tree.map(interpolate_leaf, knot_xs, knot_fs)
