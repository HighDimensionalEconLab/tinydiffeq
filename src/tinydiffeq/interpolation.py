import jax
import jax.numpy as jnp

from tinydiffeq._rodas5p import rodas_dense_value

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

        # Each leaf keeps its own dtype. This matters for mixed-precision aux
        # pytrees: a float64 time grid must not widen a float32 output leaf.
        s_leaf = s.astype(xs.dtype)
        width_leaf = width_safe.astype(xs.dtype)
        s_, w_, deg_ = bc(s_leaf), bc(width_leaf), bc(degenerate)
        h_00 = (1.0 + 2.0 * s_) * (1.0 - s_) ** 2
        h_10 = s_ * (1.0 - s_) ** 2
        h_01 = s_**2 * (3.0 - 2.0 * s_)
        h_11 = s_**2 * (s_ - 1.0)
        value = (
            h_00 * x_left + h_10 * w_ * f_left + h_01 * x_right + h_11 * w_ * f_right
        )
        return jnp.where(deg_, x_left, value)

    return jax.tree.map(interpolate_leaf, knot_xs, knot_fs)


def rodas_interpolate(ts_query, knot_ts, knot_xs, interval_coefficients):
    """Evaluate the Rodas5P continuous extension over raw attempt rows.

    ``interval_coefficients`` contains one three-pytree coefficient tuple per
    attempted step. A rejected attempt repeats its left knot, so selecting the
    row immediately after the last duplicate selects the eventual accepted
    interval without compacting or sorting the bounded scan output.
    """
    n = knot_ts.shape[0]
    idx = jnp.clip(jnp.searchsorted(knot_ts, ts_query, side="right") - 1, 0, n - 2)
    t_left, t_right = knot_ts[idx], knot_ts[idx + 1]
    width = t_right - t_left
    degenerate = width <= 0.0
    width_safe = jnp.where(degenerate, 1.0, width)
    theta = jnp.clip((ts_query - t_left) / width_safe, 0.0, 1.0)
    left = jax.tree.map(lambda values: values[idx], knot_xs)
    right = jax.tree.map(lambda values: values[idx + 1], knot_xs)
    dense = tuple(
        jax.tree.map(lambda values: values[idx], coefficient)
        for coefficient in interval_coefficients
    )

    def cast_theta(leaf):
        extra = leaf.ndim - theta.ndim
        return theta.astype(leaf.dtype).reshape(theta.shape + (1,) * extra)

    theta_tree = jax.tree.map(cast_theta, left)
    value = rodas_dense_value(theta_tree, left, right, dense)

    def select(left_leaf, value_leaf):
        extra = left_leaf.ndim - degenerate.ndim
        mask = degenerate.reshape(degenerate.shape + (1,) * extra)
        return jnp.where(mask, left_leaf, value_leaf)

    return jax.tree.map(select, left, value)


def hermite_interval_interpolate(
    ts_query,
    knot_ts,
    knot_xs,
    interval_left_fs,
    interval_right_fs,
):
    """Cubic Hermite output with derivatives stored per attempted interval."""
    n = knot_ts.shape[0]
    idx = jnp.clip(jnp.searchsorted(knot_ts, ts_query, side="right") - 1, 0, n - 2)
    t_left, t_right = knot_ts[idx], knot_ts[idx + 1]
    width = t_right - t_left
    degenerate = width <= 0.0
    width_safe = jnp.where(degenerate, 1.0, width)
    s = jnp.clip((ts_query - t_left) / width_safe, 0.0, 1.0)

    def interpolate_leaf(xs, left_fs, right_fs):
        x_left, x_right = xs[idx], xs[idx + 1]
        f_left, f_right = left_fs[idx], right_fs[idx]
        extra = xs.ndim - 1

        def broadcast(value):
            return value.reshape(value.shape + (1,) * extra)

        s_leaf = broadcast(s.astype(xs.dtype))
        width_leaf = broadcast(width_safe.astype(xs.dtype))
        mask = broadcast(degenerate)
        h_00 = (1.0 + 2.0 * s_leaf) * (1.0 - s_leaf) ** 2
        h_10 = s_leaf * (1.0 - s_leaf) ** 2
        h_01 = s_leaf**2 * (3.0 - 2.0 * s_leaf)
        h_11 = s_leaf**2 * (s_leaf - 1.0)
        value = (
            h_00 * x_left
            + h_10 * width_leaf * f_left
            + h_01 * x_right
            + h_11 * width_leaf * f_right
        )
        return jnp.where(mask, x_left, value)

    return jax.tree.map(
        interpolate_leaf,
        knot_xs,
        interval_left_fs,
        interval_right_fs,
    )
