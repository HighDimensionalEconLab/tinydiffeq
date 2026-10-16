"""Pure numerics for the collocation BVP solver, ported from scipy _bvp.py."""

import jax
import jax.numpy as jnp

# Constants from scipy.integrate._bvp (v1.18.0), same names where they exist.
MAX_ITERATION = 10
MAX_NEWTON_ITERATIONS = 8
MAX_NJEV = 4
SIGMA = 0.2
TAU = 0.5
N_TRIAL = 4
TOL_R_FACTOR = 2.0 / 3.0 * 5e-2
REFINE_FACTOR = 100.0
TOL_FLOOR_FACTOR = 100.0
# 5-point Lobatto quadrature on [0, 1]: interior nodes 0.5 +- sqrt(3/7)/2.
LOBATTO_OFFSET = 0.5 * (3.0 / 7.0) ** 0.5
LOBATTO_WEIGHT_MIDDLE = 32.0 / 45.0
LOBATTO_WEIGHT_SIDE = 49.0 / 90.0

RUNNING = -1
STATUS_CONVERGED = 0
STATUS_MAX_NODES = 1
STATUS_SINGULAR = 2
STATUS_BC_TOL = 3


def collocation_jacobian_blocks(
    h,
    df_dy,
    df_dy_middle,
    df_dz,
    df_dz_middle,
    dbc_dya,
    dbc_dyb,
    dbc_dz,
):
    # The six BABD blocks of the global Jacobian, scipy's formulas. On an
    # inactive padded interval h == 0 exactly, so the staircase blocks reduce
    # to -I and +I: a copy chain that propagates the last active node to the
    # static last column block where dbc_dyb sits. The padded system is an
    # exact algebraic embedding of the active one.
    n = df_dy.shape[-1]
    dtype = df_dy.dtype
    hb = h[:, None, None]
    eye = jnp.eye(n, dtype=dtype)
    dphi_dy_0 = (
        -eye
        - hb / 6.0 * (df_dy[:-1] + 2.0 * df_dy_middle)
        - hb**2 / 12.0 * (df_dy_middle @ df_dy[:-1])
    )
    dphi_dy_1 = (
        eye
        - hb / 6.0 * (df_dy[1:] + 2.0 * df_dy_middle)
        + hb**2 / 12.0 * (df_dy_middle @ df_dy[1:])
    )
    if df_dz is None:
        dphi_dz = None
    else:
        correction = df_dy_middle @ (df_dz[:-1] - df_dz[1:])
        dphi_dz = (
            -hb
            / 6.0
            * (df_dz[:-1] + df_dz[1:] + 4.0 * (df_dz_middle + 0.125 * hb * correction))
        )
    return dict(
        block_left=dphi_dy_0,
        block_right=dphi_dy_1,
        block_border=dphi_dz,
        boundary_first=dbc_dya,
        boundary_last=dbc_dyb,
        boundary_border=dbc_dz,
    )


def pad_tail(values, num_active):
    # Repeat row num_active - 1 over the inactive tail, bitwise. The tail
    # invariant lets the boundary condition read the static last row and
    # keeps duplicate-knot interpolation exact at the right endpoint.
    last = jax.lax.dynamic_index_in_dim(values, num_active - 1, 0, keepdims=False)
    mask = jnp.arange(values.shape[0]) < num_active
    return jnp.where(mask.reshape((-1,) + (1,) * (values.ndim - 1)), values, last)


def hermite_pair(tau, h, h_safe, y_left, y_right, f_left, f_right):
    # Cubic Hermite value and derivative at fixed local coordinate tau —
    # the same C1 spline scipy builds in create_spline. The derivative
    # divides by h_safe so inactive zero-width intervals stay finite (their
    # rows are masked by the caller).
    h00 = (1.0 + 2.0 * tau) * (1.0 - tau) ** 2
    h10 = tau * (1.0 - tau) ** 2
    h01 = tau**2 * (3.0 - 2.0 * tau)
    h11 = tau**2 * (tau - 1.0)
    d00 = 6.0 * tau**2 - 6.0 * tau
    d10 = 3.0 * tau**2 - 4.0 * tau + 1.0
    d11 = 3.0 * tau**2 - 2.0 * tau
    hb = h[:, None]
    value = h00 * y_left + h10 * hb * f_left + h01 * y_right + h11 * hb * f_right
    # d01 = -d00 exactly, so equal endpoint rows cancel bitwise on the tail.
    slope = d00 * (y_left - y_right) / h_safe[:, None]
    return value, slope + d10 * f_left + d11 * f_right


def refined_mesh(t, num_nodes, insert_1, insert_2, Y=None, f=None, h=None):
    # Refined mesh (and optional state table) via prefix-sum offsets, padded
    # back to static length so the active prefix matches scipy's modify_mesh.
    max_nodes = t.shape[0]
    active = jnp.arange(max_nodes - 1) < num_nodes - 1
    insert_1 = insert_1 & active
    insert_2 = insert_2 & active
    added = insert_1.astype(jnp.int32) + 2 * insert_2.astype(jnp.int32)
    prefix = jnp.concatenate([jnp.zeros(1, jnp.int32), jnp.cumsum(added)])
    num_new = num_nodes + prefix[-1]
    idx_keep = jnp.arange(max_nodes, dtype=jnp.int32) + prefix
    idx_1 = jnp.where(insert_1, idx_keep[:-1] + 1, max_nodes)
    idx_2l = jnp.where(insert_2, idx_keep[:-1] + 1, max_nodes)
    idx_2r = jnp.where(insert_2, idx_keep[:-1] + 2, max_nodes)
    t_new = jnp.full(max_nodes, t[-1], dtype=t.dtype)
    t_new = t_new.at[idx_keep].set(t, mode="drop")
    t_new = t_new.at[idx_1].set(0.5 * (t[:-1] + t[1:]), mode="drop")
    t_new = t_new.at[idx_2l].set((2.0 * t[:-1] + t[1:]) / 3.0, mode="drop")
    t_new = t_new.at[idx_2r].set((t[:-1] + 2.0 * t[1:]) / 3.0, mode="drop")
    t_new = jnp.where(jnp.arange(max_nodes) < num_new, t_new, t[-1])
    if Y is None:
        return t_new, num_new
    hb = h[:, None]
    y_mid = 0.5 * (Y[:-1] + Y[1:]) + 0.125 * hb * (f[:-1] - f[1:])
    y_left = (20.0 * Y[:-1] + 7.0 * Y[1:] + hb * (4.0 * f[:-1] - 2.0 * f[1:])) / 27.0
    y_right = (7.0 * Y[:-1] + 20.0 * Y[1:] + hb * (2.0 * f[:-1] - 4.0 * f[1:])) / 27.0
    y_new = jnp.broadcast_to(Y[-1:], Y.shape)
    y_new = y_new.at[idx_keep].set(Y, mode="drop")
    y_new = y_new.at[idx_1].set(y_mid, mode="drop")
    y_new = y_new.at[idx_2l].set(y_left, mode="drop")
    y_new = y_new.at[idx_2r].set(y_right, mode="drop")
    y_new = pad_tail(y_new, num_new)
    return t_new, y_new, num_new
