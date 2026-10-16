"""Bordered almost block diagonal (BABD) linear systems.

A BABD system couples chain unknowns ``x_0 .. x_E`` (size ``n`` each) and
border unknowns ``z`` (size ``k``) through ``E`` staircase equations, each
involving two adjacent chain unknowns, plus ``n + k`` boundary equations
involving the first and last chain unknowns:

    block_left[i] x_i + block_right[i] x_{i+1} + block_border[i] z = f_i
    boundary_first x_0 + boundary_last x_E + boundary_border z = g

The block container is a dict with those six keys (``block_border`` and
``boundary_border`` are ``None`` when ``k == 0``). Unknowns and equations are
flattened in that order. This is the structure of collocation and multiple
shooting discretizations of two-point boundary value problems.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy import linalg as jsp_linalg


def babd_dimensions(blocks):
    chain, n = blocks["block_left"].shape[:2]
    k = 0 if blocks["block_border"] is None else blocks["block_border"].shape[-1]
    return chain, n, k


def babd_indices(n, m, k):
    # Row/column indices of the six blocks in the dense embedding, matching
    # scipy's compute_jac_indices with node-leading value ravels. The combined
    # index set has no duplicates, so a scatter set matches sparse summing.
    i_col = np.repeat(np.arange((m - 1) * n), n)
    j_col = np.tile(np.arange(n), n * (m - 1)) + np.repeat(np.arange(m - 1) * n, n**2)

    i_bc = np.repeat(np.arange((m - 1) * n, m * n + k), n)
    j_bc = np.tile(np.arange(n), n + k)

    i_p_col = np.repeat(np.arange((m - 1) * n), k)
    j_p_col = np.tile(np.arange(m * n, m * n + k), (m - 1) * n)

    i_p_bc = np.repeat(np.arange((m - 1) * n, m * n + k), k)
    j_p_bc = np.tile(np.arange(m * n, m * n + k), n + k)

    i = np.hstack((i_col, i_col, i_bc, i_bc, i_p_col, i_p_bc))
    j = np.hstack((j_col, j_col + n, j_bc, j_bc + (m - 1) * n, j_p_col, j_p_bc))
    return i, j


def babd_dense(blocks):
    # Scatter the blocks into the dense square matrix.
    chain, n, k = babd_dimensions(blocks)
    dtype = blocks["block_left"].dtype
    i_jac, j_jac = babd_indices(n, chain + 1, k)
    values = [
        blocks["block_left"].reshape(-1),
        blocks["block_right"].reshape(-1),
        blocks["boundary_first"].reshape(-1),
        blocks["boundary_last"].reshape(-1),
    ]
    if k > 0:
        values.extend(
            [blocks["block_border"].reshape(-1), blocks["boundary_border"].reshape(-1)]
        )
    size = (chain + 1) * n + k
    flat = jnp.concatenate(values)
    return jnp.zeros((size, size), dtype).at[i_jac, j_jac].set(flat)


def babd_matvec(blocks, u):
    # Structured product with a flat vector ordered (x_0 .. x_E, z).
    chain, n, k = babd_dimensions(blocks)
    x = u[: (chain + 1) * n].reshape(chain + 1, n)
    z = u[(chain + 1) * n :]
    staircase = jnp.einsum("eij,ej->ei", blocks["block_left"], x[:-1]) + jnp.einsum(
        "eij,ej->ei", blocks["block_right"], x[1:]
    )
    boundary = blocks["boundary_first"] @ x[0] + blocks["boundary_last"] @ x[-1]
    if k > 0:
        staircase = staircase + jnp.einsum("eik,k->ei", blocks["block_border"], z)
        boundary = boundary + blocks["boundary_border"] @ z
    return jnp.concatenate([staircase.reshape(-1), boundary])


def structured_qr_factor(blocks):
    # Cyclic reduction with orthogonal eliminations (the level-parallel
    # ordering of Wright's structured orthogonal factorization): each level
    # pairs adjacent staircase equations and eliminates every shared unknown
    # at once through one batched complete QR of the stacked (2n, n) middle
    # columns, so a chain of length E costs ~log2(E) batched calls instead of
    # E sequential ones. Orthogonal eliminations keep every stored row
    # bounded by the original row norms, so saddle-path dichotomies cannot
    # overflow the way naive condensation does. What remains couples
    # (x_0, x_E, z) in one dense (2n + k)-square system.
    chain, n, k = babd_dimensions(blocks)
    dtype = blocks["block_left"].dtype
    a = blocks["block_left"]
    b = blocks["block_right"]
    c = blocks["block_border"]
    if c is None:
        c = jnp.zeros((chain, n, 0), dtype)
    boundary_border = blocks["boundary_border"]
    if boundary_border is None:
        boundary_border = jnp.zeros((blocks["boundary_first"].shape[0], 0), dtype)

    levels = []
    ok = jnp.asarray(True)
    length = chain
    while length > 1:
        pairs = length // 2
        odd = length - 2 * pairs
        a_even = a[0 : 2 * pairs : 2]
        b_even = b[0 : 2 * pairs : 2]
        c_even = c[0 : 2 * pairs : 2]
        a_odd = a[1 : 2 * pairs : 2]
        b_odd = b[1 : 2 * pairs : 2]
        c_odd = c[1 : 2 * pairs : 2]
        q, r_full = jnp.linalg.qr(
            jnp.concatenate([b_even, a_odd], axis=1), mode="complete"
        )
        q_t = jnp.swapaxes(q, -1, -2)
        zeros_pair = jnp.zeros_like(a_even)
        left_pair = q_t @ jnp.concatenate([a_even, zeros_pair], axis=1)
        right_pair = q_t @ jnp.concatenate([zeros_pair, b_odd], axis=1)
        border_pair = q_t @ jnp.concatenate([c_even, c_odd], axis=1)
        r = r_full[:, :n]
        diag_r = jnp.diagonal(r, axis1=-2, axis2=-1)
        ok = (
            ok
            & jnp.all(jnp.isfinite(diag_r))
            & (jnp.min(jnp.abs(diag_r), initial=jnp.inf) > 0.0)
        )
        levels.append(
            dict(
                q=q,
                r=r,
                left=left_pair[:, :n],
                right=right_pair[:, :n],
                border=border_pair[:, :n],
            )
        )
        a_next = left_pair[:, n:]
        b_next = right_pair[:, n:]
        c_next = border_pair[:, n:]
        if odd:
            a_next = jnp.concatenate([a_next, a[-1:]], axis=0)
            b_next = jnp.concatenate([b_next, b[-1:]], axis=0)
            c_next = jnp.concatenate([c_next, c[-1:]], axis=0)
        a, b, c = a_next, b_next, c_next
        length = pairs + odd

    final = jnp.concatenate(
        [
            jnp.concatenate([a[0], b[0], c[0]], axis=1),
            jnp.concatenate(
                [blocks["boundary_first"], blocks["boundary_last"], boundary_border],
                axis=1,
            ),
        ],
        axis=0,
    )
    lu, pivots = jsp_linalg.lu_factor(final, check_finite=False)
    ok = jax.lax.stop_gradient(
        ok & jnp.all(jnp.isfinite(lu)) & jnp.all(jnp.abs(jnp.diag(lu)) > 0.0)
    )
    # An unusable factor becomes the identity so batched lanes stay finite.
    size = final.shape[0]
    state = dict(
        levels=tuple(
            dict(
                q=jnp.where(ok, level["q"], jnp.eye(2 * n, dtype=dtype)),
                r=jnp.where(ok, level["r"], jnp.eye(n, dtype=dtype)),
                left=jnp.where(ok, level["left"], jnp.zeros_like(level["left"])),
                right=jnp.where(ok, level["right"], jnp.zeros_like(level["right"])),
                border=jnp.where(ok, level["border"], jnp.zeros_like(level["border"])),
            )
            for level in levels
        ),
        template=jnp.zeros(n, dtype),
        final_lu=jnp.where(ok, lu, jnp.eye(size, dtype=dtype)),
        final_pivots=jnp.where(ok, pivots, jnp.arange(size, dtype=pivots.dtype)),
    )
    return state, ok


def structured_qr_level_sizes(state, chain):
    sizes = []
    length = chain
    for level in state["levels"]:
        pairs = level["q"].shape[0]
        odd = length - 2 * pairs
        sizes.append((pairs, odd))
        length = pairs + odd
    return sizes


def structured_qr_solve(state, rhs):
    n = state["template"].shape[0]
    k = state["final_lu"].shape[0] - 2 * n
    chain = (rhs.shape[0] - k) // n - 1
    f = rhs[: chain * n].reshape(chain, n)
    boundary_rhs = rhs[chain * n :]
    sizes = structured_qr_level_sizes(state, chain)

    tops = []
    for level, (pairs, odd) in zip(state["levels"], sizes, strict=True):
        stacked = jnp.concatenate([f[0 : 2 * pairs : 2], f[1 : 2 * pairs : 2]], axis=1)
        transformed = jnp.einsum("pji,pj->pi", level["q"], stacked)
        tops.append(transformed[:, :n])
        f_next = transformed[:, n:]
        if odd:
            f_next = jnp.concatenate([f_next, f[-1:]], axis=0)
        f = f_next

    boundary = jsp_linalg.lu_solve(
        (state["final_lu"], state["final_pivots"]),
        jnp.concatenate([f[0], boundary_rhs]),
        trans=0,
        check_finite=False,
    )
    x_first, x_last, z = boundary[:n], boundary[n : 2 * n], boundary[2 * n :]

    x_nodes = jnp.stack([x_first, x_last], axis=0)
    for level, top, (pairs, _) in reversed(
        list(zip(state["levels"], tops, sizes, strict=True))
    ):
        rhs_mid = (
            top
            - jnp.einsum("pij,pj->pi", level["left"], x_nodes[:pairs])
            - jnp.einsum("pij,pj->pi", level["right"], x_nodes[1 : pairs + 1])
            - jnp.einsum("pik,k->pi", level["border"], z)
        )
        x_mid = jsp_linalg.solve_triangular(
            level["r"], rhs_mid[..., None], lower=False, check_finite=False
        )[..., 0]
        interleaved = jnp.stack([x_nodes[:pairs], x_mid], axis=1).reshape(2 * pairs, n)
        x_nodes = jnp.concatenate([interleaved, x_nodes[pairs:]], axis=0)
    return jnp.concatenate([x_nodes.reshape(-1), z])


def structured_qr_transpose_solve(state, rhs):
    # J = QU from the factorization, so J^T y = b is the lower-triangular
    # sweep U^T v = b down the levels followed by y = Q v back up.
    n = state["template"].shape[0]
    dtype = state["template"].dtype
    k = state["final_lu"].shape[0] - 2 * n
    nodes = (rhs.shape[0] - k) // n
    chain = nodes - 1
    b_nodes = rhs[: nodes * n].reshape(nodes, n)
    b_z = rhs[nodes * n :]
    sizes = structured_qr_level_sizes(state, chain)

    sum_border = jnp.zeros(k, dtype)
    adjoints = []
    for level, (pairs, _) in zip(state["levels"], sizes, strict=True):
        v = jsp_linalg.solve_triangular(
            level["r"],
            b_nodes[1 : 2 * pairs : 2][..., None],
            lower=False,
            trans=1,
            check_finite=False,
        )[..., 0]
        adjoints.append(v)
        sum_border = sum_border + jnp.einsum("pik,pi->k", level["border"], v)
        contrib_left = jnp.einsum("pij,pi->pj", level["left"], v)
        contrib_right = jnp.einsum("pij,pi->pj", level["right"], v)
        reduced = jnp.concatenate(
            [b_nodes[0 : 2 * pairs : 2], b_nodes[2 * pairs :]], axis=0
        )
        tail = reduced.shape[0] - pairs
        reduced = reduced - jnp.concatenate(
            [contrib_left, jnp.zeros((tail, n), dtype)], axis=0
        )
        reduced = reduced - jnp.concatenate(
            [jnp.zeros((1, n), dtype), contrib_right, jnp.zeros((tail - 1, n), dtype)],
            axis=0,
        )
        b_nodes = reduced

    boundary = jsp_linalg.lu_solve(
        (state["final_lu"], state["final_pivots"]),
        jnp.concatenate([b_nodes[0], b_nodes[1], b_z - sum_border]),
        trans=1,
        check_finite=False,
    )

    y_rows = boundary[:n][None]
    for level, v, (pairs, _) in reversed(
        list(zip(state["levels"], adjoints, sizes, strict=True))
    ):
        pair = jnp.einsum(
            "pij,pj->pi", level["q"], jnp.concatenate([v, y_rows[:pairs]], axis=1)
        )
        interleaved = jnp.stack([pair[:, :n], pair[:, n:]], axis=1).reshape(
            2 * pairs, n
        )
        y_rows = jnp.concatenate([interleaved, y_rows[pairs:]], axis=0)
    return jnp.concatenate([y_rows.reshape(-1), boundary[n:]])
