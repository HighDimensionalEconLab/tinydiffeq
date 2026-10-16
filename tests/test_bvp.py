import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from scipy.integrate import solve_bvp as scipy_solve_bvp
from scipy.special import erf

from tinydiffeq import bvp as bvp_module
from tinydiffeq import (
    hermite_derivative,
    hermite_interpolate,
    solve_bvp,
)
from tinydiffeq._bvp_core import refined_mesh
from tinydiffeq.babd import (
    babd_matvec,
    structured_qr_factor,
    structured_qr_solve,
    structured_qr_transpose_solve,
)


def exp_fun(t, y):
    return jnp.array([y[1], y[0]])


def exp_bc(ya, yb):
    return jnp.array([ya[0] - 1.0, yb[0]])


def exp_sol(t):
    return (np.exp(-t) - np.exp(t - 2.0)) / (1.0 - np.exp(-2.0))


def sl_fun(t, y, z):
    return jnp.array([y[1], -(z[0] ** 2) * y[0]])


def sl_bc(ya, yb, z):
    return jnp.array([ya[0], yb[0], ya[1] - z[0]])


def emden_fun(t, y):
    return jnp.array([y[1], -(y[0] ** 5)])


def emden_bc(ya, yb):
    return jnp.array([ya[1], yb[0] - (3.0 / 4.0) ** 0.5])


EMDEN_S = jnp.array([[0.0, 0.0], [0.0, -2.0]])


def emden_sol(t):
    return (1.0 + t**2 / 3.0) ** -0.5


def make_config(fun, bc, max_nodes, n, k, has_singular_term=False):
    node = jnp.zeros(n)
    _, unravel_y = ravel_pytree(node)
    if k > 0:
        z_node = jnp.zeros(k)
        _, unravel_z = ravel_pytree(z_node)
        z_treedef = jax.tree.structure(z_node)
        z_leaf_specs = ((z_node.shape, str(z_node.dtype)),)
    else:
        unravel_z = bvp_module._unravel_empty
        z_treedef = None
        z_leaf_specs = None
    fun_arity = bvp_module.bvp_arity(fun, "fun", "")
    bc_arity = bvp_module.bvp_arity(bc, "bc", "")
    return bvp_module._BVPConfig(
        fun=fun,
        bc=bc,
        fun_arity=fun_arity,
        bc_arity=bc_arity,
        max_nodes=max_nodes,
        n=n,
        k=k,
        has_aux=False,
        fun_jac_ad="auto",
        bc_jac_ad="auto",
        has_singular_term=has_singular_term,
        y_treedef=jax.tree.structure(node),
        y_leaf_specs=((node.shape, str(node.dtype)),),
        z_treedef=z_treedef,
        z_leaf_specs=z_leaf_specs,
        fun_canon=bvp_module.canonicalize_bvp_fun(fun, fun_arity),
        bc_canon=bvp_module.canonicalize_bvp_bc(bc, bc_arity),
        unravel_y=unravel_y,
        unravel_z=unravel_z,
    )


def relative_residual_norm(fun_values, derivative_values):
    residual = (derivative_values - fun_values) / (1.0 + np.abs(fun_values))
    return np.sum(residual**2, axis=1) ** 0.5


def test_modify_mesh_matches_scipy():
    t = jnp.array([0.0, 1.0, 3.0, 9.0, 9.0, 9.0, 9.0])
    insert_1 = jnp.array([True, False, False, False, False, False])
    insert_2 = jnp.array([False, False, True, False, False, False])
    t_new, num_new = refined_mesh(t, jnp.asarray(4, jnp.int32), insert_1, insert_2)
    assert int(num_new) == 7
    assert jnp.array_equal(t_new, jnp.array([0.0, 0.5, 1.0, 3.0, 5.0, 7.0, 9.0]))

    t = jnp.concatenate([jnp.array([-6.0, -3.0, 0.0, 3.0, 6.0]), jnp.full(7, 6.0)])
    insert_1 = jnp.zeros(11, bool).at[1].set(True)
    insert_2 = jnp.zeros(11, bool).at[jnp.array([0, 2, 3])].set(True)
    t_new, num_new = refined_mesh(t, jnp.asarray(5, jnp.int32), insert_1, insert_2)
    assert int(num_new) == 12
    expected = jnp.array(
        [-6.0, -5.0, -4.0, -3.0, -1.5, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    )
    assert jnp.array_equal(t_new, expected)


def test_exponential_no_refinement():
    t = jnp.linspace(0.0, 1.0, 5)
    sol = solve_bvp(exp_fun, exp_bc, t, jnp.zeros((5, 2)), max_nodes=32)
    assert int(sol.status) == 0
    assert bool(sol.ok)
    assert int(sol.num_nodes) == 5
    num = int(sol.num_nodes)

    t_test = jnp.linspace(0.0, 1.0, 100)
    y_test = hermite_interpolate(t_test, sol.t, sol.y, sol.yp)
    np.testing.assert_allclose(
        np.asarray(y_test[:, 0]), exp_sol(np.asarray(t_test)), atol=1e-5
    )
    assert np.all(np.asarray(sol.rms_residuals[: num - 1]) < 1e-3)

    # The returned arrays are exactly the spline's knots and knot derivatives.
    np.testing.assert_allclose(
        np.asarray(hermite_interpolate(sol.t, sol.t, sol.y, sol.yp)),
        np.asarray(sol.y),
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        np.asarray(hermite_derivative(sol.t, sol.t, sol.y, sol.yp)),
        np.asarray(sol.yp),
        rtol=1e-10,
        atol=1e-10,
    )

    yp_test = hermite_derivative(t_test, sol.t, sol.y, sol.yp)
    f_test = jax.vmap(exp_fun, in_axes=(0, 0))(t_test, y_test)
    norm_res = relative_residual_norm(np.asarray(f_test), np.asarray(yp_test))
    assert np.all(norm_res < 1e-3)


def test_exponential_matches_scipy():
    def np_fun(x, y):
        return np.vstack((y[1], y[0]))

    def np_bc(ya, yb):
        return np.array([ya[0] - 1.0, yb[0]])

    def np_fun_jac(x, y):
        df_dy = np.zeros((2, 2, x.shape[0]))
        df_dy[0, 1] = 1.0
        df_dy[1, 0] = 1.0
        return df_dy

    def np_bc_jac(ya, yb):
        return np.array([[1.0, 0.0], [0.0, 0.0]]), np.array([[0.0, 0.0], [1.0, 0.0]])

    reference = scipy_solve_bvp(
        np_fun,
        np_bc,
        np.linspace(0, 1, 5),
        np.zeros((2, 5)),
        fun_jac=np_fun_jac,
        bc_jac=np_bc_jac,
    )
    sol = solve_bvp(
        exp_fun, exp_bc, jnp.linspace(0, 1, 5), jnp.zeros((5, 2)), max_nodes=32
    )
    num = int(sol.num_nodes)
    assert num == reference.x.size
    assert int(sol.num_iterations) == reference.niter
    np.testing.assert_allclose(np.asarray(sol.t[:num]), reference.x, rtol=1e-12)
    np.testing.assert_allclose(
        np.asarray(sol.y[:num]).T, reference.y, rtol=1e-8, atol=1e-14
    )
    np.testing.assert_allclose(
        np.asarray(sol.yp[:num]).T, reference.yp, rtol=1e-8, atol=1e-14
    )
    np.testing.assert_allclose(
        np.asarray(sol.rms_residuals[: num - 1]),
        reference.rms_residuals,
        rtol=1e-6,
        atol=1e-14,
    )


def test_sturm_liouville_unknown_parameter():
    t = jnp.linspace(0.0, jnp.pi, 5)
    sol = solve_bvp(sl_fun, sl_bc, t, jnp.ones((5, 2)), jnp.array([0.5]), max_nodes=32)
    assert int(sol.status) == 0
    num = int(sol.num_nodes)
    assert num < 10
    np.testing.assert_allclose(np.asarray(sol.z), [1.0], rtol=1e-4)
    t_active = np.asarray(sol.t[:num])
    np.testing.assert_allclose(
        np.asarray(sol.y[:num, 0]), np.sin(t_active), rtol=1e-4, atol=1e-4
    )
    assert np.all(np.asarray(sol.rms_residuals[: num - 1]) < 1e-3)

    def np_fun(x, y, p):
        return np.vstack((y[1], -(p[0] ** 2) * y[0]))

    def np_bc(ya, yb, p):
        return np.array([ya[0], yb[0], ya[1] - p[0]])

    def np_fun_jac(x, y, p):
        df_dy = np.zeros((2, 2, x.shape[0]))
        df_dy[0, 1] = 1.0
        df_dy[1, 0] = -(p[0] ** 2)
        df_dp = np.zeros((2, 1, x.shape[0]))
        df_dp[1, 0] = -2.0 * p[0] * y[0]
        return df_dy, df_dp

    def np_bc_jac(ya, yb, p):
        dbc_dya = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
        dbc_dyb = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
        dbc_dp = np.array([[0.0], [0.0], [-1.0]])
        return dbc_dya, dbc_dyb, dbc_dp

    reference = scipy_solve_bvp(
        np_fun,
        np_bc,
        np.linspace(0, np.pi, 5),
        np.ones((2, 5)),
        p=[0.5],
        fun_jac=np_fun_jac,
        bc_jac=np_bc_jac,
    )
    assert num == reference.x.size
    np.testing.assert_allclose(np.asarray(sol.z), reference.p, rtol=1e-9)
    np.testing.assert_allclose(
        np.asarray(sol.y[:num]).T, reference.y, rtol=1e-8, atol=1e-12
    )


def test_singular_term_emden():
    t = jnp.linspace(0.0, 1.0, 10)
    y_0 = jnp.stack([jnp.full(10, (3.0 / 4.0) ** 0.5), jnp.full(10, 1e-4)], axis=1)
    sol = solve_bvp(emden_fun, emden_bc, t, y_0, S=EMDEN_S, max_nodes=32)
    assert int(sol.status) == 0
    assert int(sol.num_nodes) == 10

    t_test = jnp.linspace(0.05, 1.0, 100)
    y_test = hermite_interpolate(t_test, sol.t, sol.y, sol.yp)
    np.testing.assert_allclose(
        np.asarray(y_test[:, 0]), emden_sol(np.asarray(t_test)), atol=1e-5
    )

    yp_test = hermite_derivative(t_test, sol.t, sol.y, sol.yp)
    f_raw = jax.vmap(emden_fun, in_axes=(0, 0))(t_test, y_test)
    f_test = (
        np.asarray(f_raw)
        + np.asarray(y_test) @ np.asarray(EMDEN_S).T / (np.asarray(t_test)[:, None])
    )
    norm_res = relative_residual_norm(f_test, np.asarray(yp_test))
    assert np.all(norm_res < 1e-3)


def test_shock_layer():
    eps = 1e-3

    def shock_fun(t, y):
        return jnp.array(
            [
                y[1],
                -(
                    t * y[1]
                    + eps * jnp.pi**2 * jnp.cos(jnp.pi * t)
                    + jnp.pi * t * jnp.sin(jnp.pi * t)
                )
                / eps,
            ]
        )

    def shock_bc(ya, yb):
        return jnp.array([ya[0] + 2.0, yb[0]])

    t = jnp.linspace(-1.0, 1.0, 5)
    sol = solve_bvp(shock_fun, shock_bc, t, jnp.zeros((5, 2)), max_nodes=128)
    assert int(sol.status) == 0
    num = int(sol.num_nodes)
    assert num < 110
    t_active = np.asarray(sol.t[:num])
    exact = np.cos(np.pi * t_active) + erf(t_active / np.sqrt(2.0 * eps)) / erf(
        1.0 / np.sqrt(2.0 * eps)
    )
    np.testing.assert_allclose(np.asarray(sol.y[:num, 0]), exact, rtol=1e-5, atol=1e-5)


def test_big_problem_with_parameters():
    def big_fun(t, y, z):
        f = jnp.zeros_like(y)
        f = f.at[::2].set(y[1::2])
        f = f.at[1::4].set(-(z[0] ** 2) * y[::4])
        f = f.at[3::4].set(-(z[1] ** 2) * y[2::4])
        return f

    def big_bc(ya, yb, z):
        return jnp.concatenate(
            [ya[::2], yb[::2], jnp.array([ya[1] - z[0], ya[3] - z[1]])]
        )

    t = jnp.linspace(0.0, jnp.pi, 5)
    sol = solve_bvp(
        big_fun, big_bc, t, jnp.ones((5, 60)), jnp.array([0.5, 0.5]), max_nodes=24
    )
    assert int(sol.status) == 0
    num = int(sol.num_nodes)
    np.testing.assert_allclose(np.asarray(sol.z), [1.0, 1.0], rtol=1e-4)
    t_active = np.asarray(sol.t[:num])
    np.testing.assert_allclose(
        np.asarray(sol.y[:num, 0]), np.sin(t_active), rtol=1e-4, atol=1e-4
    )
    np.testing.assert_allclose(
        np.asarray(sol.y[:num, 2]), np.sin(t_active), rtol=1e-4, atol=1e-4
    )


def test_failures():
    sol = solve_bvp(
        exp_fun, exp_bc, jnp.linspace(0, 1, 2), jnp.zeros((2, 2)), tol=1e-5, max_nodes=5
    )
    assert int(sol.status) == 1
    assert not bool(sol.ok)

    def undefined_fun(t, y):
        return jnp.zeros(2)

    def undefined_bc(ya, yb):
        return jnp.array([ya[0], yb[0] - 1.0])

    sol = solve_bvp(
        undefined_fun,
        undefined_bc,
        jnp.linspace(0, 1, 5),
        jnp.zeros((5, 2)),
        max_nodes=8,
    )
    assert int(sol.status) == 2
    assert not bool(sol.ok)
    assert bool(jnp.all(jnp.isfinite(sol.y)))

    # A NaN boundary residual follows scipy's elif chain to status 3 at the
    # tenth iteration instead of looping to the safety bound.
    def nan_bc(ya, yb):
        return jnp.array([ya[0] - 1.0, yb[0] + jnp.where(yb[0] < 0.0, jnp.nan, 0.0)])

    sol = solve_bvp(
        exp_fun, nan_bc, jnp.linspace(0, 1, 5), -jnp.ones((5, 2)), max_nodes=16
    )
    assert int(sol.status) == 3
    assert int(sol.num_iterations) == 10


def scipy_sl_j_true(t, h, y, z):
    m = t.shape[0]
    n = 2

    def j_block(h_i, z_0):
        return np.array(
            [
                [
                    h_i**2 * z_0**2 / 12 - 1,
                    -0.5 * h_i,
                    -(h_i**2) * z_0**2 / 12 + 1,
                    -0.5 * h_i,
                ],
                [
                    0.5 * h_i * z_0**2,
                    h_i**2 * z_0**2 / 12 - 1,
                    0.5 * h_i * z_0**2,
                    1 - h_i**2 * z_0**2 / 12,
                ],
            ]
        )

    j_true = np.zeros((m * n + 1, m * n + 1))
    for i in range(m - 1):
        j_true[i * n : (i + 1) * n, i * n : (i + 2) * n] = j_block(h[i], z[0])
    j_true[: (m - 1) * n : 2, -1] = z * h**2 / 6 * (y[0, :-1] - y[0, 1:])
    j_true[1 : (m - 1) * n : 2, -1] = z * (
        h * (y[0, :-1] + y[0, 1:]) + h**2 / 6 * (y[1, :-1] - y[1, 1:])
    )
    j_true[(m - 1) * n, 0] = 1
    j_true[(m - 1) * n + 1, (m - 1) * n] = 1
    j_true[(m - 1) * n + 2, 1] = 1
    j_true[(m - 1) * n + 2, -1] = -1
    return j_true


def test_global_jacobian_matches_closed_form():
    t = jnp.linspace(0.0, 1.0, 5)
    h = t[1:] - t[:-1]
    y_rows = jnp.stack([jnp.sin(jnp.pi * t), jnp.pi * jnp.cos(jnp.pi * t)], axis=1)
    z = jnp.array([3.0])
    cfg = make_config(sl_fun, sl_bc, 5, 2, 1)
    system = bvp_module.build_system(cfg, None, None, t[0])
    _, y_middle, _, _ = system["collocation_parts"](t, h, y_rows, z, None)
    jac = system["jacobian_at"](t, h, y_rows, z, y_middle, None)
    j_true = scipy_sl_j_true(
        np.asarray(t), np.asarray(h), np.asarray(y_rows).T, np.asarray(z)
    )
    np.testing.assert_allclose(np.asarray(jac), j_true, rtol=1e-10, atol=1e-14)


def test_global_jacobian_padding_is_an_exact_embedding():
    t5 = jnp.linspace(0.0, 1.0, 5)
    y5 = jnp.stack([jnp.sin(jnp.pi * t5), jnp.pi * jnp.cos(jnp.pi * t5)], axis=1)
    z = jnp.array([3.0])
    cfg5 = make_config(sl_fun, sl_bc, 5, 2, 1)
    system5 = bvp_module.build_system(cfg5, None, None, t5[0])
    h5 = t5[1:] - t5[:-1]
    _, y_middle5, _, _ = system5["collocation_parts"](t5, h5, y5, z, None)
    jac5 = np.asarray(system5["jacobian_at"](t5, h5, y5, z, y_middle5, None))

    t8 = jnp.concatenate([t5, jnp.full(3, t5[-1])])
    y8 = jnp.concatenate([y5, jnp.broadcast_to(y5[-1], (3, 2))])
    cfg8 = make_config(sl_fun, sl_bc, 8, 2, 1)
    system8 = bvp_module.build_system(cfg8, None, None, t8[0])
    h8 = t8[1:] - t8[:-1]
    _, y_middle8, _, _ = system8["collocation_parts"](t8, h8, y8, z, None)
    jac8 = np.asarray(system8["jacobian_at"](t8, h8, y8, z, y_middle8, None))

    # Active collocation rows are identical, with the z column relocated.
    np.testing.assert_allclose(jac8[:8, :10], jac5[:8, :10], rtol=1e-14)
    np.testing.assert_allclose(jac8[:8, 16], jac5[:8, 10], rtol=1e-14)
    # Padded interval rows are the exact copy chain [-I | I].
    eye = np.eye(2)
    for block in range(4, 7):
        rows = jac8[block * 2 : (block + 1) * 2]
        expected = np.zeros((2, 17))
        expected[:, block * 2 : (block + 1) * 2] = -eye
        expected[:, (block + 1) * 2 : (block + 2) * 2] = eye
        assert np.array_equal(rows, expected)
    # Boundary rows move to the padded last node block.
    np.testing.assert_allclose(jac8[14:, :2], jac5[8:, :2], rtol=1e-14)
    np.testing.assert_allclose(jac8[14:, 14:16], jac5[8:, 8:10], rtol=1e-14)
    np.testing.assert_allclose(jac8[14:, 16], jac5[8:, 10], rtol=1e-14)
    # The embedding preserves the determinant magnitude.
    sign5, logdet5 = np.linalg.slogdet(jac5)
    sign8, logdet8 = np.linalg.slogdet(jac8)
    np.testing.assert_allclose(logdet8, logdet5, rtol=1e-10)


def test_global_jacobian_matches_autodiff():
    # Sturm-Liouville with an unknown parameter, padded mesh.
    t = jnp.concatenate([jnp.linspace(0.0, 1.0, 5), jnp.full(3, 1.0)])
    y_rows = jnp.stack([jnp.sin(jnp.pi * t), jnp.pi * jnp.cos(jnp.pi * t)], axis=1)
    z = jnp.array([3.0])
    cfg = make_config(sl_fun, sl_bc, 8, 2, 1)
    system = bvp_module.build_system(cfg, None, None, t[0])
    h = t[1:] - t[:-1]
    _, y_middle, _, _ = system["collocation_parts"](t, h, y_rows, z, None)
    jac = system["jacobian_at"](t, h, y_rows, z, y_middle, None)

    def residual_vector(u):
        y_flat = u[:16].reshape(8, 2)
        z_flat = u[16:]
        col_res, _, _, _ = system["collocation_parts"](t, h, y_flat, z_flat, None)
        bc_res = system["bc_values"](y_flat[0], y_flat[-1], z_flat, None)
        return jnp.concatenate([col_res.reshape(-1), bc_res])

    u_0 = jnp.concatenate([y_rows.reshape(-1), z])
    jac_ad = jax.jacfwd(residual_vector)(u_0)
    np.testing.assert_allclose(
        np.asarray(jac), np.asarray(jac_ad), rtol=1e-10, atol=1e-12
    )

    # Emden with the singular term: the D/S wrapping must differentiate
    # consistently with the assembled blocks.
    t = jnp.linspace(0.0, 1.0, 10)
    y_rows = jnp.stack([emden_sol(t), jnp.full(10, 0.1)], axis=1)
    cfg = make_config(emden_fun, emden_bc, 10, 2, 0, has_singular_term=True)
    system = bvp_module.build_system(cfg, None, EMDEN_S, t[0])
    h = t[1:] - t[:-1]
    _, y_middle, _, _ = system["collocation_parts"](t, h, y_rows, None, None)
    jac = system["jacobian_at"](t, h, y_rows, None, y_middle, None)

    def emden_residual(u):
        y_flat = u.reshape(10, 2)
        col_res, _, _, _ = system["collocation_parts"](t, h, y_flat, None, None)
        bc_res = system["bc_values"](y_flat[0], y_flat[-1], None, None)
        return jnp.concatenate([col_res.reshape(-1), bc_res])

    jac_ad = jax.jacfwd(emden_residual)(y_rows.reshape(-1))
    np.testing.assert_allclose(
        np.asarray(jac), np.asarray(jac_ad), rtol=1e-10, atol=1e-12
    )


def test_structured_qr_matches_dense_linear_algebra():
    rng = np.random.default_rng(0)

    # Padded Sturm-Liouville system with an unknown-parameter border (k = 1).
    t = jnp.concatenate([jnp.linspace(0.0, 1.0, 5), jnp.full(3, 1.0)])
    y_rows = jnp.stack([jnp.sin(jnp.pi * t), jnp.pi * jnp.cos(jnp.pi * t)], axis=1)
    z = jnp.array([3.0])
    cfg = make_config(sl_fun, sl_bc, 8, 2, 1)
    system = bvp_module.build_system(cfg, None, None, t[0])
    h = t[1:] - t[:-1]
    _, y_middle, _, _ = system["collocation_parts"](t, h, y_rows, z, None)
    blocks = system["jacobian_blocks"](t, h, y_rows, z, y_middle, None)

    # Emden with the singular term, no border (k = 0).
    t_e = jnp.linspace(0.0, 1.0, 10)
    y_e = jnp.stack([emden_sol(t_e), jnp.full(10, 0.1)], axis=1)
    cfg_e = make_config(emden_fun, emden_bc, 10, 2, 0, has_singular_term=True)
    system_e = bvp_module.build_system(cfg_e, None, EMDEN_S, t_e[0])
    h_e = t_e[1:] - t_e[:-1]
    _, y_middle_e, _, _ = system_e["collocation_parts"](t_e, h_e, y_e, None, None)
    blocks_e = system_e["jacobian_blocks"](t_e, h_e, y_e, None, y_middle_e, None)

    for case_blocks in (blocks, blocks_e):
        dense = np.asarray(bvp_module.babd_dense(case_blocks))
        size = dense.shape[0]
        rhs = jnp.asarray(rng.standard_normal(size))
        np.testing.assert_allclose(
            np.asarray(babd_matvec(case_blocks, rhs)),
            dense @ np.asarray(rhs),
            rtol=1e-13,
            atol=1e-13,
        )
        state, ok = structured_qr_factor(case_blocks)
        assert bool(ok)
        np.testing.assert_allclose(
            np.asarray(structured_qr_solve(state, rhs)),
            np.linalg.solve(dense, np.asarray(rhs)),
            rtol=1e-11,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            np.asarray(structured_qr_transpose_solve(state, rhs)),
            np.linalg.solve(dense.T, np.asarray(rhs)),
            rtol=1e-11,
            atol=1e-12,
        )


def test_pytree_state_matches_flat():
    def tree_fun(t, y):
        return {"a": y["b"][0], "b": jnp.array([y["a"]])}

    def tree_bc(ya, yb):
        return jnp.array([ya["a"] - 1.0, yb["a"]])

    t = jnp.linspace(0.0, 1.0, 5)
    tree_sol = solve_bvp(
        tree_fun,
        tree_bc,
        t,
        {"a": jnp.zeros(5), "b": jnp.zeros((5, 1))},
        max_nodes=16,
    )
    flat_sol = solve_bvp(exp_fun, exp_bc, t, jnp.zeros((5, 2)), max_nodes=16)
    assert int(tree_sol.status) == int(flat_sol.status)
    assert int(tree_sol.num_nodes) == int(flat_sol.num_nodes)
    assert jnp.array_equal(tree_sol.t, flat_sol.t)
    assert jnp.array_equal(tree_sol.y["a"], flat_sol.y[:, 0])
    assert jnp.array_equal(tree_sol.y["b"][:, 0], flat_sol.y[:, 1])
    assert jnp.array_equal(tree_sol.yp["a"], flat_sol.yp[:, 0])
    assert jnp.array_equal(tree_sol.rms_residuals, flat_sol.rms_residuals)


def test_padded_tail_is_exact():
    sol = solve_bvp(
        exp_fun, exp_bc, jnp.linspace(0, 1, 5), jnp.zeros((5, 2)), max_nodes=16
    )
    num = int(sol.num_nodes)
    assert jnp.array_equal(sol.t[num:], jnp.full(16 - num, sol.t[num - 1]))
    assert jnp.array_equal(sol.y[num:], jnp.broadcast_to(sol.y[num - 1], (16 - num, 2)))
    assert jnp.array_equal(
        sol.yp[num:], jnp.broadcast_to(sol.yp[num - 1], (16 - num, 2))
    )
    assert jnp.array_equal(sol.rms_residuals[num - 1 :], jnp.zeros(16 - num))


def test_aux_at_solution():
    def aux_fun(t, y, z, args, p):
        return jnp.array([y[1], y[0]]), p * y[0]

    def aux_bc(ya, yb, z, args, p):
        return jnp.array([ya[0] - 1.0, yb[0]])

    t = jnp.linspace(0.0, 1.0, 5)
    sol = solve_bvp(aux_fun, aux_bc, t, jnp.zeros((5, 2)), p=2.0, max_nodes=16)
    assert jnp.array_equal(sol.aux, 2.0 * sol.y[:, 0])
    explicit = solve_bvp(
        aux_fun, aux_bc, t, jnp.zeros((5, 2)), p=2.0, max_nodes=16, has_aux=True
    )
    assert jnp.array_equal(explicit.aux, sol.aux)
    disabled = solve_bvp(
        exp_fun, exp_bc, t, jnp.zeros((5, 2)), max_nodes=16, has_aux=False
    )
    assert disabled.aux is None


def test_float32():
    t = jnp.linspace(0.0, 1.0, 5, dtype=jnp.float32)
    y_0 = jnp.zeros((5, 2), jnp.float32)
    sol = solve_bvp(exp_fun, exp_bc, t, y_0, max_nodes=16)
    assert sol.y.dtype == jnp.float32
    assert sol.t.dtype == jnp.float32
    assert int(sol.status) == 0
    num = int(sol.num_nodes)
    assert num == 5
    np.testing.assert_allclose(
        np.asarray(sol.y[:num, 0]), exp_sol(np.asarray(sol.t[:num])), atol=1e-4
    )
    # A tolerance below the float32 floor clamps to 100 * eps and converges.
    clamped = solve_bvp(exp_fun, exp_bc, t, y_0, tol=1e-12, max_nodes=32)
    assert int(clamped.status) == 0
