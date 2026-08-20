import jax
import jax.numpy as jnp
import numpy as np

from tinydiffeq import solve_bvp

# y'' = -z^2 p y on [0, 1] with bc [y(0), y(1), y'(0) - z]: the eigenvalue is
# z*(p) = pi / sqrt(p) and y(t) = sin(pi t) / sqrt(p), so at p = 1
#   dz*/dp = -pi/2, dy/dp = -sin(pi t)/2, dy'/dp = -pi cos(pi t)/2.


def eigen_fun(t, y, z, args, p):
    return jnp.array([y[1], -(z[0] ** 2) * p * y[0]])


def eigen_bc(ya, yb, z, args, p):
    return jnp.array([ya[0], yb[0], ya[1] - z[0]])


T_GRID = jnp.linspace(0.0, 1.0, 9)
Y_GUESS = jnp.stack(
    [jnp.sin(jnp.pi * T_GRID), jnp.pi * jnp.cos(jnp.pi * T_GRID)], axis=1
)
MAX_NODES = 128


def eigen_solve(p):
    return solve_bvp(
        eigen_fun,
        eigen_bc,
        T_GRID,
        Y_GUESS,
        jnp.array([3.0]),
        p=p,
        tol=1e-6,
        max_nodes=MAX_NODES,
    )


def test_jvp_matches_closed_form():
    sol, tangent = jax.jvp(eigen_solve, (1.0,), (1.0,))
    assert int(sol.status) == 0
    num = int(sol.num_nodes)
    np.testing.assert_allclose(float(tangent.z[0]), -np.pi / 2, rtol=1e-6)
    t_active = np.asarray(sol.t[:num])
    np.testing.assert_allclose(
        np.asarray(tangent.y[:num, 0]),
        -np.sin(np.pi * t_active) / 2,
        rtol=1e-5,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        np.asarray(tangent.yp[:num, 0]),
        -np.pi * np.cos(np.pi * t_active) / 2,
        rtol=1e-5,
        atol=1e-7,
    )
    assert jnp.array_equal(tangent.t, jnp.zeros(MAX_NODES))
    assert jnp.array_equal(tangent.rms_residuals, jnp.zeros(MAX_NODES - 1))


def test_vjp_matches_closed_form():
    gradient = jax.grad(lambda p: eigen_solve(p).z[0])(1.0)
    np.testing.assert_allclose(float(gradient), -np.pi / 2, rtol=1e-6)

    gradient = jax.grad(lambda p: jnp.sum(eigen_solve(p).y[:, 0]))(1.0)
    sol = eigen_solve(1.0)
    num = int(sol.num_nodes)
    t_active = np.asarray(sol.t[:num])
    # The padded tail repeats the last active row, so its tangent repeats too.
    expected = np.sum(-np.sin(np.pi * t_active) / 2) + (MAX_NODES - num) * (
        -np.sin(np.pi * t_active[-1]) / 2
    )
    np.testing.assert_allclose(float(gradient), expected, rtol=1e-5, atol=1e-8)


def test_second_order_matches_closed_form():
    # z*(p) = pi p^(-1/2): d2z/dp2 = (3 pi / 4) p^(-5/2) = 3 pi / 4 at p = 1.
    hessian = jax.hessian(lambda p: eigen_solve(p).z[0])(1.0)
    np.testing.assert_allclose(float(hessian), 3 * np.pi / 4, rtol=1e-5)
    reverse_over_forward = jax.grad(
        lambda p: jax.jvp(lambda q: eigen_solve(q).z[0], (p,), (1.0,))[1]
    )(1.0)
    np.testing.assert_allclose(float(reverse_over_forward), 3 * np.pi / 4, rtol=1e-5)


def test_vjp_is_the_transpose_of_jvp():
    cotangent = jax.random.normal(jax.random.key(0), (MAX_NODES, 2))

    def scalar(p):
        sol = eigen_solve(p)
        return jnp.sum(cotangent * sol.y) + 0.7 * sol.z[0]

    _, jvp_value = jax.jvp(scalar, (1.0,), (1.0,))
    vjp_value = jax.grad(scalar)(1.0)
    np.testing.assert_allclose(float(vjp_value), float(jvp_value), rtol=1e-12)


def test_aux_tangent():
    def aux_fun(t, y, z, args, p):
        return jnp.array([y[1], -(z[0] ** 2) * p * y[0]]), p * y[0]

    def solve(p):
        return solve_bvp(
            aux_fun,
            eigen_bc,
            T_GRID,
            Y_GUESS,
            jnp.array([3.0]),
            p=p,
            tol=1e-6,
            max_nodes=MAX_NODES,
        )

    sol, tangent = jax.jvp(solve, (1.0,), (1.0,))
    num = int(sol.num_nodes)
    t_active = np.asarray(sol.t[:num])
    # aux = p y(t): d aux/dp = y + p dy/dp = sin(pi t)/2 at p = 1.
    np.testing.assert_allclose(
        np.asarray(tangent.aux[:num]),
        np.sin(np.pi * t_active) / 2,
        rtol=1e-5,
        atol=1e-8,
    )


def test_inert_inputs_have_exactly_zero_gradients():
    def with_args_fun(t, y, z, args, p):
        return jnp.array([y[1], -(z[0] ** 2) * p * args * y[0]])

    def objective(t, y_0, z_0, args):
        sol = solve_bvp(
            with_args_fun,
            eigen_bc,
            t,
            y_0,
            z_0,
            p=1.0,
            args=args,
            tol=1e-6,
            max_nodes=MAX_NODES,
        )
        return sol.z[0]

    gradients = jax.grad(objective, argnums=(0, 1, 2, 3))(
        T_GRID, Y_GUESS, jnp.array([3.0]), 1.0
    )
    for gradient in gradients:
        assert bool(jnp.all(gradient == 0.0))


def test_failed_solve_has_zero_finite_tangents():
    def solve(p):
        # tol = 1e-8 needs far more than 64 nodes: status 1.
        return solve_bvp(
            eigen_fun,
            eigen_bc,
            T_GRID,
            Y_GUESS,
            jnp.array([3.0]),
            p=p,
            tol=1e-8,
            max_nodes=64,
        )

    sol, tangent = jax.jvp(solve, (1.0,), (1.0,))
    assert int(sol.status) == 1
    assert jnp.array_equal(tangent.z, jnp.zeros(1))
    assert jnp.array_equal(tangent.y, jnp.zeros((64, 2)))
    gradient = jax.grad(lambda p: solve(p).z[0])(1.0)
    assert float(gradient) == 0.0


def test_failed_solve_with_nonfinite_tangent_program_stays_zero():
    # d/dp sqrt(y + p) is infinite at the inert guess y = 0, p = 0, so the
    # failed lane's tangent program produces non-finite intermediates that
    # the zero-tangent contract must mask by selection, not multiplication.
    def sqrt_fun(t, y, z, args, p):
        return jnp.array([y[1], jnp.sqrt(y[0] + p)])

    def sqrt_bc(ya, yb, z, args, p):
        return jnp.array([ya[0] + 2.0, yb[0]])

    t = jnp.linspace(0.0, 1.0, 5)

    def solve(p):
        return solve_bvp(
            sqrt_fun, sqrt_bc, t, jnp.zeros((5, 2)), p=p, tol=1e-10, max_nodes=8
        )

    sol, tangent = jax.jvp(solve, (0.0,), (1.0,))
    assert int(sol.status) != 0
    assert jnp.array_equal(tangent.y, jnp.zeros((8, 2)))
    gradient = jax.grad(lambda p: solve(p).y[0, 0])(0.0)
    assert float(gradient) == 0.0


def test_singular_solution_jacobian_reports_nan_in_both_modes():
    # y' = 0 with bc ya^2 - ya^3/2 - p: p = 0 converges to the double root
    # ya = 0, where the bc Jacobian (so the AD-rule refactor) is singular.
    def flat_fun(t, y, z, args, p):
        return jnp.zeros_like(y)

    def root_bc(ya, yb, z, args, p):
        return jnp.array([ya[0] ** 2 - 0.5 * ya[0] ** 3 - p])

    t = jnp.linspace(0.0, 1.0, 3)

    def solve(p):
        return solve_bvp(flat_fun, root_bc, t, jnp.ones((3, 1)), p=p, max_nodes=8)

    sol = solve(0.0)
    assert int(sol.status) == 0
    assert float(sol.y[0, 0]) == 0.0
    _, tangent = jax.jvp(solve, (0.0,), (1.0,))
    assert bool(jnp.all(jnp.isnan(tangent.y)))
    gradient = jax.grad(lambda p: solve(p).y[0, 0])(0.0)
    assert bool(jnp.isnan(gradient))


def test_jit_and_ad_compose():
    gradient = jax.grad(lambda p: eigen_solve(p).z[0])(1.0)
    jitted = jax.jit(jax.grad(lambda p: eigen_solve(p).z[0]))(1.0)
    np.testing.assert_allclose(float(jitted), float(gradient), rtol=1e-13)
    _, jvp_eager = jax.jvp(lambda p: eigen_solve(p).z[0], (1.0,), (1.0,))
    jvp_jitted = jax.jit(
        lambda p: jax.jvp(lambda q: eigen_solve(q).z[0], (p,), (1.0,))[1]
    )(1.0)
    np.testing.assert_allclose(float(jvp_jitted), float(jvp_eager), rtol=1e-12)


def test_singular_term_gradient_matches_finite_differences():
    def emden_fun(t, y, z, args, p):
        return jnp.array([y[1], -(y[0] ** 5)])

    def emden_bc(ya, yb, z, args, p):
        return jnp.array([ya[1], yb[0] - p * (3.0 / 4.0) ** 0.5])

    S = jnp.array([[0.0, 0.0], [0.0, -2.0]])
    t = jnp.linspace(0.0, 1.0, 10)
    y_0 = jnp.stack([jnp.full(10, (3.0 / 4.0) ** 0.5), jnp.full(10, 1e-4)], axis=1)

    def solve(p):
        return solve_bvp(emden_fun, emden_bc, t, y_0, p=p, S=S, tol=1e-6, max_nodes=64)

    value = solve(1.0)
    assert int(value.status) == 0
    gradient = jax.grad(lambda p: solve(p).y[5, 0])(1.0)
    step = 1e-6
    upper, lower = solve(1.0 + step), solve(1.0 - step)
    # Central differences are only valid on an unchanged mesh.
    assert int(upper.num_nodes) == int(lower.num_nodes) == int(value.num_nodes)
    finite_difference = (float(upper.y[5, 0]) - float(lower.y[5, 0])) / (2 * step)
    np.testing.assert_allclose(float(gradient), finite_difference, rtol=1e-4)
