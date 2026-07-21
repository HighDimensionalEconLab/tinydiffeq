import jax
import jax.numpy as jnp
import pytest

from tinydiffeq import (
    RK4,
    IController,
    LMRootSolver,
    SaveAt,
    Tsit5,
    solve_ode,
    solve_semi_explicit_dae,
)
from tinydiffeq.dae import _build_algebraic_solver


def linear_f(y, z, t, args, p):
    return p * z


def identity_constraint(y, z):
    return z - y


def solve_linear(p, y_0, z_0, solver, save_at, **kwargs):
    return solve_semi_explicit_dae(
        linear_f,
        identity_constraint,
        solver,
        0.0,
        1.0,
        y_0,
        z_0,
        p=p,
        dt_0=kwargs.pop("dt_0", 0.05),
        max_steps=kwargs.pop("max_steps", 128),
        save_at=save_at,
        **kwargs,
    )


def test_lm_root_solver_uses_nlls_24_defaults_and_forwards_options():
    def constraint(y, z, t, args, p):
        return z - y

    defaults = _build_algebraic_solver(constraint, LMRootSolver(), False)
    assert LMRootSolver().max_steps_is_success
    assert defaults.linear_solver == "auto"
    assert defaults.jacobian_mode == "auto"
    assert defaults.ad_solver == "auto"
    assert defaults.ad_solver_penalty is None
    assert not defaults.cache_jacobian
    assert not defaults.geodesic_acceleration

    configured = _build_algebraic_solver(
        constraint,
        LMRootSolver(
            init_damping=2e-3,
            damping_decrease=0.4,
            damping_increase=3.0,
            max_damping=10.0,
            linear_solver="qr",
            jacobian_mode="rev",
            iterative_tol=1e-4,
            iterative_atol=1e-6,
            iterative_maxiter=17,
            ad_solver="augmented_qr",
            ad_solver_tol=1e-5,
            ad_solver_atol=1e-7,
            ad_solver_maxiter=13,
            ad_solver_penalty=1e-8,
        ),
        False,
    )
    assert configured.init_damping == 2e-3
    assert configured.damping_decrease == 0.4
    assert configured.damping_increase == 3.0
    assert configured.max_damping == 10.0
    assert configured.linear_solver == "qr"
    assert configured.jacobian_mode == "rev"
    assert configured.iterative_tol == 1e-4
    assert configured.iterative_atol == 1e-6
    assert configured.iterative_maxiter == 17
    assert configured.ad_solver == "augmented_qr"
    assert configured.ad_solver_tol == 1e-5
    assert configured.ad_solver_atol == 1e-7
    assert configured.ad_solver_maxiter == 13
    assert configured.ad_solver_penalty == 1e-8


def test_max_steps_policy_and_strict_batched_derivative():
    def one_lane(z_0, p, max_steps_is_success):
        return solve_semi_explicit_dae(
            lambda y, z: jnp.zeros_like(y),
            lambda y, z, t, args, p: z - p,
            RK4(),
            0.0,
            0.1,
            jnp.asarray(0.0),
            z_0,
            p=p,
            dt_0=0.1,
            max_steps=1,
            root_solver=LMRootSolver(
                max_steps=1,
                max_steps_is_success=max_steps_is_success,
                atol=1e-12,
            ),
        )

    p = jnp.asarray(1.0)
    forgiving = one_lane(jnp.asarray(0.0), p, True)
    strict = one_lane(jnp.asarray(0.0), p, False)
    assert bool(forgiving.ok)
    assert not bool(strict.ok)

    def strict_batch(parameter):
        return jax.vmap(lambda z_0: one_lane(z_0, parameter, False))(
            jnp.asarray([1.0, 0.0])
        )

    def endpoint(parameter):
        result = strict_batch(parameter)
        return jnp.where(result.ok, result.zs, 0.0)

    result = strict_batch(p)
    value, tangent = jax.jvp(endpoint, (p,), (jnp.ones_like(p),))
    _, pullback = jax.vjp(endpoint, p)
    assert jnp.array_equal(result.ok, jnp.asarray([True, False]))
    assert jnp.array_equal(value, jnp.asarray([1.0, 0.0]))
    assert jnp.array_equal(tangent, jnp.asarray([1.0, 0.0]))
    assert jnp.array_equal(pullback(jnp.ones_like(value))[0], jnp.asarray(1.0))


def test_rk4_initial_consistency_and_endpoint_accuracy():
    sol = solve_linear(
        jnp.asarray(1.0),
        jnp.asarray(1.0),
        jnp.asarray(-4.0),
        RK4(),
        SaveAt(t_1=True),
        dt_0=0.05,
        max_steps=20,
    )
    assert bool(sol.ok)
    assert int(sol.num_accepted) == 20
    assert jnp.abs(sol.ys - jnp.e) < 2e-5
    assert jnp.abs(sol.zs - sol.ys) < 2e-6


def test_rk4_fourth_order_convergence():
    def error(n):
        sol = solve_linear(
            jnp.asarray(-1.0),
            jnp.asarray(1.0),
            jnp.asarray(1.0),
            RK4(),
            SaveAt(t_1=True),
            dt_0=1.0 / n,
            max_steps=n,
        )
        return jnp.abs(sol.ys - jnp.exp(-1.0))

    assert error(4) / error(8) > 10.0


def test_adaptive_tsit5_and_steps_padding():
    sol = solve_linear(
        jnp.asarray(2.0),
        jnp.asarray(1.0),
        jnp.asarray(1.0),
        Tsit5(),
        SaveAt(steps=True),
        dt_0=0.3,
        max_steps=64,
        controller=IController(rtol=1e-5, atol=1e-7),
    )
    assert bool(sol.ok)
    assert sol.ts.shape == (65,)
    assert int(sol.accepted.sum()) == int(sol.num_accepted) + 1
    assert bool(jnp.all(sol.ts[1:] >= sol.ts[:-1]))
    assert jnp.max(jnp.abs((sol.zs - sol.ys)[sol.accepted])) < 2e-6
    assert jnp.abs(sol.ys[-1] - jnp.exp(2.0)) < 2e-4


def test_interpolated_z_has_small_constraint_defect():
    def f(y, z):
        return z

    def g(y, z):
        return z**2 - (y + 2.0)

    grid = jnp.linspace(0.0, 1.0, 13)
    sol = solve_semi_explicit_dae(
        f,
        g,
        Tsit5(),
        0.0,
        1.0,
        jnp.asarray(1.0),
        jnp.asarray(2.0),
        dt_0=0.2,
        controller=IController(rtol=1e-5, atol=1e-7),
        max_steps=64,
        save_at=SaveAt(ts=grid),
    )
    assert bool(sol.ok)
    assert jnp.array_equal(sol.ts, grid)
    assert jnp.max(jnp.abs(sol.zs**2 - sol.ys - 2.0)) < 2e-6


def test_jvp_vjp_and_reverse_over_forward():
    y_0 = jnp.asarray(0.7)
    grid = jnp.linspace(0.0, 1.0, 7)

    def endpoint(p):
        return solve_linear(
            p,
            y_0,
            jnp.asarray(3.0),
            Tsit5(),
            SaveAt(t_1=True),
            dt_0=0.1,
            max_steps=128,
            controller=IController(rtol=1e-6, atol=1e-8),
            root_solver=LMRootSolver(atol=1e-7),
        ).ys

    p = jnp.asarray(1.3)
    tangent = jnp.asarray(1.0)
    exact_first = y_0 * jnp.exp(p)
    jvp = jax.jvp(endpoint, (p,), (tangent,))[1]
    vjp = jax.grad(endpoint)(p)
    second = jax.grad(lambda q: jax.jvp(endpoint, (q,), (tangent,))[1])(p)
    assert jnp.abs(jvp - exact_first) < 2e-5
    assert jnp.abs(vjp - exact_first) < 2e-5
    assert jnp.abs(second - exact_first) < 5e-5

    def sampled_sum(q):
        return jnp.sum(
            solve_linear(
                q,
                y_0,
                y_0,
                Tsit5(),
                SaveAt(ts=grid),
                dt_0=0.1,
                max_steps=128,
                controller=IController(rtol=1e-6, atol=1e-8),
                root_solver=LMRootSolver(atol=1e-7),
            ).zs
        )

    exact_sampled = jnp.sum(grid * y_0 * jnp.exp(p * grid))
    assert jnp.abs(jax.grad(sampled_sum)(p) - exact_sampled) < 5e-4


@pytest.mark.parametrize(
    ("dtype", "root_atol", "comparison_atol", "transpose_atol"),
    [
        (jnp.float32, 1e-6, 1e-7, 1e-7),
        (jnp.float64, 1e-12, 1e-15, 1e-15),
    ],
)
def test_nonsymmetric_square_constraint_jvp_and_vjp_match_closed_form(
    dtype, root_atol, comparison_atol, transpose_atol
):
    matrix = jnp.asarray([[2.0, -1.0], [0.5, 1.5]], dtype=dtype)
    parameter_map = jnp.asarray([[1.0, 2.0], [-0.5, 1.0]], dtype=dtype)
    y_0 = jnp.asarray([0.7, -0.2], dtype=dtype)
    z_guess = jnp.zeros(2, dtype=dtype)
    direction = jnp.asarray([0.6, -0.8], dtype=dtype)
    cotangent = jnp.asarray([-0.5, 1.2], dtype=dtype)

    def endpoint(p):
        return solve_semi_explicit_dae(
            lambda y, z: jnp.zeros_like(y),
            lambda y, z, t, args, p: matrix @ z - y - parameter_map @ p,
            RK4(),
            0.0,
            0.1,
            y_0,
            z_guess,
            p=p,
            dt_0=0.1,
            max_steps=1,
            root_solver=LMRootSolver(max_steps=8, atol=root_atol),
        ).zs

    def transformed(p):
        value, tangent = jax.jvp(endpoint, (p,), (direction,))
        _, pullback = jax.vjp(endpoint, p)
        return value, tangent, pullback(cotangent)[0]

    p = jnp.asarray([0.3, -0.4], dtype=dtype)
    value, tangent, pulled_back = jax.jit(transformed)(p)
    expected_value = jnp.linalg.solve(matrix, y_0 + parameter_map @ p)
    expected_tangent = jnp.linalg.solve(matrix, parameter_map @ direction)
    expected_pullback = parameter_map.T @ jnp.linalg.solve(matrix.T, cotangent)

    assert jnp.allclose(value, expected_value, rtol=0.0, atol=comparison_atol)
    assert jnp.allclose(tangent, expected_tangent, rtol=0.0, atol=comparison_atol)
    assert jnp.allclose(pulled_back, expected_pullback, rtol=0.0, atol=comparison_atol)
    assert jnp.allclose(
        cotangent @ tangent,
        pulled_back @ direction,
        rtol=0.0,
        atol=transpose_atol,
    )


def test_root_guess_has_zero_derivative_and_y_0_differentiates():
    p = jnp.asarray(0.8)

    def from_guess(z_0):
        return solve_linear(
            p,
            jnp.asarray(1.0),
            z_0,
            RK4(),
            SaveAt(t_1=True),
            dt_0=0.05,
            max_steps=20,
        ).ys

    def from_y_0(y_0):
        return solve_linear(
            p,
            y_0,
            jnp.asarray(1.0),
            RK4(),
            SaveAt(t_1=True),
            dt_0=0.025,
            max_steps=40,
        ).ys

    assert jax.grad(from_guess)(jnp.asarray(2.0)) == 0.0
    assert jnp.abs(jax.grad(from_y_0)(jnp.asarray(1.0)) - jnp.exp(p)) < 2e-5


def test_jit_and_vmap():
    @jax.jit
    def endpoint(p, y_0):
        return solve_linear(
            p,
            y_0,
            y_0,
            Tsit5(),
            SaveAt(t_1=True),
            dt_0=0.1,
            max_steps=64,
            controller=IController(rtol=1e-5, atol=1e-7),
        ).ys

    ps = jnp.asarray([0.5, 1.0, 1.5])
    y_0s = jnp.asarray([0.7, 1.0, 1.3])
    got = jax.vmap(endpoint)(ps, y_0s)
    assert jnp.max(jnp.abs(got - y_0s * jnp.exp(ps))) < 2e-4


def test_kernels_optimal_advertising_system_matches_elimination():
    beta, cost, kappa = 0.05, 0.5, 0.5

    def f(y, u, t, args, rho):
        x, mu = y
        gamma = (beta + rho) / cost
        return jnp.asarray(
            [(1.0 - x) * u - beta * x, -gamma + (rho + beta) * mu + mu * u]
        )

    def g(y, u):
        x, mu = y
        return u - kappa * mu * (1.0 - x)

    def reduced(y, t, args, rho):
        x, mu = y
        u = kappa * mu * (1.0 - x)
        return f(y, u, t, args, rho)

    rho = jnp.asarray(0.11)
    y_0 = jnp.asarray([0.4, 0.8])
    dae = solve_semi_explicit_dae(
        f,
        g,
        RK4(),
        0.0,
        0.5,
        y_0,
        jnp.asarray(0.1),
        p=rho,
        dt_0=0.01,
        max_steps=50,
        save_at=SaveAt(steps=True),
    )
    ode = solve_ode(
        reduced,
        RK4(),
        0.0,
        0.5,
        y_0,
        p=rho,
        dt_0=0.01,
        max_steps=50,
        save_at=SaveAt(steps=True),
    )
    assert bool(dae.ok)
    assert jnp.max(jnp.abs(dae.ys - ode.xs)) < 2e-5
    x, mu = dae.ys[dae.accepted].T
    u = dae.zs[dae.accepted]
    assert jnp.max(jnp.abs(u - kappa * mu * (1.0 - x))) < 2e-6
    rho_grad = jax.grad(
        lambda q: solve_semi_explicit_dae(
            f,
            g,
            RK4(),
            0.0,
            0.5,
            y_0,
            jnp.asarray(0.1),
            p=q,
            dt_0=0.01,
            max_steps=50,
        ).ys[0]
    )(rho)
    assert bool(jnp.isfinite(rho_grad))


def test_kernels_one_capital_growth_system_matches_elimination():
    alpha, delta = 0.3, 0.08

    def f(y, c, t, args, rho):
        k, mu = y
        output = k**alpha
        marginal_product = alpha * k ** (alpha - 1.0)
        return jnp.asarray(
            [output - delta * k - c, -mu * (marginal_product - delta - rho)]
        )

    def g(y, c):
        return y[1] * c - 1.0

    def reduced(y, t, args, rho):
        return f(y, 1.0 / y[1], t, args, rho)

    rho = jnp.asarray(0.04)
    y_0 = jnp.asarray([0.8, 1.2])
    dae = solve_semi_explicit_dae(
        f,
        g,
        RK4(),
        0.0,
        0.5,
        y_0,
        jnp.asarray(0.5),
        p=rho,
        dt_0=0.01,
        max_steps=50,
        save_at=SaveAt(steps=True),
    )
    ode = solve_ode(
        reduced,
        RK4(),
        0.0,
        0.5,
        y_0,
        p=rho,
        dt_0=0.01,
        max_steps=50,
        save_at=SaveAt(steps=True),
    )
    assert bool(dae.ok)
    assert jnp.max(jnp.abs(dae.ys - ode.xs)) < 2e-5
    mu = dae.ys[dae.accepted, 1]
    consumption = dae.zs[dae.accepted]
    assert jnp.max(jnp.abs(mu * consumption - 1.0)) < 2e-6


def test_initial_root_failure_and_time_budget_failure():
    failed_root = solve_semi_explicit_dae(
        lambda y, z: z,
        lambda y, z: z**2 + 1.0,
        RK4(),
        0.0,
        1.0,
        jnp.asarray(1.0),
        jnp.asarray(0.0),
        dt_0=0.1,
        max_steps=10,
        root_solver=LMRootSolver(max_steps_is_success=False),
    )
    assert not bool(failed_root.ok)
    assert int(failed_root.num_accepted) == 0

    starved = solve_linear(
        jnp.asarray(1.0),
        jnp.asarray(1.0),
        jnp.asarray(1.0),
        RK4(),
        SaveAt(t_1=True),
        dt_0=0.1,
        max_steps=3,
    )
    assert not bool(starved.ok)
    assert int(starved.num_accepted) == 3
    assert jnp.abs(starved.ts - 0.3) < 1e-7


def test_adaptive_stage_root_failure_retries_with_smaller_step():
    # One damped LM iteration cannot meet the root tolerance at dt_0, but it
    # can after the adaptive controller reduces the step. A fixed controller
    # would terminate on the same stage-root failure.
    sol = solve_semi_explicit_dae(
        lambda y, z: jnp.ones_like(y),
        identity_constraint,
        Tsit5(),
        0.0,
        0.005,
        jnp.asarray(0.0),
        jnp.asarray(0.0),
        dt_0=0.005,
        controller=IController(),
        root_solver=LMRootSolver(max_steps=1, max_steps_is_success=False, atol=1e-6),
        max_steps=64,
        save_at=SaveAt(steps=True),
    )
    assert bool(sol.ok)
    assert int(sol.num_accepted) > 1
    assert jnp.abs(sol.ys[-1] - 0.005) < 1e-7
    assert jnp.abs(sol.zs[-1] - sol.ys[-1]) < 1e-6


def test_masked_failed_lane_has_safe_implicit_root_jvp_and_vjp():
    y_0 = jnp.asarray([1.0, -1.0])
    z_0 = jnp.asarray([1.0, 0.0])

    def one_lane(y, z, p):
        return solve_semi_explicit_dae(
            lambda y, z: jnp.zeros_like(y),
            lambda y, z, t, args, p: z**2 - y - p,
            RK4(),
            0.0,
            0.1,
            y,
            z,
            p=p,
            dt_0=0.1,
            max_steps=1,
            root_solver=LMRootSolver(max_steps_is_success=False),
            failure_ad_reference=(1.0, 1.0, 0.0, 0.0),
        )

    def batch(p):
        return jax.vmap(lambda y, z: one_lane(y, z, p))(y_0, z_0)

    def loss(p):
        sol = batch(p)
        return jnp.sum(jnp.where(sol.ok, sol.zs, 0.0))

    p = jnp.asarray(0.0)
    assert jnp.array_equal(batch(p).ok, jnp.asarray([True, False]))
    assert jnp.allclose(jax.jvp(loss, (p,), (jnp.ones_like(p),))[1], 0.5)
    assert jnp.allclose(jax.grad(loss)(p), 0.5)


@pytest.mark.parametrize(
    ("save_at", "multiplicity"),
    [
        (SaveAt(t_1=True), 1.0),
        (SaveAt(ts=jnp.asarray([0.0, 0.1])), 2.0),
    ],
)
@pytest.mark.parametrize("domain_sign", [1.0, -1.0])
def test_nonfinite_failed_root_cannot_poison_masked_vjp(
    save_at, multiplicity, domain_sign
):
    y_0 = jnp.asarray([domain_sign, -2.0 * domain_sign])
    z_0 = jnp.asarray([1.0, 0.0])

    def one_lane(y, z, p):
        return solve_semi_explicit_dae(
            lambda y, z: jnp.zeros_like(y),
            lambda y, z, t, args, p: z - jnp.sqrt(domain_sign * (y + p)),
            RK4(),
            0.0,
            0.1,
            y,
            z,
            p=p,
            dt_0=0.1,
            max_steps=1,
            save_at=save_at,
            failure_ad_reference=(domain_sign, 1.0, 0.0, 0.0),
        )

    def batch(p):
        return jax.vmap(lambda y, z: one_lane(y, z, p))(y_0, z_0)

    def loss(p):
        sol = batch(p)
        lane_values = sol.zs.reshape(2, -1).sum(axis=1)
        return jnp.sum(jnp.where(sol.ok, lane_values, 0.0))

    p = jnp.asarray(0.0)
    expected = 0.5 * domain_sign * multiplicity
    assert jnp.array_equal(batch(p).ok, jnp.asarray([True, False]))
    assert jnp.allclose(jax.jvp(loss, (p,), (jnp.ones_like(p),))[1], expected)
    assert jnp.allclose(jax.grad(loss)(p), expected)


def test_validation():
    with pytest.raises(ValueError, match="dt_0 is required"):
        solve_semi_explicit_dae(
            lambda y, z: z,
            identity_constraint,
            RK4(),
            0.0,
            1.0,
            1.0,
            1.0,
        )
    with pytest.raises(ValueError, match="positive int"):
        LMRootSolver(max_steps=0)
    with pytest.raises(TypeError, match="max_steps_is_success must be a bool"):
        LMRootSolver(max_steps_is_success=1)
    with pytest.raises(ValueError, match="gtol must be nonnegative"):
        LMRootSolver(gtol=-1.0)
    with pytest.raises(ValueError, match="xtol must be nonnegative"):
        LMRootSolver(xtol=-1.0)
    with pytest.raises(ValueError, match="2 to 5 positional"):
        solve_semi_explicit_dae(
            lambda y: y,
            identity_constraint,
            RK4(),
            0.0,
            1.0,
            1.0,
            1.0,
            dt_0=0.1,
        )
    with pytest.raises(TypeError, match="failure_ad_reference"):
        solve_semi_explicit_dae(
            lambda y, z: z,
            identity_constraint,
            RK4(),
            0.0,
            1.0,
            1.0,
            1.0,
            dt_0=0.1,
            failure_ad_reference=(1.0, 1.0),
        )
    with pytest.raises(ValueError, match="finite"):
        solve_semi_explicit_dae(
            lambda y, z: z,
            identity_constraint,
            RK4(),
            0.0,
            1.0,
            1.0,
            1.0,
            dt_0=0.1,
            failure_ad_reference=(jnp.nan, 1.0, 0.0, None),
        )

    @jax.jit
    def jitted_with_reference(y_0):
        return solve_semi_explicit_dae(
            lambda y, z: z,
            identity_constraint,
            RK4(),
            0.0,
            0.1,
            y_0,
            jnp.asarray(1.0),
            dt_0=0.1,
            max_steps=1,
            failure_ad_reference=(1.0, 1.0, 0.0, None),
        ).ys

    assert jnp.isfinite(jitted_with_reference(jnp.asarray(1.0)))
