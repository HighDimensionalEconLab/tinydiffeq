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


def test_requested_times_resolve_nonlinear_constraint():
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
        root_solver=LMRootSolver(max_steps=1, atol=1e-6),
        max_steps=64,
        save_at=SaveAt(steps=True),
    )
    assert bool(sol.ok)
    assert int(sol.num_accepted) > 1
    assert jnp.abs(sol.ys[-1] - 0.005) < 1e-7
    assert jnp.abs(sol.zs[-1] - sol.ys[-1]) < 1e-6


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
