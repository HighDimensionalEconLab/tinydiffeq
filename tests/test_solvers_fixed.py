import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import numpy as np
import pytest
from kernels_reference import rk4_grid

from tinydiffeq import RK4, Euler, IController, SaveAt, Tsit5, solve_ode


def logistic_exact(x_0, t):
    return x_0 * jnp.exp(t) / (1.0 + x_0 * (jnp.exp(t) - 1.0))


@pytest.mark.parametrize(
    ("dtype", "euler_tol", "rk4_tol"),
    [(jnp.float32, 4e-3, 1e-4), (jnp.float64, 2e-3, 1e-12)],
)
def test_linear_system_vs_expm(dtype, euler_tol, rk4_tol):
    A = jnp.asarray([[0.0, 1.0], [-1.0, -0.3]], dtype)
    x_0 = jnp.asarray([1.0, 0.5], dtype)
    T = 2.0
    exact = jsp_linalg.expm(A * T) @ x_0

    def f(x):
        return A @ x

    n = 2000
    euler = solve_ode(f, Euler(), 0.0, T, x_0, dt_0=T / n, max_steps=n)
    rk4 = solve_ode(f, RK4(), 0.0, T, x_0, dt_0=T / n, max_steps=n)
    assert bool(euler.ok) and bool(rk4.ok)
    assert euler.xs.dtype == dtype and rk4.xs.dtype == dtype
    assert int(euler.num_steps) == int(euler.num_accepted) == n
    assert int(rk4.num_steps) == int(rk4.num_accepted) == n
    assert jnp.max(jnp.abs(euler.xs - exact)) < euler_tol
    assert jnp.max(jnp.abs(rk4.xs - exact)) < rk4_tol


@pytest.mark.parametrize(
    ("dtype", "tol"),
    [(jnp.float32, 2e-5), (jnp.float64, 1e-9)],
)
def test_logistic_closed_form(dtype, tol):
    x_0 = jnp.asarray(0.1, dtype)
    T = 3.0
    n = 300
    sol = solve_ode(
        lambda x: x * (1.0 - x), RK4(), 0.0, T, x_0, dt_0=T / n, max_steps=n
    )
    assert sol.xs.dtype == dtype
    assert jnp.abs(sol.xs - logistic_exact(x_0, jnp.asarray(T, dtype))) < tol


def test_convergence_slopes():
    x_0 = jnp.asarray(0.1)
    T = 2.0
    exact = logistic_exact(x_0, T)

    def f(x):
        return x * (1.0 - x)

    for solver, expected in ((Euler(), 1.0), (RK4(), 4.0)):
        errors, dts = [], []
        for n in (20, 40, 80, 160):
            sol = solve_ode(f, solver, 0.0, T, x_0, dt_0=T / n, max_steps=n)
            assert bool(sol.ok)
            errors.append(float(jnp.abs(sol.xs - exact)))
            dts.append(T / n)
        slope = np.polyfit(np.log(dts), np.log(errors), 1)[0]
        assert abs(slope - expected) < 0.3, (type(solver).__name__, slope)


def test_unroll_matches_rolled_and_requires_fixed_stepping():
    x_0 = jnp.asarray(0.1)
    n = 16

    def solve(unroll):
        return solve_ode(
            lambda x: x * (1.0 - x),
            RK4(),
            0.0,
            1.0,
            x_0,
            dt_0=1.0 / n,
            max_steps=n,
            save_at=SaveAt(steps=True),
            unroll=unroll,
        )

    rolled, unrolled = solve(1), solve(4)
    assert jnp.array_equal(rolled.xs, unrolled.xs)
    assert jnp.array_equal(rolled.ts, unrolled.ts)
    grad_rolled = jax.grad(
        lambda x: (
            solve_ode(
                lambda v: v * (1.0 - v), RK4(), 0.0, 1.0, x, dt_0=1.0 / n, max_steps=n
            ).xs
        )
    )(x_0)
    grad_unrolled = jax.grad(
        lambda x: (
            solve_ode(
                lambda v: v * (1.0 - v),
                RK4(),
                0.0,
                1.0,
                x,
                dt_0=1.0 / n,
                max_steps=n,
                unroll=4,
            ).xs
        )
    )(x_0)
    assert jnp.allclose(grad_rolled, grad_unrolled, rtol=1e-12, atol=1e-12)
    with pytest.raises(ValueError, match="fixed stepping"):
        solve_ode(
            lambda x: -x,
            Tsit5(),
            0.0,
            1.0,
            x_0,
            dt_0=0.1,
            controller=IController(),
            max_steps=32,
            unroll=4,
        )
    with pytest.raises(ValueError, match="at least 1"):
        solve(0)


def test_non_dividing_dt_0_lands_on_t_1():
    x_0 = jnp.asarray(0.1)
    sol = solve_ode(
        lambda x: x * (1.0 - x), RK4(), 0.0, 1.0, x_0, dt_0=0.3, max_steps=4
    )
    assert bool(sol.ok)
    assert sol.ts == 1.0
    assert int(sol.num_accepted) == 4
    assert jnp.abs(sol.xs - logistic_exact(x_0, 1.0)) < 1e-4


def test_completed_solve_skips_post_horizon_field_evaluations():
    evaluation_times = []

    def f(x, t):
        jax.debug.callback(
            lambda value: evaluation_times.append(float(value)), t, ordered=True
        )
        return -x

    sol = solve_ode(
        f,
        Euler(),
        0.0,
        1.0,
        jnp.asarray(1.0),
        dt_0=0.25,
        max_steps=20,
    )
    jax.block_until_ready(sol.xs)
    assert bool(sol.ok)
    assert evaluation_times == [0.0, 0.25, 0.5, 0.75]


def test_fixed_traced_horizon_uses_one_clipped_scan_with_static_branch_parity():
    def traced_solve(horizon):
        return solve_ode(
            lambda x: -x,
            RK4(),
            0.0,
            horizon,
            jnp.asarray(1.0),
            dt_0=0.25,
            max_steps=4,
            has_aux=False,
        )

    static = solve_ode(
        lambda x: -x,
        RK4(),
        0.0,
        1.0,
        jnp.asarray(1.0),
        dt_0=0.25,
        max_steps=4,
        has_aux=False,
    )
    traced = jax.jit(traced_solve)(jnp.asarray(1.0))
    assert jnp.array_equal(traced.ts, static.ts)
    assert jnp.array_equal(traced.xs, static.xs)
    assert traced.ok == static.ok
    assert traced.num_accepted == static.num_accepted
    assert traced.num_steps == static.num_steps

    horizons = jnp.asarray([0.75, 1.0])
    batched = jax.jit(jax.vmap(traced_solve))(horizons)
    for index, horizon in enumerate(horizons):
        scalar = jax.jit(traced_solve)(horizon)
        assert jnp.array_equal(batched.ts[index], scalar.ts)
        assert jnp.array_equal(batched.xs[index], scalar.xs)
        assert batched.ok[index] == scalar.ok
        assert batched.num_accepted[index] == scalar.num_accepted

    # A batched cond over two complete scans executes both branches. The traced
    # horizon path must instead contain only the one clipped integration scan.
    jaxpr = str(jax.make_jaxpr(jax.vmap(traced_solve))(horizons))
    assert jaxpr.count("scan[") == 1

    def static_uniform(initial):
        return solve_ode(
            lambda x: -x,
            RK4(),
            0.0,
            1.0,
            initial,
            dt_0=0.25,
            max_steps=4,
            has_aux=False,
        ).xs

    static_jaxpr = str(jax.make_jaxpr(static_uniform)(jnp.asarray(1.0)))
    assert static_jaxpr.count("scan[") == 1
    assert "cond[" not in static_jaxpr


def test_scalar_vs_vector_shapes():
    def f(x):
        return -x

    n = 8
    scalar = solve_ode(
        f, RK4(), 0.0, 1.0, 1.0, dt_0=1.0 / n, max_steps=n, save_at=SaveAt(steps=True)
    )
    vector = solve_ode(
        f,
        RK4(),
        0.0,
        1.0,
        jnp.asarray([1.0, 2.0]),
        dt_0=1.0 / n,
        max_steps=n,
        save_at=SaveAt(steps=True),
    )
    assert scalar.xs.shape == (n + 1,)
    assert vector.xs.shape == (n + 1, 2)
    endpoint = solve_ode(f, RK4(), 0.0, 1.0, 1.0, dt_0=1.0 / n, max_steps=n)
    assert endpoint.xs.shape == ()
    assert jnp.array_equal(endpoint.xs, scalar.xs[-1])


def test_parity_rk4_grid_growth_field():
    # Growth-like field with a positivity clamp; dt = 1/16 is exactly
    # representable so the horizon clip never perturbs the step and the two
    # implementations must agree bit-for-bit.
    def f(k):
        return k**0.33 - 0.1 * k

    def project(k):
        return jnp.maximum(k, 1e-6)

    n, dt = 16, 1.0 / 16.0
    for x_0 in (jnp.asarray(1.0), jnp.asarray([0.5, 1.0, 2.0])):
        reference = rk4_grid(f, x_0, n, dt, project)
        sol = solve_ode(
            f,
            RK4(),
            0.0,
            n * dt,
            x_0,
            dt_0=dt,
            max_steps=n,
            save_at=SaveAt(steps=True),
            project=project,
        )
        assert bool(sol.ok)
        assert bool(jnp.all(sol.accepted))
        assert np.array_equal(np.asarray(sol.xs), np.asarray(reference))


def test_parity_rk4_grid_binding_clamp():
    # Strong decay drives intermediate stages below the clamp, so project
    # binds inside the stage evaluations; parity must still be exact.
    def f(y):
        return -5.0 * y

    def project(y):
        return jnp.maximum(y, 0.3)

    n, dt = 8, 0.25
    x_0 = jnp.asarray(1.0)
    reference = rk4_grid(f, x_0, n, dt, project)
    sol = solve_ode(
        f,
        RK4(),
        0.0,
        n * dt,
        x_0,
        dt_0=dt,
        max_steps=n,
        save_at=SaveAt(steps=True),
        project=project,
    )
    assert np.array_equal(np.asarray(sol.xs), np.asarray(reference))
    assert bool(jnp.all(sol.xs >= 0.3 - 1e-15))
