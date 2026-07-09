import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import numpy as np
from kernels_reference import rk4_grid

from tinydiffeq import RK4, Euler, SaveAt, solve_ode


def logistic_exact(x0, t):
    return x0 * jnp.exp(t) / (1.0 + x0 * (jnp.exp(t) - 1.0))


def test_linear_system_vs_expm():
    A = jnp.asarray([[0.0, 1.0], [-1.0, -0.3]])
    x0 = jnp.asarray([1.0, 0.5])
    T = 2.0
    exact = jsp_linalg.expm(A * T) @ x0

    def f(x):
        return A @ x

    n = 2000
    euler = solve_ode(f, Euler(), 0.0, T, x0, dt0=T / n, max_steps=n)
    rk4 = solve_ode(f, RK4(), 0.0, T, x0, dt0=T / n, max_steps=n)
    assert bool(euler.ok) and bool(rk4.ok)
    assert jnp.max(jnp.abs(euler.xs - exact)) < 2e-3
    assert jnp.max(jnp.abs(rk4.xs - exact)) < 1e-12


def test_logistic_closed_form():
    x0 = jnp.asarray(0.1)
    T = 3.0
    n = 300
    sol = solve_ode(lambda x: x * (1.0 - x), RK4(), 0.0, T, x0, dt0=T / n, max_steps=n)
    assert jnp.abs(sol.xs - logistic_exact(x0, T)) < 1e-9


def test_convergence_slopes():
    x0 = jnp.asarray(0.1)
    T = 2.0
    exact = logistic_exact(x0, T)

    def f(x):
        return x * (1.0 - x)

    for solver, expected in ((Euler(), 1.0), (RK4(), 4.0)):
        errors, dts = [], []
        for n in (20, 40, 80, 160):
            sol = solve_ode(f, solver, 0.0, T, x0, dt0=T / n, max_steps=n)
            assert bool(sol.ok)
            errors.append(float(jnp.abs(sol.xs - exact)))
            dts.append(T / n)
        slope = np.polyfit(np.log(dts), np.log(errors), 1)[0]
        assert abs(slope - expected) < 0.3, (type(solver).__name__, slope)


def test_non_dividing_dt0_lands_on_t1():
    x0 = jnp.asarray(0.1)
    sol = solve_ode(lambda x: x * (1.0 - x), RK4(), 0.0, 1.0, x0, dt0=0.3, max_steps=4)
    assert bool(sol.ok)
    assert sol.ts == 1.0
    assert int(sol.num_accepted) == 4
    assert jnp.abs(sol.xs - logistic_exact(x0, 1.0)) < 1e-4


def test_scalar_vs_vector_shapes():
    def f(x):
        return -x

    n = 8
    scalar = solve_ode(
        f, RK4(), 0.0, 1.0, 1.0, dt0=1.0 / n, max_steps=n, saveat=SaveAt(steps=True)
    )
    vector = solve_ode(
        f,
        RK4(),
        0.0,
        1.0,
        jnp.asarray([1.0, 2.0]),
        dt0=1.0 / n,
        max_steps=n,
        saveat=SaveAt(steps=True),
    )
    assert scalar.xs.shape == (n + 1,)
    assert vector.xs.shape == (n + 1, 2)
    endpoint = solve_ode(f, RK4(), 0.0, 1.0, 1.0, dt0=1.0 / n, max_steps=n)
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
    for x0 in (jnp.asarray(1.0), jnp.asarray([0.5, 1.0, 2.0])):
        reference = rk4_grid(f, x0, n, dt, project)
        sol = solve_ode(
            f,
            RK4(),
            0.0,
            n * dt,
            x0,
            dt0=dt,
            max_steps=n,
            saveat=SaveAt(steps=True),
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
    x0 = jnp.asarray(1.0)
    reference = rk4_grid(f, x0, n, dt, project)
    sol = solve_ode(
        f,
        RK4(),
        0.0,
        n * dt,
        x0,
        dt0=dt,
        max_steps=n,
        saveat=SaveAt(steps=True),
        project=project,
    )
    assert np.array_equal(np.asarray(sol.xs), np.asarray(reference))
    assert bool(jnp.all(sol.xs >= 0.3 - 1e-15))
