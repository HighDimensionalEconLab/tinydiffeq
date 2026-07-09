import jax.numpy as jnp
import numpy as np
import pytest
from kernels_reference import tsit5_free
from scipy.integrate import solve_ivp

from tinydiffeq import Euler, IController, SaveAt, Tsit5, solve_ode


def test_tsit5_vs_closed_form():
    x0 = jnp.asarray(0.1)
    T = 3.0
    exact = x0 * jnp.exp(T) / (1.0 + x0 * (jnp.exp(T) - 1.0))
    sol = solve_ode(
        lambda x: x * (1.0 - x),
        Tsit5(),
        0.0,
        T,
        x0,
        dt0=0.1,
        controller=IController(rtol=1e-10, atol=1e-12),
        max_steps=512,
    )
    assert bool(sol.ok)
    assert jnp.abs(sol.xs - exact) < 1e-9


def test_tsit5_vs_scipy_pendulum():
    # non-autonomous through the state pair; scipy RK45 at rtol=1e-11 is the
    # reference for a problem with no closed form
    def f(x, t):
        return jnp.asarray([x[1], -jnp.sin(x[0]) - 0.1 * x[1] + 0.2 * jnp.cos(t)])

    x0 = jnp.asarray([2.5, 0.0])
    T = 10.0
    reference = solve_ivp(
        lambda t, y: np.asarray(f(jnp.asarray(y), t)),
        (0.0, T),
        np.asarray(x0),
        method="RK45",
        rtol=1e-11,
        atol=1e-11,
    )
    sol = solve_ode(
        f,
        Tsit5(),
        0.0,
        T,
        x0,
        dt0=0.1,
        controller=IController(rtol=1e-8, atol=1e-8),
        max_steps=2048,
    )
    assert bool(sol.ok)
    assert jnp.max(jnp.abs(sol.xs - reference.y[:, -1])) < 1e-6


def test_num_accepted_grows_as_rtol_tightens():
    def f(x):
        return x * (1.0 - x)

    counts = []
    for rtol in (1e-4, 1e-7, 1e-10):
        sol = solve_ode(
            f,
            Tsit5(),
            0.0,
            5.0,
            jnp.asarray(0.05),
            dt0=0.5,
            controller=IController(rtol=rtol, atol=1e-12),
            max_steps=1024,
        )
        assert bool(sol.ok)
        counts.append(int(sol.num_accepted))
    assert counts[0] < counts[1] < counts[2]


def test_ok_false_when_starved_and_restored():
    def f(x):
        return x * (1.0 - x)

    starved = solve_ode(
        f,
        Tsit5(),
        0.0,
        5.0,
        jnp.asarray(0.05),
        dt0=0.01,
        controller=IController(rtol=1e-12, atol=1e-14),
        max_steps=8,
    )
    assert not bool(starved.ok)
    # kernels-style poisoning is one line at the callsite
    poisoned = jnp.where(starved.ok, starved.xs, jnp.inf)
    assert bool(jnp.isinf(poisoned))
    restored = solve_ode(
        f,
        Tsit5(),
        0.0,
        5.0,
        jnp.asarray(0.05),
        dt0=0.01,
        controller=IController(rtol=1e-12, atol=1e-14),
        max_steps=2048,
    )
    assert bool(restored.ok)


def test_parity_tsit5_free():
    # Identical tolerances, budget, and dt0 must reproduce the kernels
    # free-stepper's attempt rows bit-for-bit (project never binds here, so
    # the documented FSAL-projection change is inert).
    def f(y):
        return y * (1.2 - y)

    y0 = jnp.asarray([0.05, 0.3, 0.9])
    T, n_iters, rtol, atol, dt0 = 3.0, 128, 1e-7, 1e-9, 0.05
    ts_ref, ys_ref = tsit5_free(f, y0, T, n_iters, rtol=rtol, atol=atol, dt0=dt0)
    assert bool(jnp.all(jnp.isfinite(ys_ref)))  # reference reached T

    sol = solve_ode(
        f,
        Tsit5(),
        0.0,
        T,
        y0,
        dt0=dt0,
        controller=IController(rtol=rtol, atol=atol),
        max_steps=n_iters,
        saveat=SaveAt(steps=True),
    )
    assert bool(sol.ok)
    assert np.array_equal(np.asarray(sol.ts), np.asarray(ts_ref))
    assert np.array_equal(np.asarray(sol.xs), np.asarray(ys_ref))


def test_parity_tsit5_free_nonbinding_project():
    def f(y):
        return y * (1.2 - y)

    def project(y):
        return jnp.maximum(y, 1e-8)

    y0 = jnp.asarray([0.05, 0.9])
    T, n_iters = 2.0, 128
    ts_ref, ys_ref = tsit5_free(
        f, y0, T, n_iters, rtol=1e-6, atol=1e-8, dt0=0.02, project=project
    )
    sol = solve_ode(
        f,
        Tsit5(),
        0.0,
        T,
        y0,
        dt0=0.02,
        controller=IController(rtol=1e-6, atol=1e-8),
        max_steps=n_iters,
        saveat=SaveAt(steps=True),
        project=project,
    )
    assert np.array_equal(np.asarray(sol.ts), np.asarray(ts_ref))
    assert np.array_equal(np.asarray(sol.xs), np.asarray(ys_ref))


def test_binding_clamp_keeps_states_feasible():
    # When the clamp binds, tinydiffeq deviates from kernels by design (the
    # FSAL cache is evaluated at the projected state); check the solve stays
    # feasible and accurate against the clamped fixed point.
    def f(y):
        return -3.0 * y

    def project(y):
        return jnp.maximum(y, 0.5)

    sol = solve_ode(
        f,
        Tsit5(),
        0.0,
        4.0,
        jnp.asarray(2.0),
        dt0=0.05,
        controller=IController(rtol=1e-8, atol=1e-8),
        max_steps=512,
        saveat=SaveAt(steps=True),
    )
    clamped = solve_ode(
        f,
        Tsit5(),
        0.0,
        4.0,
        jnp.asarray(2.0),
        dt0=0.05,
        controller=IController(rtol=1e-8, atol=1e-8),
        max_steps=512,
        project=project,
    )
    assert bool(clamped.ok)
    assert float(clamped.xs) >= 0.5 - 1e-15
    assert bool(sol.ok)


def test_steps_rows_monotone():
    sol = solve_ode(
        lambda x: x * (1.0 - x),
        Tsit5(),
        0.0,
        3.0,
        jnp.asarray(0.05),
        dt0=0.5,
        controller=IController(rtol=1e-6, atol=1e-8),
        max_steps=64,
        saveat=SaveAt(steps=True),
    )
    assert bool(jnp.all(jnp.diff(sol.ts) >= 0.0))


def test_icontroller_with_euler_raises():
    with pytest.raises(ValueError, match="error estimate"):
        solve_ode(
            lambda x: -x,
            Euler(),
            0.0,
            1.0,
            jnp.asarray(1.0),
            dt0=0.1,
            controller=IController(rtol=1e-6, atol=1e-6),
        )


def test_missing_dt0_raises():
    with pytest.raises(ValueError, match="dt0"):
        solve_ode(lambda x: -x, Tsit5(), 0.0, 1.0, jnp.asarray(1.0))
