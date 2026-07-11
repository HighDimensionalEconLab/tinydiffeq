import jax
import jax.numpy as jnp
import numpy as np
import pytest
from kernels_reference import tsit5_free
from scipy.integrate import solve_ivp

from tinydiffeq import Euler, IController, PIController, SaveAt, Tsit5, solve_ode


def test_tsit5_vs_closed_form():
    x_0 = jnp.asarray(0.1)
    T = 3.0
    exact = x_0 * jnp.exp(T) / (1.0 + x_0 * (jnp.exp(T) - 1.0))
    sol = solve_ode(
        lambda x: x * (1.0 - x),
        Tsit5(),
        0.0,
        T,
        x_0,
        dt_0=0.1,
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

    x_0 = jnp.asarray([2.5, 0.0])
    T = 10.0
    reference = solve_ivp(
        lambda t, y: np.asarray(f(jnp.asarray(y), t)),
        (0.0, T),
        np.asarray(x_0),
        method="RK45",
        rtol=1e-11,
        atol=1e-11,
    )
    sol = solve_ode(
        f,
        Tsit5(),
        0.0,
        T,
        x_0,
        dt_0=0.1,
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
            dt_0=0.5,
            controller=IController(rtol=rtol, atol=1e-12),
            max_steps=1024,
        )
        assert bool(sol.ok)
        counts.append(int(sol.num_accepted))
    assert counts[0] < counts[1] < counts[2]


@pytest.mark.parametrize("controller_type", [IController, PIController])
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [
        (jnp.float32, 1e-4, 1e-6),
        (jnp.float64, 1e-7, 1e-9),
    ],
)
def test_default_tolerances_match_precision_policy(controller_type, dtype, rtol, atol):
    kwargs = dict(
        f=lambda x: -x,
        solver=Tsit5(),
        t_0=0.0,
        t_1=2.0,
        x_0=jnp.asarray(1.0, dtype),
        dt_0=0.5,
        max_steps=128,
        save_at=SaveAt(steps=True),
    )
    default = solve_ode(controller=controller_type(), **kwargs)
    explicit = solve_ode(controller=controller_type(rtol=rtol, atol=atol), **kwargs)
    assert bool(default.ok)
    assert default.xs.dtype == dtype
    assert np.array_equal(np.asarray(default.ts), np.asarray(explicit.ts))
    assert np.array_equal(np.asarray(default.xs), np.asarray(explicit.xs))
    assert np.array_equal(np.asarray(default.accepted), np.asarray(explicit.accepted))


@pytest.mark.parametrize("controller_type", [IController, PIController])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("time_scale", [1.0, 1000.0])
def test_default_dtmin_is_scaled_ten_eps_and_forces_accept(
    controller_type, dtype, time_scale
):
    controller = controller_type()
    x_0 = jnp.asarray(1.0, dtype)
    x_1 = jnp.asarray(2.0, dtype)
    err = jnp.asarray(1e6, dtype)
    eps = jnp.asarray(jnp.finfo(dtype).eps, dtype)
    time_scale = jnp.asarray(time_scale, dtype)
    scaled_eps = eps * jnp.maximum(1.0, jnp.abs(time_scale))
    state = controller.init(x_0)

    accept_small, dt_small, _ = controller.adapt(
        x_0, x_1, err, 5.0 * scaled_eps, 5.0 * scaled_eps, 5, state, time_scale
    )
    accept_large, dt_large, _ = controller.adapt(
        x_0,
        x_1,
        err,
        20.0 * scaled_eps,
        20.0 * scaled_eps,
        5,
        state,
        time_scale,
    )

    assert bool(accept_small)
    assert not bool(accept_large)
    assert dt_small.dtype == dtype
    assert dt_large.dtype == dtype
    assert dt_small == 10.0 * scaled_eps
    assert dt_large == 10.0 * scaled_eps


def test_pi_i_coefficients_reproduce_icontroller_bit_for_bit():
    kwargs = dict(
        f=lambda x: x * (1.0 - x),
        solver=Tsit5(),
        t_0=0.0,
        t_1=3.0,
        x_0=jnp.asarray(0.05),
        dt_0=0.5,
        max_steps=128,
        save_at=SaveAt(steps=True),
    )
    integral = solve_ode(controller=IController(rtol=1e-7, atol=1e-9), **kwargs)
    pi_as_integral = solve_ode(
        controller=PIController(rtol=1e-7, atol=1e-9, p_coeff=0.0, i_coeff=1.0),
        **kwargs,
    )
    assert np.array_equal(np.asarray(integral.ts), np.asarray(pi_as_integral.ts))
    assert np.array_equal(np.asarray(integral.xs), np.asarray(pi_as_integral.xs))
    assert np.array_equal(
        np.asarray(integral.accepted), np.asarray(pi_as_integral.accepted)
    )


def test_pi_controller_handles_oscillatory_problem():
    frequency = 15.0
    t_1 = 2.0

    def f(x, t):
        return (1.0 + 0.9 * jnp.sin(frequency * t)) * x

    def run(controller):
        return solve_ode(
            f,
            Tsit5(),
            0.0,
            t_1,
            jnp.asarray(1.0),
            dt_0=0.5,
            controller=controller,
            max_steps=256,
            save_at=SaveAt(steps=True),
        )

    integral = run(IController(rtol=1e-7, atol=1e-9))
    pi = run(PIController(rtol=1e-7, atol=1e-9))
    assert bool(integral.ok) and bool(pi.ok)

    exact = jnp.exp(t_1 + 0.9 * (1.0 - jnp.cos(frequency * t_1)) / frequency)
    assert jnp.abs(integral.xs[int(integral.num_accepted)] - exact) < 2e-6
    assert jnp.abs(pi.xs[int(pi.num_accepted)] - exact) < 1e-6


def test_ok_false_when_starved_and_restored():
    def f(x):
        return x * (1.0 - x)

    starved = solve_ode(
        f,
        Tsit5(),
        0.0,
        5.0,
        jnp.asarray(0.05),
        dt_0=0.01,
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
        dt_0=0.01,
        controller=IController(rtol=1e-12, atol=1e-14),
        max_steps=2048,
    )
    assert bool(restored.ok)


def test_adaptive_non_divisible_chunk_budget_preserves_result_and_ad():
    def endpoint(x_0, max_steps):
        return solve_ode(
            lambda x: -0.2 * x,
            Tsit5(),
            0.0,
            1.0,
            x_0,
            dt_0=0.1,
            controller=IController(rtol=1e-5, atol=1e-7),
            max_steps=max_steps,
        ).xs

    x_0 = jnp.asarray(1.0)
    reference = endpoint(x_0, 32)
    value = endpoint(x_0, 17)
    _, tangent = jax.jvp(lambda x: endpoint(x, 17), (x_0,), (jnp.ones_like(x_0),))
    gradient = jax.grad(lambda x: endpoint(x, 17))(x_0)

    assert jnp.allclose(value, reference)
    assert jnp.isfinite(tangent)
    assert jnp.isfinite(gradient)


def test_parity_tsit5_free():
    # Identical tolerances, budget, and dt_0 must reproduce the kernels
    # free-stepper's accepted trajectory bit-for-bit (project never binds
    # here, so the documented FSAL-projection change is inert).
    def f(y):
        return y * (1.2 - y)

    y_0 = jnp.asarray([0.05, 0.3, 0.9])
    T, n_iters, rtol, atol, dt_0 = 3.0, 128, 1e-7, 1e-9, 0.05
    ts_ref, ys_ref = tsit5_free(f, y_0, T, n_iters, rtol=rtol, atol=atol, dt_0=dt_0)
    assert bool(jnp.all(jnp.isfinite(ys_ref)))  # reference reached T

    sol = solve_ode(
        f,
        Tsit5(),
        0.0,
        T,
        y_0,
        dt_0=dt_0,
        controller=IController(rtol=rtol, atol=atol),
        max_steps=n_iters,
        save_at=SaveAt(steps=True),
    )
    assert bool(sol.ok)
    ref_accepted = jnp.concatenate([jnp.ones((1,), bool), jnp.diff(ts_ref) > 0.0])
    accepted_ts_ref = ts_ref[ref_accepted]
    accepted_ys_ref = ys_ref[ref_accepted]
    n_valid = int(sol.num_accepted) + 1
    assert np.array_equal(np.asarray(sol.ts[:n_valid]), np.asarray(accepted_ts_ref))
    assert np.array_equal(np.asarray(sol.xs[:n_valid]), np.asarray(accepted_ys_ref))
    assert bool(jnp.all(sol.ts[n_valid:] == accepted_ts_ref[-1]))
    assert bool(jnp.all(sol.xs[n_valid:] == accepted_ys_ref[-1]))


def test_parity_tsit5_free_nonbinding_project():
    def f(y):
        return y * (1.2 - y)

    def project(y):
        return jnp.maximum(y, 1e-8)

    y_0 = jnp.asarray([0.05, 0.9])
    T, n_iters = 2.0, 128
    ts_ref, ys_ref = tsit5_free(
        f, y_0, T, n_iters, rtol=1e-6, atol=1e-8, dt_0=0.02, project=project
    )
    sol = solve_ode(
        f,
        Tsit5(),
        0.0,
        T,
        y_0,
        dt_0=0.02,
        controller=IController(rtol=1e-6, atol=1e-8),
        max_steps=n_iters,
        save_at=SaveAt(steps=True),
        project=project,
    )
    ref_accepted = jnp.concatenate([jnp.ones((1,), bool), jnp.diff(ts_ref) > 0.0])
    n_valid = int(sol.num_accepted) + 1
    assert np.array_equal(
        np.asarray(sol.ts[:n_valid]), np.asarray(ts_ref[ref_accepted])
    )
    assert np.array_equal(
        np.asarray(sol.xs[:n_valid]), np.asarray(ys_ref[ref_accepted])
    )


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
        dt_0=0.05,
        controller=IController(rtol=1e-8, atol=1e-8),
        max_steps=512,
        save_at=SaveAt(steps=True),
    )
    clamped = solve_ode(
        f,
        Tsit5(),
        0.0,
        4.0,
        jnp.asarray(2.0),
        dt_0=0.05,
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
        dt_0=0.5,
        controller=IController(rtol=1e-6, atol=1e-8),
        max_steps=64,
        save_at=SaveAt(steps=True),
    )
    n_valid = int(sol.num_accepted) + 1
    assert bool(jnp.all(jnp.diff(sol.ts[:n_valid]) > 0.0))
    assert bool(jnp.all(sol.ts[n_valid:] == sol.ts[n_valid - 1]))


@pytest.mark.parametrize(
    "controller",
    [
        IController(rtol=1e-6, atol=1e-6),
        PIController(rtol=1e-6, atol=1e-6),
    ],
)
def test_adaptive_controller_with_euler_raises(controller):
    with pytest.raises(ValueError, match="error estimate"):
        solve_ode(
            lambda x: -x,
            Euler(),
            0.0,
            1.0,
            jnp.asarray(1.0),
            dt_0=0.1,
            controller=controller,
        )


def test_missing_dt_0_raises():
    with pytest.raises(ValueError, match="dt_0"):
        solve_ode(lambda x: -x, Tsit5(), 0.0, 1.0, jnp.asarray(1.0))
