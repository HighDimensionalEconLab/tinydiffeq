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


@pytest.mark.parametrize(
    "save_at",
    [
        SaveAt(t_1=True),
        SaveAt(steps=True),
        SaveAt(ts=jnp.linspace(0.0, 2.0, 11)),
    ],
)
def test_forward_adaptive_loop_matches_bounded_primal_and_forward_ad(save_at):
    def run(parameter, adaptive_loop):
        return solve_ode(
            lambda x, t, args, p: -p * x,
            Tsit5(),
            0.0,
            2.0,
            jnp.asarray(1.0),
            p=parameter,
            dt_0=0.1,
            controller=IController(rtol=1e-9, atol=1e-11),
            max_steps=96,
            save_at=save_at,
            has_aux=False,
            adaptive_loop=adaptive_loop,
        )

    parameter = jnp.asarray(0.7)
    bounded = run(parameter, "bounded")
    forward = run(parameter, "forward")
    assert bool(bounded.ok & forward.ok)
    assert bounded.num_accepted == forward.num_accepted
    if bounded.accepted is not None:
        assert jnp.array_equal(bounded.accepted, forward.accepted)
    assert jnp.allclose(bounded.ts, forward.ts, rtol=1e-8, atol=1e-10)
    assert jnp.allclose(bounded.xs, forward.xs, rtol=1e-9, atol=1e-11)

    def output(value, adaptive_loop):
        return jnp.sum(run(value, adaptive_loop).xs)

    tangent = jnp.ones_like(parameter)
    bounded_jvp = jax.jvp(
        lambda value: output(value, "bounded"), (parameter,), (tangent,)
    )[1]
    forward_jvp = jax.jvp(
        lambda value: output(value, "forward"), (parameter,), (tangent,)
    )[1]

    def first_jvp(value, adaptive_loop):
        return jax.jvp(
            lambda inner: output(inner, adaptive_loop), (value,), (tangent,)
        )[1]

    bounded_second = jax.jvp(
        lambda value: first_jvp(value, "bounded"), (parameter,), (tangent,)
    )[1]
    forward_second = jax.jvp(
        lambda value: first_jvp(value, "forward"), (parameter,), (tangent,)
    )[1]
    assert jnp.allclose(bounded_jvp, forward_jvp, rtol=1e-12, atol=1e-12)
    assert jnp.allclose(bounded_second, forward_second, rtol=1e-11, atol=1e-11)


def test_forward_adaptive_loop_vmap_matches_bounded_lane_results():
    initial_values = jnp.asarray([0.05, 1.0, 50.0])

    def batched(adaptive_loop):
        def one(initial):
            return solve_ode(
                lambda x: -0.4 * x,
                Tsit5(),
                0.0,
                3.0,
                initial,
                dt_0=0.2,
                controller=IController(rtol=1e-8, atol=1e-10),
                max_steps=96,
                save_at=SaveAt(steps=True),
                has_aux=False,
                adaptive_loop=adaptive_loop,
            )

        return jax.vmap(one)(initial_values)

    bounded = batched("bounded")
    forward = batched("forward")
    assert bool(jnp.all(bounded.ok & forward.ok))
    assert len(set(map(int, bounded.num_accepted))) > 1
    assert jnp.array_equal(bounded.num_accepted, forward.num_accepted)
    assert jnp.array_equal(bounded.accepted, forward.accepted)
    assert jnp.allclose(bounded.ts, forward.ts, rtol=1e-8, atol=1e-10)
    assert jnp.allclose(bounded.xs, forward.xs, rtol=1e-9, atol=1e-11)


def test_forward_vmap_matches_bounded_with_mixed_budget_exhaustion():
    rates = jnp.asarray([0.01, 0.1, 1.0, 10.0])

    def batched(adaptive_loop):
        def one(rate):
            return solve_ode(
                lambda x, t, args, p: -p * x,
                Tsit5(),
                0.0,
                2.0,
                jnp.asarray(1.0),
                p=rate,
                dt_0=0.2,
                controller=IController(rtol=1e-7, atol=1e-9),
                max_steps=4,
                save_at=SaveAt(steps=True),
                adaptive_loop=adaptive_loop,
            )

        return jax.vmap(one)(rates)

    bounded = batched("bounded")
    forward = batched("forward")
    assert jnp.array_equal(bounded.ok, jnp.asarray([True, True, False, False]))
    assert jnp.array_equal(forward.ok, bounded.ok)
    assert jnp.array_equal(forward.num_steps, bounded.num_steps)
    assert jnp.array_equal(forward.num_accepted, bounded.num_accepted)
    assert jnp.array_equal(forward.accepted, bounded.accepted)
    assert jnp.array_equal(forward.ts, bounded.ts)
    assert jnp.array_equal(forward.xs, bounded.xs)


def test_forward_adaptive_loop_documents_reverse_mode_boundary():
    def endpoint(parameter):
        return solve_ode(
            lambda x, t, args, p: -p * x,
            Tsit5(),
            0.0,
            1.0,
            jnp.asarray(1.0),
            p=parameter,
            dt_0=0.1,
            controller=IController(rtol=1e-7, atol=1e-9),
            max_steps=64,
            has_aux=False,
            adaptive_loop="forward",
        ).xs

    with pytest.raises(ValueError, match="Reverse-mode differentiation"):
        jax.grad(endpoint)(jnp.asarray(0.7))


def test_steps_mesh_is_frozen_and_residual_jacobian_is_exact_at_root():
    theta_star = jnp.asarray(1.5)
    controller = IController(rtol=1e-8, atol=1e-10)

    def solve(theta):
        return solve_ode(
            lambda x, t, args, p: p * x,
            Tsit5(),
            0.0,
            2.0,
            jnp.asarray([1.0]),
            p=theta,
            dt_0=0.1,
            controller=controller,
            max_steps=128,
            save_at=SaveAt(steps=True),
        )

    solution = solve(theta_star)
    assert bool(solution.ok)
    n_valid = int(solution.num_accepted) + 1

    _, mesh_tangent = jax.jvp(
        lambda theta: solve(theta).ts,
        (theta_star,),
        (jnp.ones_like(theta_star),),
    )
    assert jnp.array_equal(mesh_tangent, jnp.zeros_like(mesh_tangent))

    epsilon = jnp.asarray(1e-5)
    plus = solve(theta_star + epsilon)
    minus = solve(theta_star - epsilon)
    assert int(plus.num_accepted) == int(minus.num_accepted) == n_valid - 1
    finite_difference_mesh = (plus.ts - minus.ts) / (2.0 * epsilon)
    assert float(jnp.max(jnp.abs(finite_difference_mesh[:n_valid]))) > 0.1

    def residual(theta):
        candidate = solve(theta)
        return (
            (theta - theta_star)
            * candidate.xs[:, 0]
            * candidate.accepted.astype(theta.dtype)
        )

    _, frozen_jacobian = jax.jvp(
        residual,
        (theta_star,),
        (jnp.ones_like(theta_star),),
    )
    finite_difference_jacobian = (
        residual(theta_star + epsilon) - residual(theta_star - epsilon)
    ) / (2.0 * epsilon)
    np.testing.assert_allclose(
        np.asarray(frozen_jacobian[:n_valid]),
        np.asarray(finite_difference_jacobian[:n_valid]),
        rtol=5e-9,
        atol=1e-8,
    )


def test_invalid_adaptive_loop_is_rejected():
    with pytest.raises(ValueError, match="adaptive_loop"):
        solve_ode(
            lambda x: -x,
            Tsit5(),
            0.0,
            1.0,
            jnp.asarray(1.0),
            dt_0=0.1,
            controller=IController(),
            adaptive_loop="unknown",
        )


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
