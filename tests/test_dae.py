import jax
import jax.numpy as jnp
import pytest
from nlls_gram import LU, QR, Cholesky

from tinydiffeq import (
    RK4,
    IController,
    LMRootSolver,
    Rodas5P,
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


def test_lm_root_solver_uses_nlls_defaults_and_forwards_options():
    def constraint(y, z, t, args, p):
        return z - y

    defaults = _build_algebraic_solver(constraint, LMRootSolver(), False)
    # Everything algorithmic is nlls-gram's default; only the two invariants
    # this package owns are pinned.
    assert isinstance(defaults.linear_solver, Cholesky)
    assert defaults.jacobian_mode == "auto"
    assert defaults.ad_solver is None
    assert not defaults.cache_jacobian
    assert not defaults.geodesic_acceleration

    configured = _build_algebraic_solver(
        constraint,
        LMRootSolver(
            solver_options={
                "init_damping": 2e-3,
                "damping_decrease": 0.4,
                "damping_increase": 3.0,
                "jacobian_mode": "rev",
                "linear_solver": QR(),
            }
        ),
        False,
    )
    assert configured.init_damping == 2e-3
    assert configured.damping_decrease == 0.4
    assert configured.damping_increase == 3.0
    assert configured.jacobian_mode == "rev"
    assert isinstance(configured.linear_solver, QR)
    # The invariants survive a populated pass-through.
    assert not configured.cache_jacobian
    assert not configured.geodesic_acceleration

    direct_ad = _build_algebraic_solver(
        constraint,
        LMRootSolver(solver_options={"ad_solver": LU()}),
        False,
    )
    assert isinstance(direct_ad.ad_solver, LU)


def test_lm_root_solver_options_normalize_and_reject_fixed_keys():
    # A mapping and the equivalent pairs must compare and hash equal, so
    # _cached_algebraic_solver shares one compiled solver between them.
    mapping = LMRootSolver(solver_options={"jacobian_mode": "rev", "init_damping": 0.1})
    pairs = LMRootSolver(
        solver_options=(("init_damping", 0.1), ("jacobian_mode", "rev"))
    )
    assert mapping == pairs
    assert hash(mapping) == hash(pairs)
    # Re-wrapping an already-normalized value is idempotent.
    assert LMRootSolver(solver_options=mapping.solver_options) == mapping

    for fixed in ("cache_jacobian", "geodesic_acceleration"):
        with pytest.raises(ValueError, match=fixed):
            LMRootSolver(solver_options={fixed: True})
    with pytest.raises(TypeError, match="mapping or key/value pairs"):
        LMRootSolver(solver_options=5)
    with pytest.raises(TypeError, match="predictor must be a string"):
        LMRootSolver(predictor=1)
    with pytest.raises(ValueError, match="predictor must be either"):
        LMRootSolver(predictor="quadratic")


def test_max_steps_policy_requires_root_residual_and_strict_batched_derivative():
    def one_lane(z_0, p):
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
            root_solver=LMRootSolver(max_steps=1, atol=1e-12),
        )

    p = jnp.asarray(1.0)
    starved = one_lane(jnp.asarray(-1e6), p)
    strict = one_lane(jnp.asarray(0.0), p)
    assert not bool(starved.ok)
    assert not bool(strict.ok)
    assert int(starved.num_root_solves) == 1
    assert int(starved.num_root_steps) == 1
    assert starved.zs == jnp.asarray(-1e6)

    def strict_batch(parameter):
        return jax.vmap(lambda z_0: one_lane(z_0, parameter))(jnp.asarray([1.0, 0.0]))

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


def test_dae_roots_reject_nonresidual_stopping_rules():
    with pytest.raises(ValueError, match="gtol must be zero"):
        LMRootSolver(gtol=1e-6)
    with pytest.raises(ValueError, match="xtol must be zero"):
        LMRootSolver(xtol=1e-6)


def test_root_budget_exhaustion_is_never_a_differentiable_root():
    guesses = jnp.asarray([1.0, 0.0])

    def solve_one(parameter, guess):
        return solve_semi_explicit_dae(
            lambda y, z: jnp.zeros_like(y),
            lambda y, z, t, args, p: z**2 + p,
            RK4(),
            0.0,
            0.1,
            jnp.asarray(0.0),
            guess,
            p=parameter,
            dt_0=0.1,
            max_steps=1,
            root_solver=LMRootSolver(max_steps=1, atol=1e-12),
        )

    def endpoints(parameters):
        return jax.vmap(solve_one)(parameters, guesses).zs

    parameters = jnp.asarray([-1.0, 1.0])
    solutions = jax.vmap(solve_one)(parameters, guesses)
    value, tangent = jax.jvp(
        endpoints,
        (parameters,),
        (jnp.ones_like(parameters),),
    )
    _, pullback = jax.vjp(endpoints, parameters)
    expected_derivative = jnp.asarray([-0.5, 0.0])
    assert jnp.array_equal(solutions.ok, jnp.asarray([True, False]))
    assert jnp.array_equal(value, guesses)
    assert jnp.array_equal(tangent, expected_derivative)
    assert jnp.array_equal(pullback(jnp.ones_like(value))[0], expected_derivative)
    assert jnp.array_equal(
        jax.grad(lambda p: jnp.sum(endpoints(p)))(parameters), expected_derivative
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
    assert sol.num_root_solves.dtype == jnp.int32
    assert int(sol.num_root_solves) == 1 + 4 * int(sol.num_steps)
    assert 0 <= int(sol.num_root_steps) <= 8 * int(sol.num_root_solves)
    assert jnp.abs(sol.ys - jnp.e) < 2e-5
    assert jnp.abs(sol.zs - sol.ys) < 2e-6


@pytest.mark.parametrize("solver", [RK4(), Rodas5P()])
def test_fixed_dae_large_negative_initial_time_snaps_to_horizon(solver):
    # The float32 nominal fifth time is 0.99975586. Its roundoff is set by
    # |t_0|, not |t_1|; the local tolerance should snap it without changing h
    # appreciably or depending on the attempt budget.
    t_0, t_1, num_steps = -10_000.0, 1.0, 5
    solution = solve_semi_explicit_dae(
        lambda y, z: jnp.zeros_like(y),
        identity_constraint,
        solver,
        t_0,
        t_1,
        jnp.asarray(1.0, jnp.float32),
        jnp.asarray(1.0, jnp.float32),
        dt_0=(t_1 - t_0) / num_steps,
        max_steps=num_steps,
        root_solver=LMRootSolver(atol=1e-6),
        save_at=SaveAt(steps=True),
    )
    assert bool(solution.ok)
    assert solution.ts[-1] == jnp.asarray(t_1, jnp.float32)
    assert int(solution.num_steps) == int(solution.num_accepted) == num_steps


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
    assert int(sol.num_root_solves) == 1 + 6 * int(sol.num_steps)
    assert 0 <= int(sol.num_root_steps) <= 8 * int(sol.num_root_solves)
    assert sol.ts.shape == (65,)
    assert int(sol.accepted.sum()) == int(sol.num_accepted) + 1
    assert bool(jnp.all(sol.ts[1:] >= sol.ts[:-1]))
    assert jnp.max(jnp.abs((sol.zs - sol.ys)[sol.accepted])) < 2e-6
    assert jnp.abs(sol.ys[-1] - jnp.exp(2.0)) < 2e-4


def test_dae_rejects_ode_only_exact_save_mode():
    with pytest.raises(ValueError, match="only supported by solve_ode"):
        solve_linear(
            jnp.asarray(1.0),
            jnp.asarray(0.0),
            jnp.asarray(1.0),
            RK4(),
            SaveAt(ts=jnp.asarray([0.0, 0.15, 1.0]), exact=True),
            dt_0=0.1,
            max_steps=10,
        )


@pytest.mark.parametrize("predictor", ["previous", "secant"])
def test_forward_adaptive_dae_loop_matches_bounded_and_supports_forward_ad(predictor):
    def run(parameter, adaptive_loop):
        return solve_linear(
            parameter,
            jnp.asarray(0.7),
            jnp.asarray(1.0),
            Tsit5(),
            SaveAt(steps=True),
            dt_0=0.1,
            max_steps=96,
            controller=IController(rtol=1e-7, atol=1e-9),
            root_solver=LMRootSolver(atol=1e-10, predictor=predictor),
            adaptive_loop=adaptive_loop,
        )

    parameter = jnp.asarray(1.3)
    bounded = run(parameter, "bounded")
    forward = run(parameter, "forward")
    assert bool(bounded.ok & forward.ok)
    assert bounded.num_accepted == forward.num_accepted
    assert bounded.num_steps == forward.num_steps
    assert bounded.num_root_solves == forward.num_root_solves
    assert bounded.num_root_steps == forward.num_root_steps
    assert jnp.array_equal(bounded.accepted, forward.accepted)
    assert jnp.allclose(bounded.ts, forward.ts, rtol=2e-7, atol=2e-9)
    assert jnp.allclose(bounded.ys, forward.ys, rtol=2e-7, atol=2e-9)
    assert jnp.allclose(bounded.zs, forward.zs, rtol=2e-7, atol=2e-9)

    tangent = jnp.ones_like(parameter)

    def endpoint(value, adaptive_loop):
        return run(value, adaptive_loop).ys[-1]

    bounded_jvp = jax.jvp(
        lambda value: endpoint(value, "bounded"), (parameter,), (tangent,)
    )[1]
    forward_jvp = jax.jvp(
        lambda value: endpoint(value, "forward"), (parameter,), (tangent,)
    )[1]

    def first_jvp(value, adaptive_loop):
        return jax.jvp(
            lambda inner: endpoint(inner, adaptive_loop), (value,), (tangent,)
        )[1]

    bounded_second = jax.jvp(
        lambda value: first_jvp(value, "bounded"), (parameter,), (tangent,)
    )[1]
    forward_second = jax.jvp(
        lambda value: first_jvp(value, "forward"), (parameter,), (tangent,)
    )[1]
    assert jnp.allclose(bounded_jvp, forward_jvp, rtol=2e-6, atol=2e-8)
    assert jnp.allclose(bounded_second, forward_second, rtol=2e-5, atol=2e-7)


def test_forward_adaptive_dae_loop_documents_reverse_boundary():
    def endpoint(parameter):
        return solve_linear(
            parameter,
            jnp.asarray(0.7),
            jnp.asarray(1.0),
            Tsit5(),
            SaveAt(t_1=True),
            dt_0=0.1,
            max_steps=64,
            controller=IController(rtol=1e-6, atol=1e-8),
            adaptive_loop="forward",
        ).ys

    with pytest.raises(ValueError, match="Reverse-mode differentiation"):
        jax.grad(endpoint)(jnp.asarray(1.3))


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
        sol = solve_linear(
            p,
            y_0,
            y_0,
            Tsit5(),
            SaveAt(t_1=True),
            dt_0=0.1,
            max_steps=64,
            controller=IController(rtol=1e-5, atol=1e-7),
        )
        return sol.ys, sol.num_root_solves, sol.num_root_steps

    ps = jnp.asarray([0.5, 1.0, 1.5])
    y_0s = jnp.asarray([0.7, 1.0, 1.3])
    got, root_solves, root_steps = jax.vmap(endpoint)(ps, y_0s)
    assert jnp.max(jnp.abs(got - y_0s * jnp.exp(ps))) < 2e-4
    assert root_solves.shape == root_steps.shape == ps.shape
    assert root_solves.dtype == root_steps.dtype == jnp.int32
    scalar_stats = jnp.stack(
        [jnp.stack(endpoint(p, y_0)[1:]) for p, y_0 in zip(ps, y_0s, strict=True)]
    )
    assert jnp.array_equal(root_solves, scalar_stats[:, 0])
    assert jnp.array_equal(root_steps, scalar_stats[:, 1])


def test_root_statistics_are_ad_inert_without_changing_state_jvp():
    def full_solution(parameter):
        return solve_linear(
            parameter,
            jnp.asarray(1.0),
            jnp.asarray(1.0),
            RK4(),
            SaveAt(t_1=True),
            dt_0=0.05,
            max_steps=20,
        )

    parameter = jnp.asarray(0.4)
    _, tangent = jax.jvp(full_solution, (parameter,), (jnp.ones_like(parameter),))
    assert jnp.abs(tangent.ys - jnp.exp(parameter)) < 2e-5
    assert tangent.num_root_solves.dtype == jax.dtypes.float0
    assert tangent.num_root_steps.dtype == jax.dtypes.float0


@pytest.mark.parametrize(("solver", "roots_per_step"), [(RK4(), 4), (Tsit5(), 6)])
def test_secant_predictor_preserves_explicit_solution_and_root_call_count(
    solver, roots_per_step
):
    def run(predictor):
        return solve_linear(
            jnp.asarray(0.4),
            jnp.asarray(1.0),
            jnp.asarray(1.0),
            solver,
            SaveAt(t_1=True),
            dt_0=0.1,
            max_steps=10,
            root_solver=LMRootSolver(atol=1e-10, predictor=predictor),
        )

    previous = run("previous")
    secant = run("secant")
    assert bool(previous.ok & secant.ok)
    assert int(previous.num_root_solves) == int(secant.num_root_solves)
    assert int(secant.num_root_solves) == 1 + roots_per_step * 10
    assert int(secant.num_root_steps) <= int(previous.num_root_steps)
    assert jnp.allclose(secant.ys, previous.ys, rtol=1e-9, atol=1e-10)
    assert jnp.allclose(secant.zs, previous.zs, rtol=1e-9, atol=1e-10)


def test_secant_predictor_preserves_mixed_differential_algebraic_dtypes():
    y_0 = jnp.asarray(1.0, dtype=jnp.float64)
    z_0 = jnp.asarray(1.0, dtype=jnp.float32)

    sol = solve_semi_explicit_dae(
        lambda y, z: z.astype(y.dtype),
        lambda y, z: z - y.astype(z.dtype),
        RK4(),
        0.0,
        0.2,
        y_0,
        z_0,
        dt_0=0.1,
        max_steps=2,
        root_solver=LMRootSolver(atol=1e-6, predictor="secant"),
        save_at=SaveAt(steps=True),
    )

    assert bool(sol.ok)
    assert sol.ys.dtype == jnp.float64
    assert sol.zs.dtype == jnp.float32


@pytest.mark.parametrize("predictor", ["previous", "secant"])
def test_explicit_root_predictor_preserves_implicit_jvp_and_vjp(predictor):
    def endpoint(parameter):
        return solve_linear(
            parameter,
            jnp.asarray(1.0),
            jnp.asarray(1.0),
            Tsit5(),
            SaveAt(t_1=True),
            dt_0=0.1,
            max_steps=10,
            root_solver=LMRootSolver(atol=1e-10, predictor=predictor),
        ).ys

    parameter = jnp.asarray(0.4)
    expected = jnp.exp(parameter)
    tangent = jax.jvp(endpoint, (parameter,), (jnp.ones_like(parameter),))[1]
    cotangent = jax.grad(endpoint)(parameter)
    assert jnp.allclose(tangent, expected, rtol=2e-8, atol=2e-10)
    assert jnp.allclose(cotangent, expected, rtol=2e-8, atol=2e-10)


def test_secant_predictor_continues_a_locally_unique_positive_root():
    def run(predictor):
        return solve_semi_explicit_dae(
            lambda y, z: -0.2 * z,
            lambda y, z: z**2 - y - 2.0,
            Tsit5(),
            0.0,
            1.0,
            jnp.asarray(1.0),
            jnp.sqrt(jnp.asarray(3.0)),
            dt_0=0.1,
            controller=IController(rtol=1e-8, atol=1e-10),
            root_solver=LMRootSolver(atol=1e-10, predictor=predictor),
            max_steps=64,
            save_at=SaveAt(steps=True),
        )

    previous = run("previous")
    secant = run("secant")
    assert bool(previous.ok & secant.ok)
    assert jnp.all(secant.zs[secant.accepted] > 0.0)
    assert int(secant.num_root_steps) <= int(previous.num_root_steps)
    assert jnp.allclose(
        secant.ys[secant.accepted],
        previous.ys[previous.accepted],
        rtol=2e-9,
        atol=2e-10,
    )


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
        root_solver=LMRootSolver(),
    )
    assert not bool(failed_root.ok)
    assert int(failed_root.num_accepted) == 0
    assert int(failed_root.num_root_solves) == 1
    assert 0 <= int(failed_root.num_root_steps) <= 8

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
    assert int(starved.num_root_solves) == 1 + 4 * int(starved.num_steps)
    assert jnp.abs(starved.ts - 0.3) < 1e-7


@pytest.mark.parametrize("predictor", ["previous", "secant"])
def test_adaptive_stage_root_failure_retries_with_smaller_step(predictor):
    # One damped LM iteration cannot meet the root tolerance at dt_0, but it
    # can after the adaptive controller reduces the step. A fixed controller
    # would terminate on the same stage-root failure.
    sol = solve_semi_explicit_dae(
        lambda y, z: jnp.ones_like(y),
        identity_constraint,
        Tsit5(),
        0.0,
        0.0001,
        jnp.asarray(0.0),
        jnp.asarray(0.0),
        dt_0=0.0001,
        controller=IController(),
        root_solver=LMRootSolver(
            max_steps=1,
            atol=1e-8,
            predictor=predictor,
        ),
        max_steps=64,
        save_at=SaveAt(steps=True),
    )
    assert bool(sol.ok)
    assert int(sol.num_accepted) > 1
    assert int(sol.num_steps) > int(sol.num_accepted)
    assert jnp.abs(sol.ys[-1] - 0.0001) < 1e-9
    assert jnp.abs(sol.zs[-1] - sol.ys[-1]) < 1e-8


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
            root_solver=LMRootSolver(),
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
    with pytest.raises(ValueError, match="atol must be positive or None"):
        LMRootSolver(atol=0.0)
    with pytest.raises(ValueError, match="gtol must be zero"):
        LMRootSolver(gtol=-1.0)
    with pytest.raises(ValueError, match="xtol must be zero"):
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
