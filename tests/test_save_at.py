import jax
import jax.numpy as jnp
import pytest

from tinydiffeq import (
    RK4,
    ConstantStepSize,
    Euler,
    IController,
    SaveAt,
    Tsit5,
    solve_ode,
)


def solve(save_at, *, rtol=1e-9, atol=1e-12, max_steps=256, dt_0=0.2):
    return solve_ode(
        lambda x: -x,
        Tsit5(),
        0.0,
        2.0,
        jnp.asarray(1.0),
        dt_0=dt_0,
        controller=IController(rtol=rtol, atol=atol),
        max_steps=max_steps,
        save_at=save_at,
    )


def test_ts_grid_interpolation_error():
    # cubic Hermite dense output is 4th order between 5th-order knots; well
    # under 1e-6 for these tolerances even though looser than the knot error
    grid = jnp.linspace(0.0, 2.0, 37)
    sol = solve(SaveAt(ts=grid))
    assert bool(sol.ok)
    assert sol.xs.shape == grid.shape
    assert jnp.max(jnp.abs(sol.xs - jnp.exp(-grid))) < 1e-7


def test_ts_queries_at_t_0_knots_and_t_1():
    steps = solve(SaveAt(steps=True))
    endpoint = solve(SaveAt(t_1=True))
    knot_ts = steps.ts[steps.accepted]
    sol = solve(SaveAt(ts=knot_ts))
    knot_xs = steps.xs[steps.accepted]
    # exact reproduction at t_0 and every accepted knot (including t_1)
    assert jnp.max(jnp.abs(sol.xs - knot_xs)) < 1e-14
    assert jnp.abs(sol.xs[0] - 1.0) < 1e-14
    assert jnp.abs(sol.xs[-1] - endpoint.xs) < 1e-14


def test_ts_frozen_tail_flat_extrapolation():
    # starve the solve so it stops short of t_1; queries beyond the reached
    # time hit zero-width brackets and return the last state
    starved = solve(SaveAt(steps=True), rtol=1e-13, atol=1e-15, max_steps=6)
    assert not bool(starved.ok)
    n_valid = int(starved.num_accepted) + 1
    assert bool(jnp.all(starved.accepted[:n_valid]))
    assert not bool(jnp.any(starved.accepted[n_valid:]))
    assert bool(jnp.all(starved.ts[n_valid:] == starved.ts[n_valid - 1]))
    assert bool(jnp.all(starved.xs[n_valid:] == starved.xs[n_valid - 1]))
    assert starved.ts[n_valid - 1] < 2.0
    reached_t = starved.ts[-1]
    reached_x = starved.xs[-1]
    grid = jnp.asarray([float(reached_t) + 0.1, 1.9, 2.0])
    sol = solve(SaveAt(ts=grid), rtol=1e-13, atol=1e-15, max_steps=6)
    assert not bool(sol.ok)
    assert bool(jnp.all(sol.xs == reached_x))


def test_steps_omit_rejections_and_pad_with_last_value():
    # A huge dt_0 forces initial rejections, but public rows contain only the
    # accepted trajectory followed by padding.
    sol = solve(SaveAt(steps=True), rtol=1e-10, atol=1e-12, dt_0=2.0)
    assert bool(sol.ok)
    n_valid = int(sol.num_accepted) + 1
    assert bool(jnp.all(sol.accepted[:n_valid]))
    assert not bool(jnp.any(sol.accepted[n_valid:]))
    assert bool(jnp.all(jnp.diff(sol.ts[:n_valid]) > 0.0))
    assert sol.ts[int(sol.num_accepted)] == 2.0
    assert bool(jnp.all(sol.ts[n_valid:] == sol.ts[n_valid - 1]))
    assert bool(jnp.all(sol.xs[n_valid:] == sol.xs[n_valid - 1]))
    assert int(sol.num_steps) > int(sol.num_accepted)


def test_fill_inf_masks_non_accepted_rows():
    sol = solve(SaveAt(steps=True, fill="inf"), rtol=1e-10, atol=1e-12, dt_0=2.0)
    assert bool(jnp.all(jnp.isfinite(sol.xs[sol.accepted])))
    assert bool(jnp.all(jnp.isinf(sol.xs[~sol.accepted])))
    assert bool(jnp.all(jnp.isinf(sol.ts[~sol.accepted])))
    assert bool(sol.accepted[0])


def test_accepted_mask_counts_num_accepted():
    sol = solve(SaveAt(steps=True))
    assert int(sol.accepted.sum()) == int(sol.num_accepted) + 1
    assert bool(jnp.all(sol.accepted[:-1] >= sol.accepted[1:]))


def test_python_sequence_and_array_grids_match():
    times = [0.0, 0.25, 0.25, 1.0, 2.0]
    from_list = solve(SaveAt(ts=times))
    from_array = solve(SaveAt(ts=jnp.asarray(times)))
    assert bool(jnp.array_equal(from_list.ts, from_array.ts))
    assert bool(jnp.array_equal(from_list.xs, from_array.xs))


def test_requested_grid_does_not_change_adaptive_step_count():
    endpoint = solve(SaveAt(t_1=True))
    sampled = solve(SaveAt(ts=[0.0, 0.1, 0.7, 2.0]))
    assert endpoint.num_accepted == sampled.num_accepted
    assert sampled.xs[-1] == endpoint.xs


def test_exact_fixed_grid_gathers_internal_rk4_states_and_supports_ad():
    grid = jnp.linspace(0.0, 2.0, 9)

    def sampled(parameter):
        return solve_ode(
            lambda x, t, args, p: -p * x,
            RK4(),
            0.0,
            2.0,
            jnp.asarray(1.0),
            p=parameter,
            dt_0=0.125,
            controller=ConstantStepSize(),
            max_steps=16,
            save_at=SaveAt(ts=grid, exact=True),
            has_aux=False,
        )

    parameter = jnp.asarray(0.4)
    solution = sampled(parameter)
    steps = solve_ode(
        lambda x, t, args, p: -p * x,
        RK4(),
        0.0,
        2.0,
        jnp.asarray(1.0),
        p=parameter,
        dt_0=0.125,
        controller=ConstantStepSize(),
        max_steps=16,
        save_at=SaveAt(steps=True),
        has_aux=False,
    )
    assert bool(solution.ok)
    assert jnp.array_equal(solution.xs, steps.xs[::2])
    tangent = jax.jvp(
        lambda value: sampled(value).xs,
        (parameter,),
        (jnp.ones_like(parameter),),
    )[1]
    gradient = jax.grad(lambda value: jnp.sum(sampled(value).xs))(parameter)
    assert bool(jnp.all(jnp.isfinite(tangent)))
    assert bool(jnp.isfinite(gradient))


def test_exact_grid_accepts_clipped_final_step_and_rejects_nonknots():
    def run(grid):
        return solve_ode(
            lambda x: -x,
            RK4(),
            0.0,
            1.0,
            jnp.asarray(1.0),
            dt_0=0.3,
            max_steps=4,
            save_at=SaveAt(ts=grid, exact=True),
            has_aux=False,
        )

    aligned = run(jnp.asarray([0.0, 0.3, 0.6, 0.9, 1.0]))
    unaligned = run(jnp.asarray([0.0, 0.3, 0.65, 1.0]))
    assert bool(aligned.ok)
    assert not bool(unaligned.ok)
    assert aligned.xs[-1] == unaligned.xs[-1]


def test_float32_long_exact_grid_selects_unique_late_knots():
    dtype = jnp.float32
    horizon = jnp.asarray(100.0, dtype)
    max_steps = 4096
    dt = horizon / max_steps
    grid = horizon - dt * jnp.asarray([2.0, 1.0, 0.0], dtype)
    solution = solve_ode(
        lambda x, t: jnp.ones_like(x),
        RK4(),
        jnp.asarray(0.0, dtype),
        horizon,
        jnp.asarray(0.0, dtype),
        dt_0=dt,
        max_steps=max_steps,
        save_at=SaveAt(ts=grid, exact=True),
        has_aux=False,
    )
    assert bool(solution.ok)
    assert jnp.all(jnp.diff(solution.xs) > 0.0)
    assert jnp.allclose(solution.xs, grid, rtol=2e-6, atol=2e-5)


def test_float32_exact_grid_alignment_accounts_for_large_time_offset():
    dtype = jnp.float32
    t_0 = jnp.asarray(-1000.0, dtype)
    t_1 = jnp.asarray(1.0, dtype)
    max_steps = 100
    dt = (t_1 - t_0) / max_steps
    # linspace and t_0 + i * dt are mathematically the same grid, but their
    # cancellation error near zero is governed by |t_0|, not the local time.
    grid = jnp.linspace(t_0, t_1, max_steps + 1)
    solution = solve_ode(
        lambda x: jnp.ones_like(x),
        RK4(),
        t_0,
        t_1,
        jnp.asarray(0.0, dtype),
        dt_0=dt,
        max_steps=max_steps,
        save_at=SaveAt(ts=grid, exact=True),
        has_aux=False,
    )
    assert bool(solution.ok)
    assert jnp.allclose(solution.xs, grid - t_0, rtol=2e-6, atol=2e-4)


def test_exact_and_dense_fixed_grid_outputs_match_at_internal_knots():
    def run(save_at):
        return solve_ode(
            lambda x, t, args, p: jnp.sin(t) - p * x,
            RK4(),
            -0.5,
            1.0,
            jnp.asarray(0.7),
            p=jnp.asarray(0.2),
            dt_0=0.125,
            max_steps=12,
            save_at=save_at,
            has_aux=False,
        )

    steps = run(SaveAt(steps=True))
    knots = steps.ts[steps.accepted]
    exact = run(SaveAt(ts=knots, exact=True))
    dense = run(SaveAt(ts=knots))
    assert bool(exact.ok & dense.ok)
    assert jnp.array_equal(exact.xs, steps.xs[steps.accepted])
    assert jnp.allclose(dense.xs, exact.xs, rtol=1e-14, atol=1e-14)


def test_float32_fixed_solution_is_invariant_to_nonbinding_budget():
    dtype = jnp.float32
    horizon = jnp.asarray(100.0, dtype)
    dt = jnp.asarray(0.025, dtype)

    def run(max_steps):
        return solve_ode(
            lambda x, t: jnp.sin(jnp.asarray(20.0, dtype) * t),
            RK4(),
            jnp.asarray(0.0, dtype),
            horizon,
            jnp.asarray(0.0, dtype),
            dt_0=dt,
            max_steps=max_steps,
            has_aux=False,
        )

    exact_budget = run(4000)
    extra_budget = run(8192)
    assert bool(exact_budget.ok & extra_budget.ok)
    assert int(exact_budget.num_accepted) == 4000
    assert int(extra_budget.num_accepted) == 4000
    assert exact_budget.ts == horizon
    assert extra_budget.ts == horizon
    assert jnp.allclose(exact_budget.xs, extra_budget.xs, rtol=1e-6, atol=1e-6)


def test_float32_uniform_gate_caps_tolerance_at_large_time_offsets():
    dtype = jnp.float32

    def run(max_steps):
        return solve_ode(
            lambda x, t: jnp.where(
                t >= jnp.asarray(1_000_000.25, dtype),
                jnp.asarray(1e38, dtype),
                jnp.zeros_like(x),
            ),
            Euler(),
            1_000_000.0,
            1_000_000.25,
            jnp.asarray(0.0, dtype),
            dt_0=0.125,
            max_steps=max_steps,
            has_aux=False,
        )

    exact_budget = run(2)
    extra_budget = run(3)
    assert bool(exact_budget.ok & extra_budget.ok)
    assert int(exact_budget.num_steps) == int(extra_budget.num_steps) == 2
    assert exact_budget.ts == extra_budget.ts == jnp.asarray(1_000_000.25, dtype)
    assert exact_budget.xs == extra_budget.xs == jnp.asarray(0.0, dtype)


def test_float32_fixed_horizon_tolerance_accounts_for_large_start_time():
    dtype = jnp.float32
    t_0 = -9_834_724.0
    t_1 = 6.2654047
    max_steps = 410
    dt_0 = (t_1 - t_0) / max_steps

    sol = solve_ode(
        lambda x: jnp.zeros_like(x),
        Euler(),
        t_0,
        t_1,
        jnp.asarray(0.0, dtype),
        dt_0=dt_0,
        max_steps=max_steps,
        has_aux=False,
    )

    assert bool(sol.ok)
    assert int(sol.num_steps) == max_steps
    assert sol.ts == jnp.asarray(t_1, dtype)


def test_float32_uniform_gate_requires_first_horizon_reach_at_final_slot():
    dtype = jnp.float32
    t_0 = 1_000_000.0
    t_1 = 1_000_000.0625
    dt_0 = (t_1 - t_0) / 10

    def run(max_steps):
        return solve_ode(
            lambda x, t: jnp.where(
                t >= jnp.asarray(t_1, dtype),
                jnp.asarray(1e38, dtype),
                jnp.zeros_like(x),
            ),
            Euler(),
            t_0,
            t_1,
            jnp.asarray(0.0, dtype),
            dt_0=dt_0,
            max_steps=max_steps,
            has_aux=False,
        )

    exact_budget = run(10)
    extra_budget = run(11)
    loose_budget = run(20)
    assert bool(exact_budget.ok & extra_budget.ok & loose_budget.ok)
    assert 0 < int(exact_budget.num_steps) < 10
    assert exact_budget.num_steps == extra_budget.num_steps == loose_budget.num_steps
    assert (
        exact_budget.ts == extra_budget.ts == loose_budget.ts == jnp.asarray(t_1, dtype)
    )
    assert (
        exact_budget.xs == extra_budget.xs == loose_budget.xs == jnp.asarray(0.0, dtype)
    )


def test_float32_uniform_gate_does_not_accept_a_different_horizon():
    dtype = jnp.float32
    dt = jnp.asarray(100.0 / 4096, dtype)

    def run(horizon):
        return solve_ode(
            lambda x: jnp.ones_like(x),
            RK4(),
            jnp.asarray(0.0, dtype),
            jnp.asarray(horizon, dtype),
            jnp.asarray(0.0, dtype),
            dt_0=dt,
            max_steps=4096,
            has_aux=False,
        )

    shorter = run(99.9)
    longer = run(100.1)
    assert bool(shorter.ok)
    assert shorter.ts == jnp.asarray(99.9, dtype)
    assert not bool(longer.ok)
    assert longer.ts < jnp.asarray(100.1, dtype)


def test_exact_grid_aux_and_derivatives_match_selected_step_rows():
    grid = jnp.linspace(0.0, 2.0, 9)

    def run(parameter, save_at):
        return solve_ode(
            lambda x, t, args, p: (-p * x, {"level": p * x + t}),
            RK4(),
            0.0,
            2.0,
            jnp.asarray(1.0),
            p=parameter,
            dt_0=0.125,
            max_steps=16,
            save_at=save_at,
            has_aux=True,
        )

    parameter = jnp.asarray(0.4)
    exact = run(parameter, SaveAt(ts=grid, exact=True))
    steps = run(parameter, SaveAt(steps=True))
    assert bool(exact.ok)
    assert jnp.array_equal(exact.xs, steps.xs[::2])
    assert jnp.array_equal(exact.aux["level"], steps.aux["level"][::2])

    def selected(value, exact_mode):
        save_at = SaveAt(ts=grid, exact=True) if exact_mode else SaveAt(steps=True)
        values = run(value, save_at).aux["level"]
        return values if exact_mode else values[::2]

    exact_jvp = jax.jvp(
        lambda value: selected(value, True),
        (parameter,),
        (jnp.ones_like(parameter),),
    )[1]
    step_jvp = jax.jvp(
        lambda value: selected(value, False),
        (parameter,),
        (jnp.ones_like(parameter),),
    )[1]
    exact_vjp = jax.grad(lambda value: jnp.sum(selected(value, True)))(parameter)
    step_vjp = jax.grad(lambda value: jnp.sum(selected(value, False)))(parameter)
    assert jnp.allclose(exact_jvp, step_jvp, rtol=1e-12, atol=1e-12)
    assert jnp.allclose(exact_vjp, step_vjp, rtol=1e-12, atol=1e-12)


def test_exact_grid_requires_fixed_explicit_ode():
    with pytest.raises(ValueError, match="ConstantStepSize"):
        solve(SaveAt(ts=[0.0, 1.0, 2.0], exact=True))


def test_save_at_exclusivity():
    with pytest.raises(ValueError, match="exactly one"):
        SaveAt()
    with pytest.raises(ValueError, match="exactly one"):
        SaveAt(t_1=True, steps=True)
    with pytest.raises(ValueError, match="exactly one"):
        SaveAt(t_1=True, ts=jnp.linspace(0.0, 1.0, 5))
    with pytest.raises(ValueError, match="fill"):
        SaveAt(steps=True, fill="zero")
    with pytest.raises(ValueError, match="requires ts"):
        SaveAt(steps=True, exact=True)


def test_noncanonical_public_spellings_are_rejected():
    with pytest.raises(TypeError):
        SaveAt(t1=True)
    with pytest.raises(TypeError):
        IController(dtmin=1e-10)
    with pytest.raises(TypeError):
        solve_ode(
            lambda x: -x,
            Tsit5(),
            0.0,
            1.0,
            jnp.asarray(1.0),
            dt0=0.1,
        )
    with pytest.raises(TypeError):
        solve_ode(
            lambda x: -x,
            Tsit5(),
            0.0,
            1.0,
            jnp.asarray(1.0),
            dt_0=0.1,
            saveat=SaveAt(t_1=True),
        )
