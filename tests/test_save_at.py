import jax.numpy as jnp
import pytest

from tinydiffeq import IController, SaveAt, Tsit5, solve_ode


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


def test_save_at_exclusivity():
    with pytest.raises(ValueError, match="exactly one"):
        SaveAt()
    with pytest.raises(ValueError, match="exactly one"):
        SaveAt(t_1=True, steps=True)
    with pytest.raises(ValueError, match="exactly one"):
        SaveAt(t_1=True, ts=jnp.linspace(0.0, 1.0, 5))
    with pytest.raises(ValueError, match="fill"):
        SaveAt(steps=True, fill="zero")


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
