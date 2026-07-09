import jax.numpy as jnp
import pytest

from tinydiffeq import IController, SaveAt, Tsit5, solve_ode


def solve(saveat, *, rtol=1e-9, atol=1e-12, max_steps=256, dt0=0.2):
    return solve_ode(
        lambda x: -x,
        Tsit5(),
        0.0,
        2.0,
        jnp.asarray(1.0),
        dt0=dt0,
        controller=IController(rtol=rtol, atol=atol),
        max_steps=max_steps,
        saveat=saveat,
    )


def test_ts_grid_interpolation_error():
    # cubic Hermite dense output is 4th order between 5th-order knots; well
    # under 1e-6 for these tolerances even though looser than the knot error
    grid = jnp.linspace(0.0, 2.0, 37)
    sol = solve(SaveAt(ts=grid))
    assert bool(sol.ok)
    assert sol.xs.shape == grid.shape
    assert jnp.max(jnp.abs(sol.xs - jnp.exp(-grid))) < 1e-7


def test_ts_queries_at_t0_knots_and_t1():
    steps = solve(SaveAt(steps=True))
    endpoint = solve(SaveAt(t1=True))
    knot_ts = steps.ts[steps.accepted]
    sol = solve(SaveAt(ts=knot_ts))
    knot_xs = steps.xs[steps.accepted]
    # exact reproduction at t0 and every accepted knot (including t1)
    assert jnp.max(jnp.abs(sol.xs - knot_xs)) < 1e-14
    assert jnp.abs(sol.xs[0] - 1.0) < 1e-14
    assert jnp.abs(sol.xs[-1] - endpoint.xs) < 1e-14


def test_ts_frozen_tail_flat_extrapolation():
    # starve the solve so it stops short of t1; queries beyond the reached
    # time hit zero-width brackets and return the last state
    starved = solve(SaveAt(steps=True), rtol=1e-13, atol=1e-15, max_steps=6)
    assert not bool(starved.ok)
    reached_t = starved.ts[-1]
    reached_x = starved.xs[-1]
    grid = jnp.asarray([float(reached_t) + 0.1, 1.9, 2.0])
    sol = solve(SaveAt(ts=grid), rtol=1e-13, atol=1e-15, max_steps=6)
    assert not bool(sol.ok)
    assert bool(jnp.all(sol.xs == reached_x))


def test_fill_last_keeps_rejection_duplicates():
    # a huge dt0 forces the first attempts to reject, duplicating row 0
    sol = solve(SaveAt(steps=True), rtol=1e-10, atol=1e-12, dt0=2.0)
    assert bool(sol.ok)
    assert not bool(sol.accepted[1])
    assert sol.ts[1] == sol.ts[0]
    assert sol.xs[1] == sol.xs[0]


def test_fill_inf_masks_non_accepted_rows():
    sol = solve(SaveAt(steps=True, fill="inf"), rtol=1e-10, atol=1e-12, dt0=2.0)
    assert bool(jnp.all(jnp.isfinite(sol.xs[sol.accepted])))
    assert bool(jnp.all(jnp.isinf(sol.xs[~sol.accepted])))
    assert bool(jnp.all(jnp.isinf(sol.ts[~sol.accepted])))
    assert bool(sol.accepted[0])


def test_accepted_mask_counts_num_accepted():
    sol = solve(SaveAt(steps=True))
    assert int(sol.accepted.sum()) == int(sol.num_accepted) + 1


def test_saveat_exclusivity():
    with pytest.raises(ValueError, match="exactly one"):
        SaveAt()
    with pytest.raises(ValueError, match="exactly one"):
        SaveAt(t1=True, steps=True)
    with pytest.raises(ValueError, match="exactly one"):
        SaveAt(t1=True, ts=jnp.linspace(0.0, 1.0, 5))
    with pytest.raises(ValueError, match="fill"):
        SaveAt(steps=True, fill="zero")
