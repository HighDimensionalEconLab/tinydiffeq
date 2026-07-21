import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tinydiffeq import (
    EulerMaruyama,
    LMRootSolver,
    SaveAt,
    solve_sde,
    solve_semi_explicit_sdae,
)

MU, SIGMA, Y_0, T = 0.4, 0.3, 1.0, 1.0


def drift(y, z, t, args, p, algebraic_aux):
    return p["mu"] * z, algebraic_aux


def diffusion(y, z, t, args, p, algebraic_aux):
    return p["sigma"] * z


def constraint(y, z, t, args, p):
    return z - y, {"scaled": p["scale"] * z, "square": z**2}


def sdae(key, n_steps, *, save_at=None, y_0=Y_0, p=None):
    if p is None:
        p = {
            "mu": jnp.asarray(MU),
            "sigma": jnp.asarray(SIGMA),
            "scale": jnp.asarray(2.0),
        }
    return solve_semi_explicit_sdae(
        drift,
        diffusion,
        constraint,
        EulerMaruyama(),
        0.0,
        T,
        jnp.asarray(y_0),
        jnp.asarray(0.7),
        key=key,
        n_steps=n_steps,
        p=p,
        save_at=save_at,
        has_aux=True,
        has_algebraic_aux=True,
    )


def test_sdae_matches_reduced_sde_on_identical_noise_path():
    key = jax.random.key(4)
    n = 32
    full = sdae(key, n, save_at=SaveAt(steps=True))
    reduced = solve_sde(
        lambda y, t, args, p: p["mu"] * y,
        lambda y, t, args, p: p["sigma"] * y,
        EulerMaruyama(),
        0.0,
        T,
        jnp.asarray(Y_0),
        key=key,
        n_steps=n,
        p={"mu": jnp.asarray(MU), "sigma": jnp.asarray(SIGMA)},
        save_at=SaveAt(steps=True),
    )
    assert bool(full.ok)
    assert jnp.allclose(full.ys, reduced.xs, atol=2e-6, rtol=2e-6)
    assert jnp.allclose(full.zs, full.ys, atol=2e-6)
    assert jnp.allclose(full.aux["scaled"], 2.0 * full.zs)
    assert jnp.allclose(full.aux["square"], full.zs**2)


def test_sdae_aux_and_state_jvp_vjp_under_fixed_key():
    key = jax.random.key(9)

    def output(mu):
        p = {
            "mu": mu,
            "sigma": jnp.asarray(SIGMA),
            "scale": mu,
        }
        sol = sdae(key, 64, p=p, save_at=SaveAt(steps=True))
        return jnp.sum(sol.ys + sol.zs + sol.aux["scaled"])

    mu = jnp.asarray(MU)
    tangent = jax.jvp(output, (mu,), (jnp.ones_like(mu),))[1]
    cotangent = jax.grad(output)(mu)
    eps = 1e-6
    finite_difference = (output(mu + eps) - output(mu - eps)) / (2 * eps)
    assert jnp.abs(tangent - finite_difference) < 2e-5
    assert jnp.abs(cotangent - finite_difference) < 2e-5


def test_sdae_strong_half_order_against_same_path_exact_solution():
    keys = jax.random.split(jax.random.key(0), 300)
    levels = (16, 32, 64, 128)

    def path_error(n):
        numerical = jax.vmap(lambda key: sdae(key, n).ys)(keys)

        def exact(key):
            # Reproduce the solver's documented scalar draw exactly, coupling
            # each numerical endpoint to its exact Brownian path.
            dt = T / n
            d_w = jnp.sqrt(dt) * jax.random.normal(key, (n,), dtype=jnp.float64)
            return Y_0 * jnp.exp((MU - 0.5 * SIGMA**2) * T + SIGMA * jnp.sum(d_w))

        exact_values = jax.vmap(exact)(keys)
        return float(jnp.sqrt(jnp.mean((numerical - exact_values) ** 2)))

    errors = [path_error(n) for n in levels]
    slope = np.polyfit(np.log([T / n for n in levels]), np.log(errors), 1)[0]
    assert 0.35 < slope < 0.7, (slope, errors)


def test_sdae_root_guess_has_zero_tangent_and_ts_raises():
    key = jax.random.key(1)

    def from_guess(z_0):
        return solve_semi_explicit_sdae(
            lambda y, z: z,
            lambda y, z: 0.0 * y,
            lambda y, z: z - y,
            EulerMaruyama(),
            0.0,
            0.2,
            jnp.asarray(1.0),
            z_0,
            key=key,
            n_steps=2,
        ).ys

    assert jax.grad(from_guess)(jnp.asarray(3.0)) == 0.0
    with pytest.raises(ValueError, match="not supported"):
        sdae(key, 4, save_at=SaveAt(ts=jnp.linspace(0.0, 1.0, 3)))


def test_sdae_pytree_states_and_aux():
    y_0 = {"a": jnp.asarray(1.0), "b": jnp.asarray([2.0, 3.0])}
    z_0 = (jnp.asarray(0.8), jnp.asarray([1.8, 2.8]))

    def tree_drift(y, z, t, args, p, algebraic_aux):
        value = {"a": -0.1 * z[0], "b": -0.1 * z[1]}
        return value, algebraic_aux

    def tree_diffusion(y, z, t, args, p, algebraic_aux):
        return jax.tree.map(lambda leaf: 0.05 * jnp.ones_like(leaf), y)

    def tree_constraint(y, z):
        residual = jnp.concatenate([jnp.atleast_1d(z[0] - y["a"]), z[1] - y["b"]])
        return residual, {"total": z[0] + jnp.sum(z[1])}

    sol = solve_semi_explicit_sdae(
        tree_drift,
        tree_diffusion,
        tree_constraint,
        EulerMaruyama(),
        0.0,
        0.5,
        y_0,
        z_0,
        key=jax.random.key(5),
        n_steps=8,
        save_at=SaveAt(steps=True),
        has_aux=True,
        has_algebraic_aux=True,
    )
    assert bool(sol.ok)
    assert jax.tree.structure(sol.ys) == jax.tree.structure(y_0)
    assert jnp.allclose(sol.zs[0], sol.ys["a"], atol=2e-6)
    assert jnp.allclose(sol.zs[1], sol.ys["b"], atol=2e-6)
    assert sol.aux["total"].shape == sol.ts.shape


def test_sdae_root_failure_returns_consistent_prefix_and_padding():
    def failing_constraint(y, z, t):
        return z**2 + t - 0.1, {"z": z}

    def differential(y, z, t, args, p, algebraic_aux):
        return jnp.ones_like(y), algebraic_aux

    def stochastic(y, z, t, args, p, algebraic_aux):
        return jnp.zeros_like(y)

    sol = solve_semi_explicit_sdae(
        differential,
        stochastic,
        failing_constraint,
        EulerMaruyama(),
        0.0,
        1.0,
        jnp.asarray(0.0),
        jnp.asarray(0.3),
        key=jax.random.key(0),
        n_steps=4,
        save_at=SaveAt(steps=True),
        has_aux=True,
        has_algebraic_aux=True,
        root_solver=LMRootSolver(max_steps_is_success=False),
    )
    assert not bool(sol.ok)
    assert int(sol.accepted.sum()) == int(sol.num_accepted) + 1
    assert jnp.all(sol.ys[~sol.accepted] == sol.ys[sol.num_accepted])
    assert jnp.all(sol.aux["z"][~sol.accepted] == sol.aux["z"][sol.num_accepted])


@pytest.mark.parametrize(
    ("save_at", "multiplicity"),
    [(SaveAt(t_1=True), 1.0), (SaveAt(steps=True), 2.0)],
)
def test_sdae_masked_failed_lane_has_safe_aux_jvp_and_vjp(save_at, multiplicity):
    y_0 = jnp.asarray([1.0, -1.0])
    z_0 = jnp.asarray([1.0, 0.0])

    def one_lane(y, z, p):
        def zero_drift(y, z, t, args, p, algebraic_aux):
            return jnp.zeros_like(y), algebraic_aux

        def zero_diffusion(y, z, t, args, p, algebraic_aux):
            return jnp.zeros_like(y)

        return solve_semi_explicit_sdae(
            zero_drift,
            zero_diffusion,
            lambda y, z, t, args, p: (
                z**2 - y - p,
                {"off_domain": p * jnp.sqrt(y)},
            ),
            EulerMaruyama(),
            0.0,
            0.1,
            y,
            z,
            p=p,
            key=jax.random.key(0),
            n_steps=1,
            save_at=save_at,
            has_aux=True,
            has_algebraic_aux=True,
            root_solver=LMRootSolver(max_steps_is_success=False),
            failure_ad_reference=(1.0, 1.0, 0.0, 0.0),
        )

    def batch(p):
        return jax.vmap(lambda y, z: one_lane(y, z, p))(y_0, z_0)

    def loss(p):
        sol = batch(p)
        lane_values = sol.zs.reshape(2, -1).sum(axis=1)
        lane_values += sol.aux["off_domain"].reshape(2, -1).sum(axis=1)
        return jnp.sum(jnp.where(sol.ok, lane_values, 0.0))

    p = jnp.asarray(0.0)
    sol = batch(p)
    expected = 1.5 * multiplicity
    assert jnp.array_equal(sol.ok, jnp.asarray([True, False]))
    assert jnp.all(jnp.isfinite(sol.aux["off_domain"]))
    assert jnp.allclose(jax.jvp(loss, (p,), (jnp.ones_like(p),))[1], expected)
    assert jnp.allclose(jax.grad(loss)(p), expected)


def test_sdae_nonfinite_initial_aux_fails_without_time_steps():
    def bad_drift(y, z, t, args, p, algebraic_aux):
        return jnp.full_like(y, jnp.nan), algebraic_aux

    def bad_diffusion(y, z, t, args, p, algebraic_aux):
        return jnp.full_like(y, jnp.nan)

    sol = solve_semi_explicit_sdae(
        bad_drift,
        bad_diffusion,
        lambda y, z: (z - y, {"bad": jnp.sqrt(-y)}),
        EulerMaruyama(),
        0.0,
        1.0,
        jnp.asarray(1.0),
        jnp.asarray(1.0),
        key=jax.random.key(0),
        n_steps=8,
        save_at=SaveAt(steps=True),
        has_aux=True,
        has_algebraic_aux=True,
    )
    assert not bool(sol.ok)
    assert int(sol.num_accepted) == 0
    assert jnp.all(sol.accepted == jnp.asarray([True] + [False] * 8))
    assert jnp.all(sol.ys == 1.0)
    assert jnp.all(sol.aux["bad"] == 0.0)


def test_sdae_nonfinite_endpoint_aux_terminates_at_previous_node():
    def unit_drift(y, z, t, args, p, algebraic_aux):
        return jnp.ones_like(y), algebraic_aux

    def zero_diffusion(y, z, t, args, p, algebraic_aux):
        return jnp.zeros_like(y)

    sol = solve_semi_explicit_sdae(
        unit_drift,
        zero_diffusion,
        lambda y, z, t: (z - y, {"limited": jnp.sqrt(0.05 - t)}),
        EulerMaruyama(),
        0.0,
        0.2,
        jnp.asarray(0.0),
        jnp.asarray(0.0),
        key=jax.random.key(0),
        n_steps=2,
        save_at=SaveAt(steps=True),
        has_aux=True,
        has_algebraic_aux=True,
    )
    assert not bool(sol.ok)
    assert int(sol.num_accepted) == 0
    assert jnp.all(sol.ys == 0.0)
    assert jnp.all(sol.aux["limited"] == sol.aux["limited"][0])
