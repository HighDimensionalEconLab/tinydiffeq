import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tinydiffeq import Euler, EulerMaruyama, SaveAt, solve_ode, solve_sde

# Geometric Brownian motion dX = mu X dt + sigma X d_w has the exact solution
# X_T = X_0 exp((mu - sigma^2/2) T + sigma W_T). solve_sde presamples its
# increments as sqrt(dt) * normal(key, (n_steps,) + shape), so the test can
# regenerate the SAME path and evaluate the exact endpoint on it -- the
# strong error at each dt level compares EM and the exact solution driven by
# identical noise.

MU, SIGMA, X_0, T = 0.7, 0.5, 1.0, 1.0


def drift(x, t, args, p):
    mu, _ = p
    return mu * x


def diffusion(x, t, args, p):
    _, sigma = p
    return sigma * x


def em_endpoint(key, n_steps, p=(MU, SIGMA)):
    return solve_sde(
        drift,
        diffusion,
        EulerMaruyama(),
        0.0,
        T,
        jnp.asarray(X_0),
        key=key,
        n_steps=n_steps,
        p=p,
    ).xs


def exact_endpoint(key, n_steps):
    dt = T / n_steps
    d_w = jnp.sqrt(dt) * jax.random.normal(key, (n_steps,), dtype=jnp.float64)
    w_T = jnp.sum(d_w)
    return X_0 * jnp.exp((MU - 0.5 * SIGMA**2) * T + SIGMA * w_T)


def test_gbm_strong_convergence_rate():
    keys = jax.random.split(jax.random.PRNGKey(0), 400)
    levels = (16, 32, 64, 128, 256)
    errors = []
    for n in levels:
        em = jax.vmap(lambda k, n=n: em_endpoint(k, n))(keys)
        exact = jax.vmap(lambda k, n=n: exact_endpoint(k, n))(keys)
        errors.append(float(jnp.mean(jnp.abs(em - exact))))
    slope = np.polyfit(np.log([T / n for n in levels]), np.log(errors), 1)[0]
    assert 0.35 < slope < 0.65, (slope, errors)


def test_same_key_reproducible():
    key = jax.random.PRNGKey(3)
    a = solve_sde(
        drift,
        diffusion,
        EulerMaruyama(),
        0.0,
        T,
        jnp.asarray(X_0),
        key=key,
        n_steps=64,
        p=(MU, SIGMA),
        save_at=SaveAt(steps=True),
    )
    b = solve_sde(
        drift,
        diffusion,
        EulerMaruyama(),
        0.0,
        T,
        jnp.asarray(X_0),
        key=key,
        n_steps=64,
        p=(MU, SIGMA),
        save_at=SaveAt(steps=True),
    )
    assert jnp.array_equal(a.xs, b.xs)
    assert jnp.array_equal(a.ts, b.ts)


def test_jvp_vjp_wrt_x_0_mu_sigma_vs_finite_differences():
    key = jax.random.PRNGKey(7)

    def endpoint(theta):
        x_0, mu, sigma = theta
        return solve_sde(
            drift,
            diffusion,
            EulerMaruyama(),
            0.0,
            T,
            x_0,
            key=key,
            n_steps=128,
            p=(mu, sigma),
        ).xs

    theta = jnp.asarray([X_0, MU, SIGMA])
    grad = jax.grad(endpoint)(theta)
    jvps = jnp.stack(
        [jax.jvp(endpoint, (theta,), (jnp.eye(3)[i],))[1] for i in range(3)]
    )
    eps = 1e-6
    for i in range(3):
        fd = (
            endpoint(theta + eps * jnp.eye(3)[i])
            - endpoint(theta - eps * jnp.eye(3)[i])
        ) / (2 * eps)
        assert jnp.abs(grad[i] - fd) < 1e-6, i
        assert jnp.abs(jvps[i] - fd) < 1e-6, i


def test_zero_diffusion_matches_euler_ode():
    n = 16  # T/n exactly representable so both paths take identical steps

    def f(x):
        return MU * x

    sde = solve_sde(
        lambda x: MU * x,
        lambda x: 0.0 * x,
        EulerMaruyama(),
        0.0,
        T,
        jnp.asarray(X_0),
        key=jax.random.PRNGKey(0),
        n_steps=n,
        save_at=SaveAt(steps=True),
    )
    ode = solve_ode(
        f,
        Euler(),
        0.0,
        T,
        jnp.asarray(X_0),
        dt_0=T / n,
        max_steps=n,
        save_at=SaveAt(steps=True),
    )
    assert jnp.max(jnp.abs(sde.xs - ode.xs)) < 1e-14


def test_saveat_ts_raises():
    with pytest.raises(ValueError, match="rough paths"):
        solve_sde(
            drift,
            diffusion,
            EulerMaruyama(),
            0.0,
            T,
            jnp.asarray(X_0),
            key=jax.random.PRNGKey(0),
            n_steps=8,
            p=(MU, SIGMA),
            save_at=SaveAt(ts=jnp.linspace(0.0, T, 5)),
        )


def test_traced_n_steps_raises():
    with pytest.raises(TypeError, match="static"):
        solve_sde(
            drift,
            diffusion,
            EulerMaruyama(),
            0.0,
            T,
            jnp.asarray(X_0),
            key=jax.random.PRNGKey(0),
            n_steps=jnp.asarray(8),
            p=(MU, SIGMA),
        )


def test_steps_mode_shapes_and_flags():
    n = 32
    sol = solve_sde(
        drift,
        diffusion,
        EulerMaruyama(),
        0.0,
        T,
        jnp.asarray([X_0, 2.0]),
        key=jax.random.PRNGKey(1),
        n_steps=n,
        p=(MU, SIGMA),
        save_at=SaveAt(steps=True),
    )
    assert sol.ts.shape == (n + 1,)
    assert sol.xs.shape == (n + 1, 2)
    assert bool(sol.ok)
    assert int(sol.num_accepted) == n
    assert bool(jnp.all(sol.accepted))
    assert sol.ts[0] == 0.0 and sol.ts[-1] == T
