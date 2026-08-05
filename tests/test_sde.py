import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tinydiffeq import (
    SRA1,
    Euler,
    EulerMaruyama,
    Milstein,
    SaveAt,
    solve_ode,
    solve_sde,
)

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
    assert int(sol.num_steps) == n
    assert bool(jnp.all(sol.accepted))
    assert sol.ts[0] == 0.0 and sol.ts[-1] == T


# Additive-noise Ornstein-Uhlenbeck dX = -theta X dt + sigma d_w for the SRA1
# tests: SRA1's strong order 1.5 holds only for state-independent diffusion.

THETA, OU_SIGMA = 1.0, 0.5


def ou_drift(x, t, args, p):
    return -THETA * x


def ou_diffusion(x, t, args, p):
    return OU_SIGMA * jnp.ones_like(x)


@pytest.mark.parametrize("solver", [EulerMaruyama(), Milstein(), SRA1()])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_explicit_noise_matches_key(solver, dtype):
    key = jax.random.PRNGKey(11)
    n = 32
    x_0 = jnp.asarray(X_0, dtype)
    noise = solver.sample_noise(x_0, key, n, jnp.asarray(T / n, dtype), dtype)
    a = solve_sde(
        ou_drift,
        ou_diffusion,
        solver,
        0.0,
        T,
        x_0,
        key=key,
        n_steps=n,
        save_at=SaveAt(steps=True),
    )
    b = solve_sde(
        ou_drift,
        ou_diffusion,
        solver,
        0.0,
        T,
        x_0,
        noise=noise,
        n_steps=n,
        save_at=SaveAt(steps=True),
    )
    assert a.xs.dtype == dtype
    assert a.ts.dtype == dtype
    assert jnp.array_equal(a.xs, b.xs)
    assert jnp.array_equal(a.ts, b.ts)


def test_key_noise_exclusivity_raises():
    x_0 = jnp.asarray(X_0)
    noise = EulerMaruyama().sample_noise(
        x_0, jax.random.PRNGKey(0), 8, jnp.asarray(T / 8), x_0.dtype
    )
    with pytest.raises(ValueError, match="exactly one"):
        solve_sde(
            drift, diffusion, EulerMaruyama(), 0.0, T, x_0, n_steps=8, p=(MU, SIGMA)
        )
    with pytest.raises(ValueError, match="exactly one"):
        solve_sde(
            drift,
            diffusion,
            EulerMaruyama(),
            0.0,
            T,
            x_0,
            key=jax.random.PRNGKey(0),
            noise=noise,
            n_steps=8,
            p=(MU, SIGMA),
        )


def test_explicit_noise_wrong_shape_raises():
    with pytest.raises(ValueError, match="shape"):
        solve_sde(
            drift,
            diffusion,
            EulerMaruyama(),
            0.0,
            T,
            jnp.asarray(X_0),
            noise=jnp.zeros((7,)),
            n_steps=8,
            p=(MU, SIGMA),
        )


def test_explicit_noise_wrong_structure_raises():
    # SRA1 expects the (d_w, d_z) pair, not a bare increment array.
    with pytest.raises(ValueError, match="structure"):
        solve_sde(
            ou_drift,
            ou_diffusion,
            SRA1(),
            0.0,
            T,
            jnp.asarray(X_0),
            noise=jnp.zeros((8,)),
            n_steps=8,
        )


def test_explicit_noise_wrong_dtype_raises():
    with pytest.raises(TypeError, match="dtype"):
        solve_sde(
            drift,
            diffusion,
            EulerMaruyama(),
            0.0,
            T,
            jnp.asarray(X_0),
            noise=jnp.zeros((8,), dtype=jnp.float32),
            n_steps=8,
            p=(MU, SIGMA),
        )


def test_grad_wrt_explicit_noise_vs_finite_differences():
    n = 16
    x_0 = jnp.asarray(X_0)
    noise = EulerMaruyama().sample_noise(
        x_0, jax.random.PRNGKey(5), n, jnp.asarray(T / n), x_0.dtype
    )

    def endpoint(noise):
        return solve_sde(
            drift,
            diffusion,
            EulerMaruyama(),
            0.0,
            T,
            x_0,
            noise=noise,
            n_steps=n,
            p=(MU, SIGMA),
        ).xs

    grad = jax.grad(endpoint)(noise)
    eps = 1e-6
    for i in (0, n // 2, n - 1):
        bump = jnp.zeros_like(noise).at[i].set(eps)
        fd = (endpoint(noise + bump) - endpoint(noise - bump)) / (2 * eps)
        assert jnp.abs(grad[i] - fd) < 1e-6, i


def test_grad_wrt_sra1_noise_pair_vs_finite_differences():
    n = 16
    x_0 = jnp.asarray(X_0)
    noise = SRA1().sample_noise(
        x_0, jax.random.PRNGKey(6), n, jnp.asarray(T / n), x_0.dtype
    )

    def endpoint(noise):
        return solve_sde(
            ou_drift, ou_diffusion, SRA1(), 0.0, T, x_0, noise=noise, n_steps=n
        ).xs

    grad_w, grad_z = jax.grad(endpoint)(noise)
    eps = 1e-6
    for leaf, grad_leaf in ((0, grad_w), (1, grad_z)):
        for i in (0, n - 1):
            bump = jnp.zeros((n,)).at[i].set(eps)
            if leaf == 0:
                plus = (noise[0] + bump, noise[1])
                minus = (noise[0] - bump, noise[1])
            else:
                plus = (noise[0], noise[1] + bump)
                minus = (noise[0], noise[1] - bump)
            fd = (endpoint(plus) - endpoint(minus)) / (2 * eps)
            assert jnp.abs(grad_leaf[i] - fd) < 1e-6, (leaf, i)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_vmap_explicit_noise_matches_loop(dtype):
    n = 16
    x_0s = jnp.asarray([0.5, 1.0, 2.0], dtype)
    keys = jax.random.split(jax.random.PRNGKey(9), 3)
    d_w, d_z = jax.vmap(
        lambda k: SRA1().sample_noise(
            jnp.asarray(X_0, dtype), k, n, jnp.asarray(T / n, dtype), dtype
        )
    )(keys)

    def one(x_0, w, z):
        return solve_sde(
            ou_drift, ou_diffusion, SRA1(), 0.0, T, x_0, noise=(w, z), n_steps=n
        ).xs

    batched = jax.vmap(one)(x_0s, d_w, d_z)
    looped = jnp.stack([one(x_0s[i], d_w[i], d_z[i]) for i in range(3)])
    assert batched.dtype == dtype
    assert jnp.max(jnp.abs(batched - looped)) < 200 * jnp.finfo(dtype).eps


def test_unroll_matches_rolled_values_and_gradients():
    key = jax.random.PRNGKey(13)
    n = 32
    x_0 = jnp.asarray(X_0)

    def endpoint(x_0, unroll):
        return solve_sde(
            ou_drift,
            ou_diffusion,
            SRA1(),
            0.0,
            T,
            x_0,
            key=key,
            n_steps=n,
            unroll=unroll,
        ).xs

    rolled = solve_sde(
        ou_drift,
        ou_diffusion,
        SRA1(),
        0.0,
        T,
        x_0,
        key=key,
        n_steps=n,
        save_at=SaveAt(steps=True),
        unroll=1,
    )
    unrolled = solve_sde(
        ou_drift,
        ou_diffusion,
        SRA1(),
        0.0,
        T,
        x_0,
        key=key,
        n_steps=n,
        save_at=SaveAt(steps=True),
        unroll=4,
    )
    assert jnp.array_equal(rolled.xs, unrolled.xs)
    grad_rolled = jax.grad(lambda x: endpoint(x, 1))(x_0)
    grad_unrolled = jax.grad(lambda x: endpoint(x, 4))(x_0)
    assert jnp.allclose(grad_rolled, grad_unrolled, rtol=1e-12, atol=1e-12)
    with pytest.raises(ValueError, match="unroll"):
        endpoint(x_0, 0)


def test_milstein_strong_order_gbm():
    keys = jax.random.split(jax.random.PRNGKey(2), 400)
    levels = (16, 32, 64, 128)
    errors = []
    for n in levels:
        endpoints = jax.vmap(
            lambda k, n=n: (
                solve_sde(
                    drift,
                    diffusion,
                    Milstein(),
                    0.0,
                    T,
                    jnp.asarray(X_0),
                    key=k,
                    n_steps=n,
                    p=(MU, SIGMA),
                ).xs
            )
        )(keys)
        exact = jax.vmap(lambda k, n=n: exact_endpoint(k, n))(keys)
        errors.append(float(jnp.mean(jnp.abs(endpoints - exact))))
    slope = np.polyfit(np.log([T / n for n in levels]), np.log(errors), 1)[0]
    assert 0.75 < slope < 1.25, (slope, errors)


def coarsen_sra1_noise(d_w, d_z, dt):
    # Pasting adjacent steps: dW_c = dW_1 + dW_2 and the time-Wiener integral
    # I_c = I_1 + I_2 + dt dW_1, then dZ back from I = (dt/2)(dW + dZ/sqrt(3)).
    i_10 = 0.5 * dt * (d_w + d_z / np.sqrt(3.0))
    d_w_c = d_w[..., 0::2] + d_w[..., 1::2]
    i_c = i_10[..., 0::2] + i_10[..., 1::2] + dt * d_w[..., 0::2]
    d_z_c = np.sqrt(3.0) * (i_c / dt - d_w_c)
    return d_w_c, d_z_c


def test_sra1_strong_order_additive_nonlinear():
    # No closed form conditions on (d_w, d_z) alone, so measure
    # self-convergence against the finest grid on consistently coarsened
    # noise; additive-noise EM on the same paths is strong order 1.0, so a
    # slope well above 1 and the error ratio both separate the schemes. The
    # observed slope may exceed the guaranteed 1.5 when the h^2 error terms
    # dominate on the tested grids (exactly 2 for linear drift).
    def cubic_drift(x):
        return -x - x**3

    n_fine = 256
    keys = jax.random.split(jax.random.PRNGKey(4), 200)
    x_0 = jnp.asarray(X_0)
    d_w, d_z = jax.vmap(
        lambda k: SRA1().sample_noise(
            x_0, k, n_fine, jnp.asarray(T / n_fine), x_0.dtype
        )
    )(keys)

    def sra1_endpoints(d_w, d_z, n):
        return jax.vmap(
            lambda w, z: (
                solve_sde(
                    cubic_drift,
                    ou_diffusion,
                    SRA1(),
                    0.0,
                    T,
                    x_0,
                    noise=(w, z),
                    n_steps=n,
                ).xs
            )
        )(d_w, d_z)

    reference = sra1_endpoints(d_w, d_z, n_fine)
    level_w, level_z, n, dt = d_w, d_z, n_fine, T / n_fine
    errors = {}
    em_error = None
    while n > 16:
        level_w, level_z = coarsen_sra1_noise(level_w, level_z, dt)
        n, dt = n // 2, 2.0 * dt
        errors[n] = float(
            jnp.mean(jnp.abs(sra1_endpoints(level_w, level_z, n) - reference))
        )
        if n == 16:
            em = jax.vmap(
                lambda w: (
                    solve_sde(
                        cubic_drift,
                        ou_diffusion,
                        EulerMaruyama(),
                        0.0,
                        T,
                        x_0,
                        noise=w,
                        n_steps=16,
                    ).xs
                )
            )(level_w)
            em_error = float(jnp.mean(jnp.abs(em - reference)))
    ns = sorted(errors)
    slope = np.polyfit(np.log([T / n for n in ns]), np.log([errors[n] for n in ns]), 1)[
        0
    ]
    assert 1.3 < slope < 2.5, (slope, errors)
    assert errors[16] < 0.2 * em_error, (errors[16], em_error)


def test_zero_diffusion_sra1_is_ralston_rk2():
    # With diffusion = 0 the tableau collapses to the deterministic two-stage
    # Ralston scheme, exact per step on dx = mu x: growth 1 + mu h + (mu h)^2/2.
    n = 16
    h = T / n
    sol = solve_sde(
        lambda x: MU * x,
        lambda x: 0.0 * x,
        SRA1(),
        0.0,
        T,
        jnp.asarray(X_0),
        key=jax.random.PRNGKey(0),
        n_steps=n,
    )
    growth = 1.0 + MU * h + 0.5 * (MU * h) ** 2
    assert jnp.abs(sol.xs - X_0 * growth**n) < 1e-12


def test_zero_diffusion_milstein_matches_euler_ode():
    n = 16
    sde = solve_sde(
        lambda x: MU * x,
        lambda x: 0.0 * x,
        Milstein(),
        0.0,
        T,
        jnp.asarray(X_0),
        key=jax.random.PRNGKey(0),
        n_steps=n,
        save_at=SaveAt(steps=True),
    )
    ode = solve_ode(
        lambda x: MU * x,
        Euler(),
        0.0,
        T,
        jnp.asarray(X_0),
        dt_0=T / n,
        max_steps=n,
        save_at=SaveAt(steps=True),
    )
    assert jnp.max(jnp.abs(sde.xs - ode.xs)) < 1e-14
