import jax
import jax.numpy as jnp
import numpy as np

from tinydiffeq import solve_bvp


def scaled_fun(t, y, z, args, p):
    return jnp.array([y[1], y[0]])


def scaled_bc(ya, yb, z, args, p):
    return jnp.array([ya[0] - p, yb[0]])


T_GRID = jnp.linspace(0.0, 1.0, 5)


def scaled_solve(p):
    return solve_bvp(
        scaled_fun, scaled_bc, T_GRID, jnp.zeros((5, 2)), p=p, max_nodes=16
    )


def test_vmap_matches_sequential():
    p_values = jnp.array([0.5, 1.0, 2.0])
    batched = jax.vmap(scaled_solve)(p_values)
    for lane, p in enumerate([0.5, 1.0, 2.0]):
        single = scaled_solve(p)
        assert int(batched.status[lane]) == int(single.status)
        assert int(batched.num_nodes[lane]) == int(single.num_nodes)
        # Batched linear-algebra kernels round differently than single ones,
        # so agreement is at roundoff rather than bitwise.
        np.testing.assert_allclose(
            np.asarray(batched.y[lane]), np.asarray(single.y), rtol=1e-12, atol=1e-14
        )
        np.testing.assert_allclose(
            np.asarray(batched.t[lane]), np.asarray(single.t), rtol=1e-12
        )

    batched_gradients = jax.vmap(jax.grad(lambda p: scaled_solve(p).y[2, 0]))(p_values)
    for lane, p in enumerate([0.5, 1.0, 2.0]):
        single = jax.grad(lambda p: scaled_solve(p).y[2, 0])(p)
        np.testing.assert_allclose(
            float(batched_gradients[lane]), float(single), rtol=1e-10
        )


def test_vmap_mixed_success_and_failure():
    def stiff_solve(eps):
        def fun(t, y, z, args, p):
            return jnp.array(
                [
                    y[1],
                    -(
                        t * y[1]
                        + p * jnp.pi**2 * jnp.cos(jnp.pi * t)
                        + jnp.pi * t * jnp.sin(jnp.pi * t)
                    )
                    / p,
                ]
            )

        def bc(ya, yb, z, args, p):
            return jnp.array([ya[0] + 2.0, yb[0]])

        return solve_bvp(
            fun, bc, jnp.linspace(-1.0, 1.0, 5), jnp.zeros((5, 2)), p=eps, max_nodes=24
        )

    eps_values = jnp.array([1e-1, 1e-4])
    batched = jax.vmap(stiff_solve)(eps_values)
    assert int(batched.status[0]) == 0
    assert int(batched.status[1]) == 1
    single = stiff_solve(1e-1)
    np.testing.assert_allclose(
        np.asarray(batched.y[0]), np.asarray(single.y), rtol=1e-12, atol=1e-14
    )

    gradients = jax.vmap(jax.grad(lambda eps: stiff_solve(eps).y[3, 0]))(eps_values)
    assert bool(jnp.all(jnp.isfinite(gradients)))
    assert float(gradients[1]) == 0.0
    single_gradient = jax.grad(lambda eps: stiff_solve(eps).y[3, 0])(1e-1)
    np.testing.assert_allclose(float(gradients[0]), float(single_gradient), rtol=1e-10)
