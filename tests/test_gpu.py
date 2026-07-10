import jax
import jax.numpy as jnp
import pytest

from tinydiffeq import (
    EulerMaruyama,
    IController,
    SaveAt,
    Tsit5,
    solve_ode,
    solve_sde,
    solve_semi_explicit_dae,
)


def gpu_devices():
    try:
        devices = jax.devices()
    except RuntimeError:
        return []
    return [device for device in devices if device.platform == "gpu"]


pytestmark = pytest.mark.skipif(
    not gpu_devices(),
    reason="JAX GPU backend is not available",
)


def test_jitted_adaptive_solve_runs_on_gpu():
    gpu = gpu_devices()[0]

    def f(x, t, args, p):
        return -p * x

    @jax.jit
    def run(x_0, p):
        return solve_ode(
            f,
            Tsit5(),
            0.0,
            1.0,
            x_0,
            p=p,
            dt_0=0.1,
            controller=IController(rtol=1e-6, atol=1e-8),
            max_steps=128,
            save_at=SaveAt(steps=True),
        )

    with jax.default_device(gpu):
        x_0 = jnp.asarray([1.0, 2.0])
        p = jnp.asarray(1.3)
        sol = run(x_0, p)
        jax.block_until_ready(sol)

    assert bool(sol.ok)
    assert next(iter(sol.xs.devices())).platform == "gpu"
    assert bool(jnp.all(jnp.isfinite(sol.xs[sol.accepted])))


def test_gradient_through_solve_on_gpu():
    gpu = gpu_devices()[0]

    def endpoint(p):
        return solve_ode(
            lambda x, t, args, q: -q * x,
            Tsit5(),
            0.0,
            1.0,
            jnp.asarray(1.0),
            p=p,
            dt_0=0.1,
            controller=IController(rtol=1e-8, atol=1e-10),
            max_steps=128,
        ).xs

    with jax.default_device(gpu):
        grad = jax.jit(jax.grad(endpoint))(jnp.asarray(1.3))
        grad.block_until_ready()

    assert next(iter(grad.devices())).platform == "gpu"
    assert bool(jnp.isfinite(grad))


def test_dae_value_and_gradient_run_on_gpu():
    gpu = gpu_devices()[0]

    def endpoint(p):
        return solve_semi_explicit_dae(
            lambda y, z, t, args, q: q * z,
            lambda y, z: z - y,
            Tsit5(),
            0.0,
            1.0,
            jnp.asarray(1.0),
            jnp.asarray(0.5),
            p=p,
            dt_0=0.1,
            controller=IController(rtol=1e-5, atol=1e-7),
            max_steps=64,
        ).ys

    with jax.default_device(gpu):
        value, grad = jax.jit(lambda p: (endpoint(p), jax.grad(endpoint)(p)))(
            jnp.asarray(1.3)
        )
        jax.block_until_ready((value, grad))

    assert next(iter(value.devices())).platform == "gpu"
    assert next(iter(grad.devices())).platform == "gpu"
    assert bool(jnp.isfinite(value))
    assert bool(jnp.isfinite(grad))


def test_pytree_ode_and_sde_run_on_gpu():
    gpu = gpu_devices()[0]
    x_0 = {"a": jnp.asarray(1.0), "b": jnp.asarray([2.0, 3.0])}

    def scale(x, factor):
        return jax.tree.map(lambda leaf: factor * leaf, x)

    @jax.jit
    def run(x):
        ode = solve_ode(
            lambda state: scale(state, -0.2),
            Tsit5(),
            0.0,
            1.0,
            x,
            dt_0=0.1,
            controller=IController(),
            max_steps=32,
        )
        sde = solve_sde(
            lambda state: scale(state, -0.2),
            lambda state: jax.tree.map(lambda leaf: jnp.ones_like(leaf) * 0.1, state),
            EulerMaruyama(),
            0.0,
            1.0,
            x,
            key=jax.random.key(0),
            n_steps=16,
        )
        return ode, sde

    with jax.default_device(gpu):
        ode, sde = run(x_0)
        jax.block_until_ready((ode, sde))

    assert bool(ode.ok & sde.ok)
    assert all(
        leaf.devices().pop().platform == "gpu" for leaf in jax.tree.leaves(ode.xs)
    )
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(sde.xs))
