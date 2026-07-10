import jax
import jax.numpy as jnp
import pytest

from tinydiffeq import IController, SaveAt, Tsit5, solve_ode


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
