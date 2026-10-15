import os

import jax
import jax.numpy as jnp
import pytest

from tinydiffeq import (
    RK4,
    SRA1,
    AdaptiveKrylovExponential,
    AssociativeMarkov,
    ContinuousTimeMarkovChain,
    DenseExponential,
    DiscreteMarkovChain,
    Euler,
    EulerMaruyama,
    IController,
    KrylovExponential,
    MatrixFreeContinuousTimeMarkovChain,
    Milstein,
    Rodas5P,
    SaveAt,
    Tsit5,
    forecast_continuous_time_markov_chain,
    forecast_markov_chain,
    jvp_linear_ode,
    simulate_continuous_time_markov_chain,
    simulate_markov_chain,
    solve_bvp,
    solve_linear_ode,
    solve_ode,
    solve_sde,
    solve_semi_explicit_dae,
    solve_semi_explicit_sdae,
    vjp_linear_ode,
)

GPU_PLATFORMS = ("gpu", "metal")


def gpu_devices():
    try:
        devices = jax.devices()
    except RuntimeError:
        return []
    return [device for device in devices if device.platform in GPU_PLATFORMS]


def gpu_supports_float64():
    devices = gpu_devices()
    return bool(devices) and devices[0].platform == "gpu"


DTYPES = pytest.mark.parametrize(
    "dtype",
    [
        jnp.float32,
        pytest.param(
            jnp.float64,
            marks=pytest.mark.skipif(
                not gpu_supports_float64(),
                reason="Active GPU backend does not support float64",
            ),
        ),
    ],
    ids=["float32", "float64"],
)


@DTYPES
def test_markov_methods_jit_and_vmap_on_gpu(dtype):
    gpu = gpu_devices()[0]
    discrete = DiscreteMarkovChain(
        jnp.asarray([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.3, 0.5]], dtype)
    )
    continuous = ContinuousTimeMarkovChain(
        jnp.asarray([[-1.0, 0.7, 0.3], [0.2, -0.6, 0.4], [0.5, 0.5, -1.0]], dtype)
    )
    keys = jax.random.split(jax.random.key(90), 64)

    with jax.default_device(gpu):
        discrete_paths = jax.jit(
            jax.vmap(
                lambda key: (
                    simulate_markov_chain(
                        discrete,
                        jnp.int32(0),
                        key=key,
                        num_steps=128,
                        method=AssociativeMarkov(),
                        save_at=SaveAt(steps=True),
                    ).xs
                )
            )
        )(keys)
        continuous_endpoints = jax.jit(
            jax.vmap(
                lambda key: simulate_continuous_time_markov_chain(
                    continuous,
                    0.0,
                    10.0,
                    jnp.int32(0),
                    key=key,
                    max_jumps=128,
                    method=AssociativeMarkov(),
                )
            )
        )(keys)
        jax.block_until_ready((discrete_paths, continuous_endpoints))

    assert next(iter(discrete_paths.devices())).platform in GPU_PLATFORMS
    assert discrete_paths.shape == (64, 129)
    assert continuous_endpoints.ts.dtype == dtype
    assert bool(jnp.all(continuous_endpoints.ok))


@DTYPES
def test_markov_distribution_forecasts_and_ad_on_gpu(dtype):
    gpu = gpu_devices()[0]
    discrete = DiscreteMarkovChain(jnp.asarray([[0.8, 0.2], [0.3, 0.7]], dtype))
    continuous = ContinuousTimeMarkovChain(
        jnp.asarray([[-2.0, 2.0], [1.0, -1.0]], dtype)
    )
    initial = jnp.asarray([0.4, 0.6], dtype)

    def objective(distribution):
        forecast = forecast_continuous_time_markov_chain(
            continuous,
            0.0,
            2.0,
            distribution,
            method=DenseExponential(),
        ).probabilities
        return jnp.asarray([1.0, -0.5], dtype) @ forecast

    with jax.default_device(gpu):
        discrete_path = jax.jit(
            lambda distribution: (
                forecast_markov_chain(
                    discrete,
                    distribution,
                    num_steps=128,
                    method=AssociativeMarkov(),
                    save_at=SaveAt(steps=True),
                ).probabilities
            )
        )(initial)
        continuous_value, continuous_gradient = jax.jit(jax.value_and_grad(objective))(
            initial
        )
        jax.block_until_ready((discrete_path, continuous_value, continuous_gradient))

    assert next(iter(discrete_path.devices())).platform in GPU_PLATFORMS
    assert discrete_path.dtype == dtype
    assert continuous_gradient.dtype == dtype
    assert jnp.allclose(discrete_path.sum(axis=1), 1.0)
    assert jnp.isfinite(continuous_value)
    assert bool(jnp.all(jnp.isfinite(continuous_gradient)))


@DTYPES
def test_matrix_free_pytree_markov_forecast_and_ad_on_gpu(dtype):
    gpu = gpu_devices()[0]
    generator = jnp.asarray(
        [[-1.0, 0.8, 0.2], [0.3, -0.7, 0.4], [0.1, 0.2, -0.3]],
        dtype,
    )

    def action(probabilities):
        flat = jnp.concatenate([probabilities["first"], probabilities["second"]])
        result = flat @ generator
        return {"first": result[:2], "second": result[2:]}

    chain = MatrixFreeContinuousTimeMarkovChain(action)
    initial = {
        "first": jnp.asarray([0.2, 0.3], dtype),
        "second": jnp.asarray([0.5], dtype),
    }

    def objective(distribution):
        probabilities = forecast_continuous_time_markov_chain(
            chain,
            0.0,
            1.0,
            distribution,
            method=KrylovExponential(krylov_dim=3),
        ).probabilities
        return jnp.sum(probabilities["first"] ** 2) + jnp.sum(
            probabilities["second"] ** 2
        )

    with jax.default_device(gpu):
        value, gradient = jax.jit(jax.value_and_grad(objective))(initial)
        jax.block_until_ready((value, gradient))

    assert next(iter(value.devices())).platform in GPU_PLATFORMS
    assert value.dtype == dtype
    assert all(
        leaf.dtype == dtype and leaf.devices().pop().platform in GPU_PLATFORMS
        for leaf in jax.tree.leaves(gradient)
    )
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(gradient))


@DTYPES
def test_dense_and_matrix_free_linear_exponential_ad_on_gpu(dtype):
    gpu = gpu_devices()[0]
    base = jnp.asarray(
        [[-0.8, 0.2, 0.0], [0.3, -0.5, 0.1], [0.0, 0.4, -0.6]],
        dtype,
    )
    direction = jnp.asarray(
        [[0.1, 0.0, -0.1], [0.0, -0.2, 0.2], [0.1, 0.0, -0.1]],
        dtype,
    )
    initial = {
        "left": jnp.asarray([0.2, -0.1], dtype),
        "right": jnp.asarray([0.5], dtype),
    }

    def flatten(state):
        return jnp.concatenate([state["left"], state["right"]])

    def endpoint(parameter, method):
        matrix = base + parameter * direction

        def operator(state):
            result = matrix @ flatten(state)
            return {"left": result[:2], "right": result[2:]}

        result = solve_linear_ode(operator, method, 0.0, 1.0, initial).xs
        return jnp.sum(flatten(result) ** 2)

    def evaluate(parameter):
        def dense(value):
            return endpoint(value, DenseExponential())

        def krylov(value):
            return endpoint(value, KrylovExponential(krylov_dim=3))

        def adaptive(value):
            return endpoint(
                value, AdaptiveKrylovExponential(krylov_dim=3, max_steps=16)
            )

        dense_value, dense_tangent = jax.jvp(
            dense, (parameter,), (jnp.ones_like(parameter),)
        )
        krylov_value, krylov_tangent = jax.jvp(
            krylov, (parameter,), (jnp.ones_like(parameter),)
        )
        adaptive_value, adaptive_tangent = jax.jvp(
            adaptive, (parameter,), (jnp.ones_like(parameter),)
        )
        return (
            dense_value,
            krylov_value,
            adaptive_value,
            dense_tangent,
            krylov_tangent,
            adaptive_tangent,
            jax.grad(dense)(parameter),
            jax.grad(krylov)(parameter),
            jax.grad(adaptive)(parameter),
        )

    with jax.default_device(gpu):
        result = jax.jit(evaluate)(jnp.asarray(0.3, dtype))
        jax.block_until_ready(result)

    assert all(
        leaf.dtype == dtype and leaf.devices().pop().platform in GPU_PLATFORMS
        for leaf in result
    )
    grouped = jnp.asarray(result).reshape(3, 3)
    assert jnp.allclose(grouped, grouped[:, :1], atol=2e-5)


@DTYPES
def test_handcoded_linear_exponential_jvp_vjp_batches_on_gpu(dtype):
    gpu = gpu_devices()[0]
    rates = jnp.linspace(0.1, 1.0, 32, dtype=dtype)
    initial = jnp.linspace(1.0, 2.0, 32, dtype=dtype)
    initial = initial / jnp.sum(initial)
    directions = jnp.stack([jnp.roll(initial, shift) - initial for shift in (1, 3, 7)])
    cotangents = jnp.stack([jnp.sin(rates), jnp.cos(rates), rates - jnp.mean(rates)])
    method = KrylovExponential(krylov_dim=16, num_substeps=2)

    def operator(state):
        flux = rates * state
        return jnp.roll(flux, 1) - flux

    def evaluate(state, tangent_batch, cotangent_batch):
        jvp_solution, tangents = jvp_linear_ode(
            operator,
            method,
            0.0,
            2.0,
            state,
            tangent_batch,
            batched=True,
        )
        vjp_solution, gradients = vjp_linear_ode(
            operator,
            method,
            0.0,
            2.0,
            state,
            cotangent_batch,
            batched=True,
        )
        return jvp_solution, tangents, vjp_solution, gradients

    with jax.default_device(gpu):
        result = jax.jit(evaluate)(initial, directions, cotangents)
        jax.block_until_ready(result)

    jvp_solution, tangents, vjp_solution, gradients = result
    assert bool(jvp_solution.ok & vjp_solution.ok)
    assert tangents.dtype == dtype and gradients.dtype == dtype
    for leaf in jax.tree.leaves(result):
        assert leaf.devices().pop().platform in GPU_PLATFORMS
        assert bool(jnp.all(jnp.isfinite(leaf)))
    pairing_forward = jnp.einsum("bi,ci->bc", tangents, cotangents)
    pairing_reverse = jnp.einsum("bi,ci->bc", directions, gradients)
    assert jnp.allclose(pairing_forward, pairing_reverse, rtol=3e-4, atol=3e-5)


def gpu_bytes_limit():
    devices = gpu_devices()
    if not devices:
        return 0
    stats = devices[0].memory_stats() or {}
    return stats.get("bytes_limit", 0)


LARGE_GPU = pytest.mark.skipif(
    gpu_bytes_limit() < 40 * 1024**3
    and os.environ.get("TINYDIFFEQ_LARGE_GPU_TESTS") != "1",
    reason="Requires >=40 GiB GPU pool or TINYDIFFEQ_LARGE_GPU_TESTS=1",
)


pytestmark = pytest.mark.skipif(
    not gpu_devices(),
    reason="JAX GPU backend is not available",
)


@DTYPES
def test_jitted_adaptive_solve_runs_on_gpu(dtype):
    gpu = gpu_devices()[0]
    rtol = 1e-5 if dtype == jnp.float32 else 1e-6
    atol = 1e-7 if dtype == jnp.float32 else 1e-8

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
            controller=IController(rtol=rtol, atol=atol),
            max_steps=128,
            save_at=SaveAt(steps=True),
        )

    with jax.default_device(gpu):
        x_0 = jnp.asarray([1.0, 2.0], dtype)
        p = jnp.asarray(1.3, dtype)
        sol = run(x_0, p)
        jax.block_until_ready(sol)

    assert bool(sol.ok)
    assert sol.xs.dtype == dtype
    assert next(iter(sol.xs.devices())).platform in GPU_PLATFORMS
    assert bool(jnp.all(jnp.isfinite(sol.xs[sol.accepted])))


@DTYPES
def test_gradient_through_solve_on_gpu(dtype):
    gpu = gpu_devices()[0]
    rtol = 1e-5 if dtype == jnp.float32 else 1e-8
    atol = 1e-7 if dtype == jnp.float32 else 1e-10

    def endpoint(p):
        return solve_ode(
            lambda x, t, args, q: -q * x,
            Tsit5(),
            0.0,
            1.0,
            jnp.asarray(1.0, dtype),
            p=p,
            dt_0=0.1,
            controller=IController(rtol=rtol, atol=atol),
            max_steps=128,
        ).xs

    with jax.default_device(gpu):
        grad = jax.jit(jax.grad(endpoint))(jnp.asarray(1.3, dtype))
        grad.block_until_ready()

    assert grad.dtype == dtype
    assert next(iter(grad.devices())).platform in GPU_PLATFORMS
    assert bool(jnp.isfinite(grad))


@DTYPES
def test_dae_value_and_gradient_run_on_gpu(dtype):
    gpu = gpu_devices()[0]

    def endpoint(p):
        return solve_semi_explicit_dae(
            lambda y, z, t, args, q: q * z,
            lambda y, z: z - y,
            Tsit5(),
            0.0,
            1.0,
            jnp.asarray(1.0, dtype),
            jnp.asarray(0.5, dtype),
            p=p,
            dt_0=0.1,
            controller=IController(rtol=1e-5, atol=1e-7),
            max_steps=64,
        ).ys

    with jax.default_device(gpu):
        value, grad = jax.jit(lambda p: (endpoint(p), jax.grad(endpoint)(p)))(
            jnp.asarray(1.3, dtype)
        )
        jax.block_until_ready((value, grad))

    assert value.dtype == dtype and grad.dtype == dtype
    assert next(iter(value.devices())).platform in GPU_PLATFORMS
    assert next(iter(grad.devices())).platform in GPU_PLATFORMS
    assert bool(jnp.isfinite(value))
    assert bool(jnp.isfinite(grad))


@DTYPES
def test_pytree_ode_and_sde_run_on_gpu(dtype):
    gpu = gpu_devices()[0]
    x_0 = {"a": jnp.asarray(1.0, dtype), "b": jnp.asarray([2.0, 3.0], dtype)}

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
        leaf.dtype == dtype and leaf.devices().pop().platform in GPU_PLATFORMS
        for leaf in jax.tree.leaves((ode.xs, sde.xs))
    )
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(sde.xs))


@DTYPES
def test_interpolated_aux_and_sdae_ad_run_on_gpu(dtype):
    gpu = gpu_devices()[0]

    def dae_output(p):
        def differential(y, z, t, args, q, algebraic_aux):
            return q * z, algebraic_aux

        sol = solve_semi_explicit_dae(
            differential,
            lambda y, z, t, args, q: (
                z - y,
                {"value": q * z + y},
            ),
            Tsit5(),
            0.0,
            1.0,
            jnp.asarray(1.0, dtype),
            jnp.asarray(0.8, dtype),
            p=p,
            dt_0=0.1,
            controller=IController(),
            max_steps=64,
            save_at=SaveAt(ts=jnp.linspace(0.0, 1.0, 9, dtype=dtype)),
            has_aux=True,
            has_algebraic_aux=True,
        )
        return jnp.sum(sol.zs + sol.aux["value"])

    def sdae_output(p):
        def stochastic_drift(y, z, t, args, q, algebraic_aux):
            return q * z, algebraic_aux

        def stochastic_diffusion(y, z, t, args, q, algebraic_aux):
            return jnp.asarray(0.1, dtype) * z

        sol = solve_semi_explicit_sdae(
            stochastic_drift,
            stochastic_diffusion,
            lambda y, z, t, args, q: (z - y, {"value": q * z}),
            EulerMaruyama(),
            0.0,
            1.0,
            jnp.asarray(1.0, dtype),
            jnp.asarray(0.8, dtype),
            p=p,
            key=jax.random.key(0),
            n_steps=16,
            has_aux=True,
            has_algebraic_aux=True,
        )
        return sol.ys + sol.aux["value"]

    with jax.default_device(gpu):
        p = jnp.asarray(0.2, dtype)
        values = jax.jit(lambda q: (dae_output(q), sdae_output(q)))(p)
        gradients = jax.jit(
            lambda q: (jax.grad(dae_output)(q), jax.grad(sdae_output)(q))
        )(p)
        jax.block_until_ready((values, gradients))

    for leaf in jax.tree.leaves((values, gradients)):
        assert leaf.dtype == dtype
        assert leaf.devices().pop().platform in GPU_PLATFORMS
        assert bool(jnp.isfinite(leaf))


@DTYPES
def test_rodas5p_ode_dae_and_derivatives_run_on_gpu(dtype):
    gpu = gpu_devices()[0]

    def ode_endpoint(rate):
        return solve_ode(
            lambda x, t, args, p: p * x,
            Rodas5P(),
            dtype(0.0),
            dtype(1.0),
            jnp.asarray(1.0, dtype),
            p=rate,
            dt_0=dtype(0.25),
            max_steps=4,
        ).xs

    def dae_endpoint(rate):
        return solve_semi_explicit_dae(
            lambda y, z, t, args, p: p * z,
            lambda y, z: z**2 - y - dtype(2.0),
            Rodas5P(),
            dtype(0.0),
            dtype(0.5),
            jnp.asarray(1.0, dtype),
            jnp.sqrt(jnp.asarray(3.0, dtype)),
            p=rate,
            dt_0=dtype(0.125),
            max_steps=4,
        ).ys

    def evaluate(rate):
        ode_value, ode_tangent = jax.jvp(
            ode_endpoint,
            (rate,),
            (jnp.ones_like(rate),),
        )
        return (
            ode_value,
            ode_tangent,
            jax.grad(ode_endpoint)(rate),
            dae_endpoint(rate),
            jax.grad(dae_endpoint)(rate),
        )

    with jax.default_device(gpu):
        result = jax.jit(evaluate)(jnp.asarray(-0.2, dtype))
        jax.block_until_ready(result)

    for leaf in jax.tree.leaves(result):
        assert leaf.dtype == dtype
        assert leaf.devices().pop().platform in GPU_PLATFORMS
        assert bool(jnp.isfinite(leaf))


DECAY = 0.2


@DTYPES
def test_adaptive_ode_endpoint_on_gpu(dtype):
    gpu = gpu_devices()[0]

    def f(x, t, args, p):
        return -p * x

    @jax.jit
    def run(x_0, p):
        # SaveAt defaults to t_1=True, so xs IS the endpoint.
        return solve_ode(
            f, Tsit5(), 0.0, 1.0, x_0, p=p, dt_0=0.1, controller=IController()
        ).xs

    with jax.default_device(gpu):
        out = jax.block_until_ready(
            run(jnp.asarray(1.0, dtype), jnp.asarray(DECAY, dtype))
        )

    assert out.dtype == dtype
    assert out.devices().pop().platform in GPU_PLATFORMS
    assert bool(jnp.isfinite(out))
    assert abs(float(out) - float(jnp.exp(-DECAY))) < 1e-3


@DTYPES
def test_sde_runs_on_gpu(dtype):
    gpu = gpu_devices()[0]

    @jax.jit
    def run(x_0):
        return solve_sde(
            lambda x, t, args, p: -DECAY * x,
            lambda x, t, args, p: jnp.asarray(0.1, x.dtype) * jnp.ones_like(x),
            EulerMaruyama(),
            0.0,
            1.0,
            x_0,
            key=jax.random.key(0),
            n_steps=64,
        ).xs

    with jax.default_device(gpu):
        out = jax.block_until_ready(run(jnp.asarray(1.0, dtype)))

    assert out.dtype == dtype
    assert out.devices().pop().platform in GPU_PLATFORMS
    assert bool(jnp.isfinite(out))
    assert abs(float(out)) < 10.0  # a diverged path, not a tolerance check


@DTYPES
def test_dae_endpoint_value_and_grad_run_on_gpu(dtype):
    gpu = gpu_devices()[0]

    def f(state, z):
        return -DECAY * z

    def g(y, z, t, args, p):
        return z - p * y

    def endpoint(p):
        return jnp.sum(
            solve_semi_explicit_dae(
                f,
                g,
                Tsit5(),
                0.0,
                1.0,
                jnp.asarray([1.0], dtype),
                jnp.asarray([1.0], dtype),
                p=p,
                dt_0=0.1,
                controller=IController(),
            ).ys[-1]
        )

    with jax.default_device(gpu):
        value, grad = jax.block_until_ready(
            jax.jit(lambda p: (endpoint(p), jax.grad(endpoint)(p)))(
                jnp.asarray(1.0, dtype)
            )
        )

    for leaf in (value, grad):
        assert leaf.dtype == dtype
        assert leaf.devices().pop().platform in GPU_PLATFORMS
        assert bool(jnp.isfinite(leaf))


@DTYPES
def test_fixed_euler_rk4_ode_on_gpu(dtype):
    gpu = gpu_devices()[0]
    n = 128

    @jax.jit
    def run(x_0):
        euler = solve_ode(
            lambda x: -DECAY * x, Euler(), 0.0, 1.0, x_0, dt_0=1.0 / n, max_steps=n
        ).xs
        rk4 = solve_ode(
            lambda x: -DECAY * x, RK4(), 0.0, 1.0, x_0, dt_0=1.0 / n, max_steps=n
        ).xs
        return euler, rk4

    with jax.default_device(gpu):
        euler, rk4 = jax.block_until_ready(run(jnp.asarray(1.0, dtype)))

    exact = float(jnp.exp(-DECAY))
    for out, tol in ((euler, 1e-3), (rk4, 1e-5)):
        assert out.dtype == dtype
        assert out.devices().pop().platform in GPU_PLATFORMS
        assert abs(float(out) - exact) < tol


@DTYPES
@pytest.mark.parametrize(
    ("solver", "diffusion"),
    [
        (Milstein(), lambda x: 0.1 * x),
        (SRA1(), lambda x: 0.1 * jnp.ones_like(x)),
    ],
    ids=["milstein", "sra1"],
)
def test_milstein_and_sra1_run_on_gpu(solver, diffusion, dtype):
    gpu = gpu_devices()[0]

    @jax.jit
    def run(x_0):
        return solve_sde(
            lambda x: -DECAY * x,
            diffusion,
            solver,
            0.0,
            1.0,
            x_0,
            key=jax.random.key(0),
            n_steps=64,
        ).xs

    with jax.default_device(gpu):
        out = jax.block_until_ready(run(jnp.asarray(1.0, dtype)))

    assert out.dtype == dtype
    assert out.devices().pop().platform in GPU_PLATFORMS
    assert bool(jnp.isfinite(out))
    assert abs(float(out)) < 10.0


# The kernels use case: vmap over B trajectories with per-trajectory x_0 and
# explicit noise, jit the whole batch, and take reverse-mode gradients of a
# scalar residual with respect to (x_0, p, noise). n_steps covers 31 and 127
# so both a scan XLA may unroll and one it will not take the same paths.


def sra1_batch_noise(batch, n_steps, dtype, seed):
    keys = jax.random.split(jax.random.key(seed), batch)
    dt = jnp.asarray(1.0 / n_steps, dtype)
    return jax.vmap(
        lambda k: SRA1().sample_noise(jnp.zeros((), dtype), k, n_steps, dt, dtype)
    )(keys)


@DTYPES
@pytest.mark.parametrize("n_steps", [31, 127])
def test_vmapped_sra1_ensemble_matches_per_trajectory_on_gpu(n_steps, dtype):
    gpu = gpu_devices()[0]
    batch = 8
    x_0s = jnp.linspace(0.5, 2.0, batch, dtype=dtype)
    noise = sra1_batch_noise(batch, n_steps, dtype, seed=3)

    def one(x_0, w, z):
        return solve_sde(
            lambda x: -DECAY * x,
            lambda x: jnp.asarray(0.1, x.dtype) * jnp.ones_like(x),
            SRA1(),
            0.0,
            1.0,
            x_0,
            noise=(w, z),
            n_steps=n_steps,
        ).xs

    with jax.default_device(gpu):
        batched = jax.block_until_ready(jax.jit(jax.vmap(one))(x_0s, *noise))
        looped = jnp.stack(
            [one(x_0s[i], noise[0][i], noise[1][i]) for i in range(batch)]
        )

    assert batched.dtype == dtype
    assert batched.devices().pop().platform in GPU_PLATFORMS
    assert jnp.allclose(batched, looped, atol=1e-5)


@DTYPES
@pytest.mark.parametrize("n_steps", [31, 127])
def test_grad_of_vmapped_sra1_ensemble_on_gpu(n_steps, dtype):
    gpu = gpu_devices()[0]
    batch = 8
    x_0s = jnp.linspace(0.5, 2.0, batch, dtype=dtype)
    theta = jnp.asarray(0.7, dtype)
    noise = sra1_batch_noise(batch, n_steps, dtype, seed=11)

    def endpoint(x_0, p, w, z):
        return solve_sde(
            lambda x, t, args, p: -p * x,
            lambda x, t, args, p: jnp.asarray(0.1, x.dtype) * jnp.ones_like(x),
            SRA1(),
            0.0,
            1.0,
            x_0,
            p=p,
            noise=(w, z),
            n_steps=n_steps,
        ).xs

    def loss(x_0s, p, noise):
        endpoints = jax.vmap(lambda x_0, w, z: endpoint(x_0, p, w, z))(x_0s, *noise)
        return jnp.sum((endpoints - 1.0) ** 2)

    with jax.default_device(gpu):
        value, (grad_x, grad_p, grad_noise) = jax.block_until_ready(
            jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2)))(x_0s, theta, noise)
        )

    for leaf in jax.tree.leaves((value, grad_x, grad_p, grad_noise)):
        assert leaf.dtype == dtype
        assert leaf.devices().pop().platform in GPU_PLATFORMS
        assert bool(jnp.all(jnp.isfinite(leaf)))

    def one_loss(x_0, w, z):
        return (endpoint(x_0, theta, w, z) - 1.0) ** 2

    per_trajectory = jnp.stack(
        [jax.grad(one_loss)(x_0s[i], noise[0][i], noise[1][i]) for i in range(batch)]
    )
    assert jnp.allclose(grad_x, per_trajectory, atol=1e-4)


@DTYPES
@pytest.mark.parametrize("n_steps", [31, 127])
def test_grad_of_vmapped_rk4_ensemble_on_gpu(n_steps, dtype):
    gpu = gpu_devices()[0]
    batch = 8
    x_0s = jnp.linspace(0.5, 2.0, batch, dtype=dtype)
    theta = jnp.asarray(0.7, dtype)

    def endpoint(x_0, p):
        return solve_ode(
            lambda x, t, args, p: -p * x,
            RK4(),
            0.0,
            1.0,
            x_0,
            p=p,
            dt_0=1.0 / n_steps,
            max_steps=n_steps,
        ).xs

    def loss(x_0s, p):
        endpoints = jax.vmap(lambda x_0: endpoint(x_0, p))(x_0s)
        return jnp.sum((endpoints - 1.0) ** 2)

    with jax.default_device(gpu):
        grad_x, grad_p = jax.block_until_ready(
            jax.jit(jax.grad(loss, argnums=(0, 1)))(x_0s, theta)
        )

    decay = jnp.exp(-theta)
    expected_x = 2.0 * (x_0s * decay - 1.0) * decay
    expected_p = jnp.sum(-2.0 * (x_0s * decay - 1.0) * x_0s * decay)
    for leaf in (grad_x, grad_p):
        assert leaf.dtype == dtype
        assert leaf.devices().pop().platform in GPU_PLATFORMS
    assert jnp.allclose(grad_x, expected_x, atol=1e-3)
    assert jnp.allclose(grad_p, expected_p, atol=1e-2)


@DTYPES
def test_bvp_value_grad_and_vmap_on_gpu(dtype):
    gpu = gpu_devices()[0]
    t = jnp.linspace(0.0, 1.0, 5, dtype=dtype)
    y_0 = jnp.zeros((5, 2), dtype=dtype)
    tol = 1e-4 if dtype == jnp.float32 else 1e-6

    def fun(t, y, z, args, p):
        return jnp.array([y[1], p * y[0]], dtype=y.dtype)

    def bc(ya, yb, z, args, p):
        return jnp.array([ya[0] - p, yb[0]], dtype=ya.dtype)

    def solve_one(p):
        return solve_bvp(fun, bc, t, y_0, p=p, tol=tol, max_nodes=32)

    def loss(p):
        return jnp.sum(solve_one(p).y[0] ** 2)

    with jax.default_device(gpu):
        p = jnp.asarray(1.0, dtype)
        ps = jnp.linspace(0.5, 1.5, 4, dtype=dtype)
        value, grad = jax.block_until_ready(jax.jit(jax.value_and_grad(loss))(p))
        batched = jax.block_until_ready(jax.jit(jax.vmap(solve_one))(ps))

    for leaf in (value, grad, batched.y):
        assert leaf.dtype == dtype
        assert leaf.devices().pop().platform in GPU_PLATFORMS
        assert bool(jnp.all(jnp.isfinite(leaf)))
    assert bool(jnp.all(batched.status == 0))


@pytest.mark.skipif(
    not gpu_supports_float64(),
    reason="Active GPU backend does not support float64",
)
def test_float32_y_with_float64_z_keeps_the_tight_root_tolerance_on_gpu():
    # dae.py picks root_atol from z_dtype alone: 1e-10 when z has more than 32
    # bits, 1e-6 otherwise. A float32 y with a float64 z therefore gets the
    # tight bar, and both dtypes must survive the round trip on device.
    gpu = gpu_devices()[0]

    def f(state, z):
        return -DECAY * state

    def g(y, z, t, args, p):
        return z - jnp.asarray(y, z.dtype)

    @jax.jit
    def run(y_0, z_0):
        solution = solve_semi_explicit_dae(
            f,
            g,
            Tsit5(),
            0.0,
            1.0,
            y_0,
            z_0,
            dt_0=0.1,
            controller=IController(),
        )
        return solution.ys[-1], solution.zs[-1]

    with jax.default_device(gpu):
        ys, zs = jax.block_until_ready(
            run(jnp.asarray([1.0], jnp.float32), jnp.asarray([1.0], jnp.float64))
        )

    assert ys.dtype == jnp.float32
    assert zs.dtype == jnp.float64
    assert bool(jnp.all(jnp.isfinite(ys))) and bool(jnp.all(jnp.isfinite(zs)))


@LARGE_GPU
@DTYPES
def test_large_vmapped_ode_and_sde_ensembles_on_gpu(dtype):
    gpu = gpu_devices()[0]
    batch = 16384
    n_steps = 2048
    dim = 2
    dt = jnp.asarray(1.0 / n_steps, dtype)
    theta = jnp.asarray(0.7, dtype)
    x_0s = jnp.linspace(0.5, 2.0, batch * dim, dtype=dtype).reshape(batch, dim)

    def ode_mean(x_0s, p):
        endpoints = jax.vmap(
            lambda x_0: (
                solve_ode(
                    lambda x, t, args, q: -q * x,
                    RK4(),
                    0.0,
                    1.0,
                    x_0,
                    p=p,
                    dt_0=1.0 / n_steps,
                    max_steps=n_steps,
                ).xs
            )
        )(x_0s)
        return jnp.mean(endpoints)

    def run_sde(solver, seed):
        keys = jax.random.split(jax.random.key(seed), batch)
        noise = jax.vmap(
            lambda k: solver.sample_noise(
                jnp.zeros((dim,), dtype), k, n_steps, dt, dtype
            )
        )(keys)

        def sde_mean(x_0s, p, w):
            endpoints = jax.vmap(
                lambda x_0, w_i: (
                    solve_sde(
                        lambda x, t, args, q: -q * x,
                        lambda x, t, args, q: jnp.full_like(x, 0.1),
                        solver,
                        0.0,
                        1.0,
                        x_0,
                        p=p,
                        noise=w_i,
                        n_steps=n_steps,
                    ).xs
                )
            )(x_0s, w)
            return jnp.mean(endpoints)

        return jax.jit(jax.value_and_grad(sde_mean, argnums=(0, 1, 2)))(
            x_0s, theta, noise
        )

    with jax.default_device(gpu):
        ode_val, (ode_grad_x, ode_grad_p) = jax.block_until_ready(
            jax.jit(jax.value_and_grad(ode_mean, argnums=(0, 1)))(x_0s, theta)
        )
        em_val, (em_grad_x, em_grad_p, em_grad_w) = jax.block_until_ready(
            run_sde(EulerMaruyama(), seed=17)
        )
        sra1_val, (sra1_grad_x, sra1_grad_p, sra1_grad_w) = jax.block_until_ready(
            run_sde(SRA1(), seed=23)
        )

    decay = jnp.exp(-theta)
    expected_mean = jnp.mean(x_0s) * decay
    expected_grad_p = -expected_mean
    expected_grad_x = jnp.full_like(x_0s, decay / x_0s.size)

    for leaf in jax.tree.leaves(
        (
            ode_val,
            ode_grad_x,
            ode_grad_p,
            em_val,
            em_grad_x,
            em_grad_p,
            em_grad_w,
            sra1_val,
            sra1_grad_x,
            sra1_grad_p,
            sra1_grad_w,
        )
    ):
        assert leaf.dtype == dtype
        assert leaf.devices().pop().platform in GPU_PLATFORMS
        assert bool(jnp.all(jnp.isfinite(leaf)))

    ode_tol = 1e-4 if dtype == jnp.float32 else 1e-8
    assert jnp.allclose(ode_val, expected_mean, atol=ode_tol)
    assert jnp.allclose(ode_grad_p, expected_grad_p, atol=ode_tol)
    assert jnp.allclose(ode_grad_x, expected_grad_x, atol=1e-6)
    for val, grad_p, grad_x in (
        (em_val, em_grad_p, em_grad_x),
        (sra1_val, sra1_grad_p, sra1_grad_x),
    ):
        assert jnp.allclose(val, expected_mean, atol=3e-3)
        assert jnp.allclose(grad_p, expected_grad_p, atol=3e-3)
        assert jnp.allclose(grad_x, expected_grad_x, atol=1e-5)
