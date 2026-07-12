import jax
import jax.numpy as jnp
import pytest

from tinydiffeq import (
    AdaptiveKrylovExponential,
    AssociativeMarkov,
    ContinuousTimeMarkovChain,
    DenseExponential,
    DiscreteMarkovChain,
    EulerMaruyama,
    IController,
    KrylovExponential,
    MatrixFreeContinuousTimeMarkovChain,
    Rodas5P,
    SaveAt,
    Tsit5,
    forecast_continuous_time_markov_chain,
    forecast_markov_chain,
    jvp_linear_ode,
    simulate_continuous_time_markov_chain,
    simulate_markov_chain,
    solve_linear_ode,
    solve_ode,
    solve_sde,
    solve_semi_explicit_dae,
    solve_semi_explicit_sdae,
    vjp_linear_ode,
)


def test_markov_methods_jit_and_vmap_on_gpu():
    gpu = gpu_devices()[0]
    discrete = DiscreteMarkovChain([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.3, 0.5]])
    continuous = ContinuousTimeMarkovChain(
        [[-1.0, 0.7, 0.3], [0.2, -0.6, 0.4], [0.5, 0.5, -1.0]]
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

    assert next(iter(discrete_paths.devices())).platform == "gpu"
    assert discrete_paths.shape == (64, 129)
    assert bool(jnp.all(continuous_endpoints.ok))


def test_markov_distribution_forecasts_and_ad_on_gpu():
    gpu = gpu_devices()[0]
    discrete = DiscreteMarkovChain([[0.8, 0.2], [0.3, 0.7]])
    continuous = ContinuousTimeMarkovChain([[-2.0, 2.0], [1.0, -1.0]])
    initial = jnp.asarray([0.4, 0.6])

    def objective(distribution):
        forecast = forecast_continuous_time_markov_chain(
            continuous,
            0.0,
            2.0,
            distribution,
            method=DenseExponential(),
        ).probabilities
        return jnp.asarray([1.0, -0.5]) @ forecast

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

    assert next(iter(discrete_path.devices())).platform == "gpu"
    assert jnp.allclose(discrete_path.sum(axis=1), 1.0)
    assert jnp.isfinite(continuous_value)
    assert bool(jnp.all(jnp.isfinite(continuous_gradient)))


def test_matrix_free_pytree_markov_forecast_and_ad_on_gpu():
    gpu = gpu_devices()[0]
    generator = jnp.asarray(
        [[-1.0, 0.8, 0.2], [0.3, -0.7, 0.4], [0.1, 0.2, -0.3]],
        jnp.float32,
    )

    def action(probabilities):
        flat = jnp.concatenate([probabilities["first"], probabilities["second"]])
        result = flat @ generator
        return {"first": result[:2], "second": result[2:]}

    chain = MatrixFreeContinuousTimeMarkovChain(action)
    initial = {
        "first": jnp.asarray([0.2, 0.3], jnp.float32),
        "second": jnp.asarray([0.5], jnp.float32),
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

    assert next(iter(value.devices())).platform == "gpu"
    assert all(
        leaf.devices().pop().platform == "gpu" for leaf in jax.tree.leaves(gradient)
    )
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(gradient))


def test_dense_and_matrix_free_linear_exponential_ad_on_gpu():
    gpu = gpu_devices()[0]
    base = jnp.asarray(
        [[-0.8, 0.2, 0.0], [0.3, -0.5, 0.1], [0.0, 0.4, -0.6]],
        jnp.float32,
    )
    direction = jnp.asarray(
        [[0.1, 0.0, -0.1], [0.0, -0.2, 0.2], [0.1, 0.0, -0.1]],
        jnp.float32,
    )
    initial = {"left": jnp.asarray([0.2, -0.1]), "right": jnp.asarray([0.5])}

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
        result = jax.jit(evaluate)(jnp.asarray(0.3, jnp.float32))
        jax.block_until_ready(result)

    assert all(leaf.devices().pop().platform == "gpu" for leaf in result)
    grouped = jnp.asarray(result).reshape(3, 3)
    assert jnp.allclose(grouped, grouped[:, :1], atol=2e-5)


def test_handcoded_linear_exponential_jvp_vjp_batches_on_gpu():
    gpu = gpu_devices()[0]
    rates = jnp.linspace(0.1, 1.0, 32, dtype=jnp.float32)
    initial = jnp.linspace(1.0, 2.0, 32, dtype=jnp.float32)
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
    for leaf in jax.tree.leaves(result):
        assert leaf.devices().pop().platform == "gpu"
        assert bool(jnp.all(jnp.isfinite(leaf)))
    pairing_forward = jnp.einsum("bi,ci->bc", tangents, cotangents)
    pairing_reverse = jnp.einsum("bi,ci->bc", directions, gradients)
    assert jnp.allclose(pairing_forward, pairing_reverse, rtol=3e-4, atol=3e-5)


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


def test_interpolated_aux_and_sdae_ad_run_on_gpu():
    gpu = gpu_devices()[0]
    dtype = jnp.float32

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
        assert leaf.devices().pop().platform == "gpu"
        assert bool(jnp.isfinite(leaf))


def test_rodas5p_ode_dae_and_derivatives_run_on_gpu():
    gpu = gpu_devices()[0]
    dtype = jnp.float32

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
        assert leaf.devices().pop().platform == "gpu"
        assert bool(jnp.isfinite(leaf))
