import jax
import jax.numpy as jnp
import pytest

from tinydiffeq import (
    AdaptiveKrylovExponential,
    AssociativeMarkov,
    ContinuousTimeMarkovChain,
    DenseExponential,
    DiscreteMarkovChain,
    KrylovExponential,
    MatrixFreeContinuousTimeMarkovChain,
    MatrixPowerMarkov,
    SaveAt,
    SequentialMarkov,
    forecast_continuous_time_markov_chain,
    forecast_markov_chain,
)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_dtmc_endpoint_methods_match_matrix_power(dtype):
    transition = jnp.asarray([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3], [0.2, 0.1, 0.7]], dtype)
    chain = DiscreteMarkovChain(transition)
    distribution_0 = jnp.asarray([0.2, 0.3, 0.5], dtype)
    expected = distribution_0 @ jnp.linalg.matrix_power(transition, 37)
    for method in (
        MatrixPowerMarkov(),
        SequentialMarkov(),
        AssociativeMarkov(),
    ):
        result = forecast_markov_chain(
            chain, distribution_0, num_steps=37, method=method
        )
        assert bool(result.ok)
        assert jnp.allclose(result.probabilities, expected, rtol=2e-5, atol=2e-6)


def test_dtmc_full_forecast_methods_and_queries_match():
    chain = DiscreteMarkovChain([[0.8, 0.2], [0.3, 0.7]])
    distribution_0 = jnp.asarray([1.0, 0.0])
    sequential = forecast_markov_chain(
        chain,
        distribution_0,
        num_steps=32,
        method=SequentialMarkov(),
        save_at=SaveAt(steps=True),
    )
    associative = forecast_markov_chain(
        chain,
        distribution_0,
        num_steps=32,
        method=AssociativeMarkov(),
        save_at=SaveAt(steps=True),
    )
    selected = forecast_markov_chain(
        chain,
        distribution_0,
        num_steps=32,
        save_at=SaveAt(ts=jnp.asarray([0, 5, 32], jnp.int32)),
    )
    assert jnp.allclose(sequential.probabilities, associative.probabilities)
    indices = jnp.asarray([0, 5, 32])
    assert jnp.allclose(selected.probabilities, sequential.probabilities[indices])
    assert jnp.allclose(sequential.probabilities.sum(axis=1), 1.0)
    assert bool(sequential.ok) and bool(associative.ok) and bool(selected.ok)


def test_dtmc_zero_steps_and_method_validation():
    chain = DiscreteMarkovChain([[0.5, 0.5], [0.5, 0.5]])
    distribution_0 = jnp.asarray([0.25, 0.75])
    endpoint = forecast_markov_chain(chain, distribution_0, num_steps=0)
    steps = forecast_markov_chain(
        chain,
        distribution_0,
        num_steps=0,
        save_at=SaveAt(steps=True),
    )
    assert jnp.array_equal(endpoint.probabilities, distribution_0)
    assert jnp.array_equal(steps.probabilities, distribution_0[None])
    with pytest.raises(ValueError, match="endpoint"):
        forecast_markov_chain(
            chain,
            distribution_0,
            num_steps=2,
            method=MatrixPowerMarkov(),
            save_at=SaveAt(steps=True),
        )


def test_dtmc_jvp_vjp_and_vmap_wrt_initial_distribution():
    transition = jnp.asarray([[0.9, 0.1], [0.25, 0.75]])
    chain = DiscreteMarkovChain(transition)
    transition_power = jnp.linalg.matrix_power(transition, 12)
    distribution_0 = jnp.asarray([0.4, 0.6])
    tangent_0 = jnp.asarray([0.3, -0.3])

    def endpoint(distribution):
        return forecast_markov_chain(chain, distribution, num_steps=12).probabilities

    value, tangent = jax.jvp(endpoint, (distribution_0,), (tangent_0,))
    weights = jnp.asarray([1.2, -0.7])
    gradient = jax.grad(lambda distribution: weights @ endpoint(distribution))(
        distribution_0
    )
    batch = jnp.asarray([[1.0, 0.0], [0.0, 1.0], [0.4, 0.6]])
    batched = jax.jit(jax.vmap(endpoint))(batch)

    assert jnp.allclose(value, distribution_0 @ transition_power)
    assert jnp.allclose(tangent, tangent_0 @ transition_power)
    assert jnp.allclose(gradient, transition_power @ weights)
    assert jnp.allclose(batched, batch @ transition_power)


def test_invalid_initial_distribution_reports_not_ok_with_finite_output():
    chain = DiscreteMarkovChain([[0.8, 0.2], [0.3, 0.7]])
    for distribution in (
        jnp.asarray([0.2, 0.2]),
        jnp.asarray([-0.1, 1.1]),
        jnp.asarray([jnp.nan, 1.0]),
    ):
        result = forecast_markov_chain(chain, distribution, num_steps=4)
        assert not bool(result.ok)
        assert bool(jnp.all(jnp.isfinite(result.probabilities)))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_two_state_ctmc_closed_form_and_grid(dtype):
    rate_01 = jnp.asarray(2.0, dtype)
    rate_10 = jnp.asarray(1.0, dtype)
    chain = ContinuousTimeMarkovChain(
        jnp.asarray([[-rate_01, rate_01], [rate_10, -rate_10]], dtype)
    )
    distribution_0 = jnp.asarray([1.0, 0.0], dtype)
    horizon = jnp.asarray(0.8, dtype)
    endpoint = forecast_continuous_time_markov_chain(
        chain, 0.0, horizon, distribution_0
    )
    grid = jnp.linspace(jnp.asarray(0.0, dtype), horizon, 17)
    trajectory = forecast_continuous_time_markov_chain(
        chain,
        0.0,
        horizon,
        distribution_0,
        save_at=SaveAt(ts=grid),
    )
    state_1 = (
        rate_01 / (rate_01 + rate_10) * (1 - jnp.exp(-(rate_01 + rate_10) * horizon))
    )
    expected = jnp.asarray([1 - state_1, state_1], dtype)
    tolerance = 2e-5 if dtype == jnp.float32 else 1e-12
    assert jnp.allclose(endpoint.probabilities, expected, atol=tolerance)
    assert jnp.allclose(trajectory.probabilities[0], distribution_0, atol=tolerance)
    assert jnp.allclose(trajectory.probabilities[-1], expected, atol=tolerance)
    assert jnp.allclose(trajectory.probabilities.sum(axis=1), 1.0, atol=tolerance)
    assert bool(endpoint.ok) and bool(trajectory.ok)


def test_ctmc_stationary_distribution_jvp_vjp_and_vmap():
    generator = jnp.asarray([[-2.0, 2.0], [1.0, -1.0]])
    chain = ContinuousTimeMarkovChain(generator)
    stationary = jnp.asarray([1.0 / 3.0, 2.0 / 3.0])
    transition = jax.scipy.linalg.expm(1.7 * generator)

    def endpoint(distribution):
        return forecast_continuous_time_markov_chain(
            chain, 0.0, 1.7, distribution
        ).probabilities

    tangent_0 = jnp.asarray([0.2, -0.2])
    value, tangent = jax.jvp(endpoint, (stationary,), (tangent_0,))
    weights = jnp.asarray([0.4, -1.1])
    gradient = jax.grad(lambda distribution: weights @ endpoint(distribution))(
        stationary
    )
    batch = jnp.asarray([[1.0, 0.0], [0.0, 1.0], stationary])
    batched = jax.jit(jax.vmap(endpoint))(batch)

    assert jnp.allclose(value, stationary)
    assert jnp.allclose(tangent, tangent_0 @ transition)
    assert jnp.allclose(gradient, transition @ weights)
    assert jnp.allclose(batched, batch @ transition)


def test_ctmc_forecast_validation():
    chain = ContinuousTimeMarkovChain([[-1.0, 1.0], [1.0, -1.0]])
    distribution_0 = jnp.asarray([1.0, 0.0])
    with pytest.raises(ValueError, match="endpoint or SaveAt.ts"):
        forecast_continuous_time_markov_chain(
            chain,
            0.0,
            1.0,
            distribution_0,
            save_at=SaveAt(steps=True),
        )
    with pytest.raises(TypeError, match="DenseExponential"):
        forecast_continuous_time_markov_chain(
            chain,
            0.0,
            1.0,
            distribution_0,
            method=SequentialMarkov(),
        )
    outside = forecast_continuous_time_markov_chain(
        chain,
        0.0,
        1.0,
        distribution_0,
        method=DenseExponential(),
        save_at=SaveAt(ts=jnp.asarray([-0.1, 0.5])),
    )
    assert not bool(outside.ok)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_matrix_free_krylov_pytree_matches_dense_exponential(dtype):
    generator = jnp.asarray(
        [
            [-1.5, 1.0, 0.5, 0.0],
            [0.2, -0.9, 0.4, 0.3],
            [0.0, 0.6, -1.1, 0.5],
            [0.7, 0.0, 0.2, -0.9],
        ],
        dtype,
    )

    def flatten(probabilities):
        return jnp.concatenate(
            [probabilities["population"], probabilities["inventory"]["mass"]],
            axis=-1,
        )

    def unravel(probabilities):
        return {
            "population": probabilities[:2],
            "inventory": {"mass": probabilities[2:]},
        }

    def forward_generator(probabilities):
        return unravel(flatten(probabilities) @ generator)

    distribution_0 = unravel(jnp.asarray([0.1, 0.2, 0.3, 0.4], dtype))
    matrix_free_chain = MatrixFreeContinuousTimeMarkovChain(forward_generator)
    dense_chain = ContinuousTimeMarkovChain(generator)
    times = jnp.linspace(jnp.asarray(0.0, dtype), jnp.asarray(1.25, dtype), 8)

    dense = forecast_continuous_time_markov_chain(
        dense_chain,
        0.0,
        1.25,
        flatten(distribution_0),
        save_at=SaveAt(ts=times),
    )
    tolerance = 3e-5 if dtype == jnp.float32 else 2e-11
    for method in (
        KrylovExponential(krylov_dim=4),
        AdaptiveKrylovExponential(krylov_dim=4, max_steps=16),
    ):
        matrix_free = forecast_continuous_time_markov_chain(
            matrix_free_chain,
            0.0,
            1.25,
            distribution_0,
            method=method,
            save_at=SaveAt(ts=times),
        )
        assert bool(matrix_free.ok)
        assert jnp.allclose(
            flatten(matrix_free.probabilities), dense.probabilities, atol=tolerance
        )


def test_matrix_free_krylov_pytree_jvp_vjp_and_vmap():
    generator = jnp.asarray([[-2.0, 1.0, 1.0], [0.4, -1.0, 0.6], [0.2, 0.3, -0.5]])

    def action(probabilities):
        flat = jnp.concatenate([probabilities["active"], probabilities["rest"]])
        result = flat @ generator
        return {"active": result[:2], "rest": result[2:]}

    chain = MatrixFreeContinuousTimeMarkovChain(action)
    distribution_0 = {"active": jnp.asarray([0.2, 0.3]), "rest": jnp.asarray([0.5])}
    tangent_0 = {"active": jnp.asarray([0.1, -0.04]), "rest": jnp.asarray([-0.06])}
    method = KrylovExponential(krylov_dim=3)

    def endpoint(distribution):
        return forecast_continuous_time_markov_chain(
            chain, 0.0, 0.7, distribution, method=method
        ).probabilities

    value, tangent = jax.jvp(endpoint, (distribution_0,), (tangent_0,))
    transition = jax.scipy.linalg.expm(0.7 * generator)
    flat_initial = jnp.concatenate([distribution_0["active"], distribution_0["rest"]])
    flat_tangent = jnp.concatenate([tangent_0["active"], tangent_0["rest"]])
    flat_value = jnp.concatenate([value["active"], value["rest"]])
    flat_output_tangent = jnp.concatenate([tangent["active"], tangent["rest"]])
    weights = {"active": jnp.asarray([0.4, -0.2]), "rest": jnp.asarray([0.8])}

    def objective(distribution):
        result = endpoint(distribution)
        return jnp.vdot(weights["active"], result["active"]) + jnp.vdot(
            weights["rest"], result["rest"]
        )

    gradient = jax.grad(objective)(distribution_0)
    flat_gradient = jnp.concatenate([gradient["active"], gradient["rest"]])
    flat_weights = jnp.concatenate([weights["active"], weights["rest"]])
    batch = {
        "active": jnp.asarray([[1.0, 0.0], [0.0, 1.0]]),
        "rest": jnp.asarray([[0.0], [0.0]]),
    }
    batched = jax.jit(jax.vmap(endpoint))(batch)
    flat_batched = jnp.concatenate([batched["active"], batched["rest"]], axis=1)

    assert jnp.allclose(flat_value, flat_initial @ transition, atol=1e-10)
    assert jnp.allclose(flat_output_tangent, flat_tangent @ transition, atol=1e-10)
    assert jnp.allclose(flat_gradient, transition @ flat_weights, atol=1e-10)
    assert jnp.allclose(flat_batched, jnp.eye(3)[:2] @ transition, atol=1e-10)
