import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tinydiffeq import (
    AssociativeMarkov,
    ContinuousTimeMarkovChain,
    DiscreteMarkovChain,
    SaveAt,
    SequentialMarkov,
    forecast_continuous_time_markov_chain,
    forecast_markov_chain,
    simulate_continuous_time_markov_chain,
    simulate_markov_chain,
)


def test_discrete_deterministic_path_and_save_modes():
    chain = DiscreteMarkovChain([[0.0, 1.0], [1.0, 0.0]])
    key = jax.random.key(0)
    steps = simulate_markov_chain(
        chain,
        jnp.int32(0),
        key=key,
        num_steps=6,
        save_at=SaveAt(steps=True),
    )
    endpoint = simulate_markov_chain(chain, jnp.int32(0), key=key, num_steps=6)
    selected = simulate_markov_chain(
        chain,
        jnp.int32(0),
        key=key,
        num_steps=6,
        save_at=SaveAt(ts=jnp.asarray([0, 3, 6], jnp.int32)),
    )
    assert jnp.array_equal(steps.xs, jnp.asarray([0, 1, 0, 1, 0, 1, 0]))
    assert endpoint.xs == 0
    assert jnp.array_equal(selected.xs, jnp.asarray([0, 1, 0]))
    assert bool(steps.ok) and bool(jnp.all(steps.accepted))
    assert int(steps.num_steps) == int(endpoint.num_steps) == 6

    passed_as_pytree = jax.jit(
        lambda prepared, random_key: (
            simulate_markov_chain(
                prepared, jnp.int32(0), key=random_key, num_steps=6
            ).xs
        )
    )(chain, key)
    assert passed_as_pytree == endpoint.xs


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_discrete_sequential_associative_and_vmap_match(dtype):
    chain = DiscreteMarkovChain(
        jnp.asarray([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3], [0.2, 0.2, 0.6]], dtype)
    )
    keys = jax.random.split(jax.random.key(1), 16)

    def run(key, method):
        return simulate_markov_chain(
            chain,
            jnp.int32(1),
            key=key,
            num_steps=127,
            method=method,
            save_at=SaveAt(steps=True),
        ).xs

    sequential = jax.jit(jax.vmap(lambda key: run(key, SequentialMarkov())))(keys)
    unrolled = jax.jit(jax.vmap(lambda key: run(key, SequentialMarkov(unroll=8))))(keys)
    associative = jax.jit(jax.vmap(lambda key: run(key, AssociativeMarkov())))(keys)
    assert jnp.array_equal(sequential, unrolled)
    assert jnp.array_equal(sequential, associative)


@pytest.mark.parametrize("method", [SequentialMarkov(), AssociativeMarkov()])
def test_float32_simulation_jaxprs_are_pure_under_x64(method):
    discrete = DiscreteMarkovChain(jnp.asarray([[0.8, 0.2], [0.3, 0.7]], jnp.float32))
    continuous = ContinuousTimeMarkovChain(
        jnp.asarray([[-1.0, 1.0], [0.5, -0.5]], jnp.float32)
    )
    keys = jax.random.split(jax.random.key(91), 4)

    def discrete_path(key):
        return simulate_markov_chain(
            discrete,
            jnp.int32(0),
            key=key,
            num_steps=16,
            method=method,
            save_at=SaveAt(steps=True),
        )

    def continuous_path(key):
        return simulate_continuous_time_markov_chain(
            continuous,
            jnp.asarray(0.0, jnp.float32),
            jnp.asarray(2.0, jnp.float32),
            jnp.int32(0),
            key=key,
            max_jumps=32,
            method=method,
            save_at=SaveAt(steps=True),
        )

    discrete_jaxpr = jax.make_jaxpr(jax.vmap(discrete_path))(keys)
    continuous_jaxpr = jax.make_jaxpr(jax.vmap(continuous_path))(keys)
    assert "f64" not in str(discrete_jaxpr), discrete_jaxpr
    assert "f64" not in str(continuous_jaxpr), continuous_jaxpr

    discrete_paths = jax.jit(jax.vmap(discrete_path))(keys)
    continuous_paths = jax.jit(jax.vmap(continuous_path))(keys)
    assert discrete_paths.xs.shape == (4, 17)
    assert continuous_paths.ts.dtype == jnp.float32
    assert bool(jnp.all(continuous_paths.ok))


def test_discrete_one_step_distribution():
    transition = jnp.asarray(
        [[0.1, 0.3, 0.6], [0.2, 0.5, 0.3], [0.7, 0.2, 0.1]],
        jnp.float32,
    )
    chain = DiscreteMarkovChain(transition)
    keys = jax.random.split(jax.random.key(2), 30_000)
    samples = jax.jit(
        jax.vmap(
            lambda key: (
                simulate_markov_chain(chain, jnp.int32(0), key=key, num_steps=1).xs
            )
        )
    )(keys)
    frequencies = jnp.bincount(samples, length=3) / samples.size
    assert jnp.max(jnp.abs(frequencies - transition[0])) < 0.012


@pytest.mark.parametrize(
    "matrix,match",
    [
        ([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], "square"),
        ([[1.0, -0.1], [0.0, 1.0]], "nonnegative"),
        ([[0.0, 0.0], [0.5, 0.5]], "positive mass"),
        ([[float("nan"), 0.0], [0.5, 0.5]], "finite"),
    ],
)
def test_discrete_validation(matrix, match):
    with pytest.raises(ValueError, match=match):
        DiscreteMarkovChain(matrix)


def test_invalid_initial_state_and_queries_report_not_ok():
    chain = DiscreteMarkovChain([[0.5, 0.5], [0.5, 0.5]])
    invalid_state = simulate_markov_chain(
        chain, jnp.int32(3), key=jax.random.key(0), num_steps=4
    )
    invalid_queries = simulate_markov_chain(
        chain,
        jnp.int32(0),
        key=jax.random.key(0),
        num_steps=4,
        save_at=SaveAt(ts=jnp.asarray([-1, 5], jnp.int32)),
    )
    assert not bool(invalid_state.ok)
    assert not bool(invalid_queries.ok)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_ctmc_methods_match_states_and_event_times(dtype):
    chain = ContinuousTimeMarkovChain(
        jnp.asarray([[-1.0, 0.7, 0.3], [0.2, -0.6, 0.4], [0.5, 0.5, -1.0]], dtype)
    )
    kwargs = dict(
        t_0=jnp.asarray(0.0, dtype),
        t_1=jnp.asarray(30.0, dtype),
        state_0=jnp.int32(0),
        key=jax.random.key(3),
        max_jumps=256,
        save_at=SaveAt(steps=True),
    )
    sequential = simulate_continuous_time_markov_chain(
        chain, method=SequentialMarkov(), **kwargs
    )
    associative = simulate_continuous_time_markov_chain(
        chain, method=AssociativeMarkov(), **kwargs
    )
    assert jnp.array_equal(sequential.xs, associative.xs)
    tolerance = 2e-4 if dtype == jnp.float32 else 1e-11
    assert jnp.allclose(sequential.ts, associative.ts, rtol=tolerance, atol=tolerance)
    assert sequential.num_accepted == associative.num_accepted
    assert sequential.num_steps == associative.num_steps
    assert bool(sequential.ok) and bool(associative.ok)


def test_ctmc_requested_times_match_event_path():
    chain = ContinuousTimeMarkovChain([[-2.0, 2.0], [1.0, -1.0]])
    key = jax.random.key(4)
    event_solution = simulate_continuous_time_markov_chain(
        chain,
        0.0,
        5.0,
        jnp.int32(0),
        key=key,
        max_jumps=64,
        save_at=SaveAt(steps=True),
    )
    query_times = jnp.linspace(0.0, 5.0, 31)
    grid_solution = simulate_continuous_time_markov_chain(
        chain,
        0.0,
        5.0,
        jnp.int32(0),
        key=key,
        max_jumps=64,
        save_at=SaveAt(ts=query_times),
    )
    valid_times = event_solution.ts[event_solution.accepted]
    valid_states = event_solution.xs[event_solution.accepted]
    counts = jnp.searchsorted(valid_times[1:], query_times, side="right")
    assert jnp.array_equal(grid_solution.xs, valid_states[counts])
    assert bool(grid_solution.ok)


def test_ctmc_absorbing_and_starved_contracts():
    absorbing = ContinuousTimeMarkovChain([[0.0, 0.0], [1.0, -1.0]])
    absorbed = simulate_continuous_time_markov_chain(
        absorbing,
        0.0,
        100.0,
        jnp.int32(0),
        key=jax.random.key(5),
        max_jumps=4,
        save_at=SaveAt(steps=True),
    )
    active = ContinuousTimeMarkovChain([[-1.0, 1.0], [1.0, -1.0]])
    starved = simulate_continuous_time_markov_chain(
        active,
        0.0,
        1_000.0,
        jnp.int32(0),
        key=jax.random.key(6),
        max_jumps=1,
    )
    assert bool(absorbed.ok)
    assert int(absorbed.num_accepted) == 0
    assert int(absorbed.num_steps) == 1
    assert jnp.all(absorbed.xs == 0)
    assert not bool(starved.ok)
    assert int(starved.num_steps) == 1
    assert starved.ts < 1_000.0


@pytest.mark.parametrize("method", [SequentialMarkov(), AssociativeMarkov()])
def test_two_state_ctmc_endpoint_distribution(method):
    rate_01 = jnp.asarray(2.0, jnp.float32)
    rate_10 = jnp.asarray(1.0, jnp.float32)
    horizon = jnp.asarray(0.8, jnp.float32)
    chain = ContinuousTimeMarkovChain(
        jnp.asarray([[-rate_01, rate_01], [rate_10, -rate_10]], jnp.float32)
    )
    keys = jax.random.split(jax.random.key(7), 20_000)
    solutions = jax.jit(
        jax.vmap(
            lambda key: simulate_continuous_time_markov_chain(
                chain,
                0.0,
                horizon,
                jnp.int32(0),
                key=key,
                max_jumps=32,
                method=method,
            )
        )
    )(keys)
    expected_state_1 = (
        rate_01 / (rate_01 + rate_10) * (1.0 - np.exp(-(rate_01 + rate_10) * horizon))
    )
    assert bool(jnp.all(solutions.ok))
    assert abs(float(jnp.mean(solutions.xs == 1)) - expected_state_1) < 0.012


@pytest.mark.parametrize(
    "generator,match",
    [
        ([[-1.0, 1.0, 0.0], [1.0, -1.0, 0.0]], "square"),
        ([[-1.0, -1.0], [1.0, -1.0]], "nonnegative"),
        ([[-1.0, 0.5], [1.0, -1.0]], "sum to zero"),
        ([[float("inf"), 0.0], [1.0, -1.0]], "finite"),
    ],
)
def test_ctmc_validation(generator, match):
    with pytest.raises(ValueError, match=match):
        ContinuousTimeMarkovChain(generator)


def test_markov_static_argument_validation():
    chain = DiscreteMarkovChain([[1.0]])
    with pytest.raises(TypeError, match="static"):
        simulate_markov_chain(
            chain, jnp.int32(0), key=jax.random.key(0), num_steps=jnp.asarray(2)
        )
    with pytest.raises(TypeError, match="scalar integer"):
        simulate_markov_chain(
            chain, jnp.asarray([0]), key=jax.random.key(0), num_steps=2
        )
    with pytest.raises(ValueError, match="positive"):
        SequentialMarkov(unroll=0)
    with pytest.raises(ValueError, match="integer states"):
        simulate_markov_chain(
            chain,
            jnp.int32(0),
            key=jax.random.key(0),
            num_steps=2,
            save_at=SaveAt(steps=True, fill="inf"),
        )


def test_markov_save_at_exact_is_rejected_by_every_public_query_api():
    discrete = DiscreteMarkovChain([[0.75, 0.25], [0.4, 0.6]])
    continuous = ContinuousTimeMarkovChain([[-0.5, 0.5], [0.25, -0.25]])
    discrete_queries = SaveAt(ts=jnp.asarray([0, 2], jnp.int32), exact=True)
    continuous_queries = SaveAt(ts=jnp.asarray([0.0, 1.0]), exact=True)
    match = "only supported by solve_ode"

    with pytest.raises(ValueError, match=match):
        simulate_markov_chain(
            discrete,
            jnp.int32(0),
            key=jax.random.key(0),
            num_steps=2,
            save_at=discrete_queries,
        )
    with pytest.raises(ValueError, match=match):
        simulate_continuous_time_markov_chain(
            continuous,
            0.0,
            1.0,
            jnp.int32(0),
            key=jax.random.key(1),
            max_jumps=8,
            save_at=continuous_queries,
        )
    with pytest.raises(ValueError, match=match):
        forecast_markov_chain(
            discrete,
            jnp.asarray([1.0, 0.0]),
            num_steps=2,
            save_at=discrete_queries,
        )
    with pytest.raises(ValueError, match=match):
        forecast_continuous_time_markov_chain(
            continuous,
            0.0,
            1.0,
            jnp.asarray([1.0, 0.0]),
            save_at=continuous_queries,
        )
