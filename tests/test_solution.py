import jax
import jax.numpy as jnp

from tinydiffeq import (
    RK4,
    DenseExponential,
    DiscreteMarkovChain,
    EulerMaruyama,
    simulate_markov_chain,
    solve_linear_ode,
    solve_ode,
    solve_sde,
)


def test_endpoint_solution_step_counts_have_uniform_array_pytree_structure():
    ode = solve_ode(
        lambda x: -x,
        RK4(),
        0.0,
        1.0,
        jnp.asarray(1.0),
        dt_0=0.25,
        max_steps=4,
        has_aux=False,
    )
    sde = solve_sde(
        lambda x: -x,
        lambda x: jnp.zeros_like(x),
        EulerMaruyama(),
        0.0,
        1.0,
        jnp.asarray(1.0),
        key=jax.random.key(0),
        n_steps=4,
        has_aux=False,
    )
    exponential = solve_linear_ode(
        jnp.asarray([[-1.0]]),
        DenseExponential(),
        0.0,
        1.0,
        jnp.asarray([1.0]),
    )
    markov = simulate_markov_chain(
        DiscreteMarkovChain([[0.0, 1.0], [1.0, 0.0]]),
        jnp.asarray(0, jnp.int32),
        key=jax.random.key(1),
        num_steps=4,
    )

    solutions = (ode, sde, exponential, markov)
    assert tuple(int(solution.num_steps) for solution in solutions) == (4, 4, 1, 4)
    expected_structure = jax.tree.structure(ode)
    for solution in solutions:
        assert jax.tree.structure(solution) == expected_structure
        assert isinstance(solution.num_steps, jax.Array)
        assert solution.num_steps.shape == ()
        assert solution.num_steps.dtype == jnp.int32
