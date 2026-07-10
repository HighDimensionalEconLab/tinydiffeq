"""Opt-in steady-state array/pytree and AD benchmarks.

Run with ``uv run --group benchmark pytest benchmarks --benchmark-only``.
"""

import jax
import jax.numpy as jnp
import pytest

from tinydiffeq import (
    RK4,
    EulerMaruyama,
    IController,
    Tsit5,
    solve_ode,
    solve_sde,
    solve_semi_explicit_dae,
    solve_semi_explicit_sdae,
)


def make_state(kind):
    if kind == "scalar":
        return jnp.asarray(1.0)
    if kind == "vector16":
        return jnp.linspace(0.5, 1.5, 16)
    return {"left": jnp.linspace(0.5, 1.0, 8), "right": jnp.linspace(1.0, 1.5, 8)}


def tree_map(fn, tree):
    return jax.tree.map(fn, tree)


def tree_sum(tree):
    return sum(jnp.sum(leaf) for leaf in jax.tree.leaves(tree))


def make_solve(method, initial):
    if method == "rk4":
        return lambda x: (
            solve_ode(
                lambda state: tree_map(lambda leaf: -0.2 * leaf, state),
                RK4(),
                0.0,
                1.0,
                x,
                dt_0=1 / 64,
                max_steps=64,
            ).xs
        )
    if method == "tsit5":
        return lambda x: (
            solve_ode(
                lambda state: tree_map(lambda leaf: -0.2 * leaf, state),
                Tsit5(),
                0.0,
                1.0,
                x,
                dt_0=0.05,
                controller=IController(),
                max_steps=64,
            ).xs
        )
    if method == "em":
        return lambda x: (
            solve_sde(
                lambda state: tree_map(lambda leaf: -0.2 * leaf, state),
                lambda state: tree_map(lambda leaf: 0.1 * jnp.ones_like(leaf), state),
                EulerMaruyama(),
                0.0,
                1.0,
                x,
                key=jax.random.key(7),
                n_steps=64,
            ).xs
        )

    z_0 = tree_map(lambda leaf: 0.9 * leaf, initial)

    def constraint(y, z):
        y_flat = jnp.concatenate([jnp.ravel(leaf) for leaf in jax.tree.leaves(y)])
        z_flat = jnp.concatenate([jnp.ravel(leaf) for leaf in jax.tree.leaves(z)])
        return z_flat - y_flat

    if method == "sdae":
        return lambda y: (
            solve_semi_explicit_sdae(
                lambda state, z: tree_map(lambda leaf: -0.2 * leaf, z),
                lambda state, z: tree_map(
                    lambda leaf: 0.1 * jnp.ones_like(leaf), state
                ),
                constraint,
                EulerMaruyama(),
                0.0,
                1.0,
                y,
                z_0,
                key=jax.random.key(7),
                n_steps=64,
            ).ys
        )

    return lambda y: (
        solve_semi_explicit_dae(
            lambda state, z: tree_map(lambda leaf: -0.2 * leaf, z),
            constraint,
            Tsit5(),
            0.0,
            1.0,
            y,
            z_0,
            dt_0=0.1,
            controller=IController(),
            max_steps=32,
        ).ys
    )


@pytest.mark.parametrize("method", ["rk4", "tsit5", "em", "dae", "sdae"])
@pytest.mark.parametrize("state_kind", ["scalar", "vector16", "tree16"])
@pytest.mark.parametrize("transform", ["primal", "jvp", "vjp"])
def test_steady_state(benchmark, method, state_kind, transform):
    initial = make_state(state_kind)
    solve = make_solve(method, initial)
    tangent = tree_map(jnp.ones_like, initial)

    if transform == "primal":
        run = jax.jit(solve)
    elif transform == "jvp":
        run = jax.jit(lambda x: jax.jvp(solve, (x,), (tangent,)))
    else:
        run = jax.jit(lambda x: jax.grad(lambda value: tree_sum(solve(value)))(x))

    compiled = run.lower(initial).compile()
    jax.block_until_ready(compiled(initial))
    benchmark.pedantic(
        lambda: jax.block_until_ready(compiled(initial)), rounds=20, iterations=10
    )
