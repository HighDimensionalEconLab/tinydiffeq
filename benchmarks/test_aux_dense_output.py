"""Opt-in DAE aux/dense-output and batched interpolation benchmarks."""

import jax
import jax.numpy as jnp
import pytest

from tinydiffeq import (
    RK4,
    ConstantStepSize,
    IController,
    LMRootSolver,
    SaveAt,
    Tsit5,
    solve_semi_explicit_dae,
)


def make_solve(n_steps, n_queries):
    grid = jnp.linspace(0.0, 1.0, n_queries)

    def run(rate):
        def differential(y, z, t, args, p, algebraic_aux):
            return p * z, algebraic_aux

        return solve_semi_explicit_dae(
            differential,
            lambda y, z, t, args, p: (
                z - y**2 - 0.1 * t,
                {"flow": p * z, "level": y + z},
            ),
            RK4(),
            0.0,
            1.0,
            jnp.asarray(0.8),
            jnp.asarray(0.6),
            p=rate,
            dt_0=1.0 / n_steps,
            controller=ConstantStepSize(),
            root_solver=LMRootSolver(),
            max_steps=n_steps,
            save_at=SaveAt(ts=grid),
            has_aux=True,
            has_algebraic_aux=True,
        )

    return run


def make_adaptive_solve(n_queries):
    grid = jnp.linspace(0.0, 1.0, n_queries)

    def run(rate):
        def differential(y, z, t, args, p, algebraic_aux):
            return p * z, algebraic_aux

        return solve_semi_explicit_dae(
            differential,
            lambda y, z, t, args, p: (
                z - y**2 - 0.1 * t,
                {"flow": p * z, "level": y + z},
            ),
            Tsit5(),
            0.0,
            1.0,
            jnp.asarray(0.8),
            jnp.asarray(0.6),
            p=rate,
            dt_0=0.1,
            controller=IController(),
            root_solver=LMRootSolver(),
            max_steps=128,
            save_at=SaveAt(ts=grid),
            has_aux=True,
            has_algebraic_aux=True,
        )

    return run


def make_no_aux_solve(n_steps, n_queries, *, explicit_no_aux):
    grid = jnp.linspace(0.0, 1.0, n_queries)
    aux_options = (
        {"has_aux": False, "has_algebraic_aux": False} if explicit_no_aux else {}
    )

    def run(rate):
        return solve_semi_explicit_dae(
            lambda y, z, t, args, p: p * z,
            lambda y, z, t, args, p: z - y**2 - 0.1 * t,
            RK4(),
            0.0,
            1.0,
            jnp.asarray(0.8),
            jnp.asarray(0.6),
            p=rate,
            dt_0=1.0 / n_steps,
            controller=ConstantStepSize(),
            root_solver=LMRootSolver(),
            max_steps=n_steps,
            save_at=SaveAt(ts=grid),
            **aux_options,
        )

    return run


@pytest.mark.parametrize("n_steps", [8, 32, 128])
@pytest.mark.parametrize("n_queries", [8, 32, 128])
@pytest.mark.parametrize("transform", ["primal", "jvp", "vjp"])
def test_aux_dense_output_crossover(benchmark, n_steps, n_queries, transform):
    solve = make_solve(n_steps, n_queries)
    rate = jnp.asarray(0.2)
    assert bool(solve(rate).ok)

    def scalar_output(value):
        sol = solve(value)
        return jnp.sum(sol.ys + sol.zs + sol.aux["flow"] + sol.aux["level"])

    if transform == "primal":
        run = jax.jit(solve)
    elif transform == "jvp":
        run = jax.jit(
            lambda value: jax.jvp(scalar_output, (value,), (jnp.ones_like(value),))
        )
    else:
        run = jax.jit(jax.grad(scalar_output))
    compiled = run.lower(rate).compile()
    jax.block_until_ready(compiled(rate))
    benchmark.pedantic(
        lambda: jax.block_until_ready(compiled(rate)), rounds=10, iterations=5
    )


@pytest.mark.parametrize("n_steps", [8, 128])
@pytest.mark.parametrize("n_queries", [8, 128])
@pytest.mark.parametrize("transform", ["primal", "jvp", "vjp"])
@pytest.mark.parametrize("explicit_no_aux", [False, True], ids=["auto", "explicit"])
def test_no_aux_grid_output_crossover(
    benchmark, n_steps, n_queries, transform, explicit_no_aux
):
    """Cross-version baseline for the intentionally changed DAE grid path."""
    solve = make_no_aux_solve(n_steps, n_queries, explicit_no_aux=explicit_no_aux)
    rate = jnp.asarray(0.2)
    assert bool(solve(rate).ok)

    def scalar_output(value):
        sol = solve(value)
        return jnp.sum(sol.ys + sol.zs)

    if transform == "primal":
        run = jax.jit(solve)
    elif transform == "jvp":
        run = jax.jit(
            lambda value: jax.jvp(scalar_output, (value,), (jnp.ones_like(value),))
        )
    else:
        run = jax.jit(jax.grad(scalar_output))
    compiled = run.lower(rate).compile()
    jax.block_until_ready(compiled(rate))
    benchmark.pedantic(
        lambda: jax.block_until_ready(compiled(rate)), rounds=10, iterations=5
    )


def test_batched_adaptive_dense_output(benchmark):
    solve = make_adaptive_solve(32)
    rates = jnp.linspace(0.1, 1.0, 16)
    assert bool(jnp.all(jax.vmap(solve)(rates).ok))
    run = jax.jit(jax.vmap(lambda value: solve(value).aux["flow"]))
    compiled = run.lower(rates).compile()
    jax.block_until_ready(compiled(rates))
    benchmark.pedantic(
        lambda: jax.block_until_ready(compiled(rates)), rounds=10, iterations=5
    )
