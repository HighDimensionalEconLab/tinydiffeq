import jax
import jax.numpy as jnp
import pytest

from benchmarks.test_pytree_overhead import (
    benchmark_batch_size,
    benchmark_direction_count,
)
from tinydiffeq import (
    RK4,
    ConstantStepSize,
    IController,
    LMRootSolver,
    SaveAt,
    Tsit5,
    solve_semi_explicit_dae,
)

STATE_DIM = 16
N_STEPS = 128
N_QUERIES = 128


def make_solve(n_steps=N_STEPS, n_queries=N_QUERIES, *, with_aux=True):
    grid = jnp.linspace(0.0, 1.0, n_queries)
    y0 = jnp.linspace(0.4, 0.9, STATE_DIM)
    z0 = y0**2

    def run(rate_vec):
        if with_aux:

            def differential(y, z, t, args, p, algebraic_aux):
                return p * z + 0.05 * (jnp.roll(y, 1) - y), algebraic_aux

            def algebraic(y, z, t, args, p):
                return (
                    z - y**2 - 0.1 * t,
                    {"flow": p * z, "level": y + z},
                )

            return solve_semi_explicit_dae(
                differential,
                algebraic,
                RK4(),
                0.0,
                1.0,
                y0,
                z0,
                p=rate_vec,
                dt_0=1.0 / n_steps,
                controller=ConstantStepSize(),
                root_solver=LMRootSolver(),
                max_steps=n_steps,
                save_at=SaveAt(ts=grid),
                has_aux=True,
                has_algebraic_aux=True,
            )

        return solve_semi_explicit_dae(
            lambda y, z, t, args, p: p * z + 0.05 * (jnp.roll(y, 1) - y),
            lambda y, z, t, args, p: z - y**2 - 0.1 * t,
            RK4(),
            0.0,
            1.0,
            y0,
            z0,
            p=rate_vec,
            dt_0=1.0 / n_steps,
            controller=ConstantStepSize(),
            root_solver=LMRootSolver(),
            max_steps=n_steps,
            save_at=SaveAt(ts=grid),
            has_aux=False,
            has_algebraic_aux=False,
        )

    return run


def make_adaptive_solve(n_queries=N_QUERIES):
    grid = jnp.linspace(0.0, 1.0, n_queries)
    y0 = jnp.linspace(0.4, 0.9, STATE_DIM)
    z0 = y0**2

    def run(rate_vec):
        def differential(y, z, t, args, p, algebraic_aux):
            return p * z + 0.05 * (jnp.roll(y, 1) - y), algebraic_aux

        return solve_semi_explicit_dae(
            differential,
            lambda y, z, t, args, p: (
                z - y**2 - 0.1 * t,
                {"flow": p * z, "level": y + z},
            ),
            Tsit5(),
            0.0,
            1.0,
            y0,
            z0,
            p=rate_vec,
            dt_0=0.05,
            controller=IController(),
            root_solver=LMRootSolver(),
            max_steps=N_STEPS,
            save_at=SaveAt(ts=grid),
            has_aux=True,
            has_algebraic_aux=True,
        )

    return run


@pytest.mark.parametrize("with_aux", [True, False], ids=["with_aux", "no_aux"])
@pytest.mark.parametrize(
    "transform",
    ["primal", "vmap_primal", "jvp", "vmap_jvp", "vjp", "vmap_vjp"],
)
def test_dense_output_large(benchmark, with_aux, transform):
    solve = make_solve(with_aux=with_aux)
    batch = benchmark_batch_size()
    k_dir = min(STATE_DIM, benchmark_direction_count())
    rate = jnp.linspace(0.1, 0.4, STATE_DIM)
    rate_batch = rate[None, :] + jnp.linspace(-0.05, 0.05, batch)[:, None]
    assert bool(solve(rate).ok)

    def vector_output(value):
        sol = solve(value)
        if with_aux:
            return jnp.mean(
                sol.ys + sol.zs + sol.aux["flow"] + sol.aux["level"], axis=0
            )[:k_dir]
        return jnp.mean(sol.ys + sol.zs, axis=0)[:k_dir]

    if transform == "primal":
        run, arg = jax.jit(solve), rate
    elif transform == "vmap_primal":
        run, arg = jax.jit(jax.vmap(solve)), rate_batch
    elif transform == "jvp":
        run, arg = (
            jax.jit(
                lambda value: jax.jvp(vector_output, (value,), (jnp.ones_like(value),))
            ),
            rate,
        )
    elif transform == "vmap_jvp":
        tangents = jnp.eye(STATE_DIM, dtype=rate.dtype)[:k_dir]

        def run_vmap_jvp(value):
            _, pushforward = jax.linearize(vector_output, value)
            return jax.vmap(pushforward)(tangents)

        run, arg = jax.jit(run_vmap_jvp), rate
    elif transform == "vjp":
        run, arg = (
            jax.jit(jax.grad(lambda value: jnp.sum(vector_output(value)))),
            rate,
        )
    else:
        run, arg = jax.jit(jax.jacrev(vector_output)), rate

    compiled = run.lower(arg).compile()
    jax.block_until_ready(compiled(arg))
    benchmark.pedantic(
        lambda: jax.block_until_ready(compiled(arg)), rounds=10, iterations=4
    )


@pytest.mark.parametrize(
    "transform",
    ["vmap_primal", "vmap_jvp", "vmap_vjp"],
)
def test_batched_adaptive_dense_output(benchmark, transform):
    solve = make_adaptive_solve()
    batch = benchmark_batch_size()
    k_dir = min(STATE_DIM, benchmark_direction_count())
    rate = jnp.linspace(0.1, 0.4, STATE_DIM)
    rates = rate[None, :] + jnp.linspace(-0.05, 0.05, batch)[:, None]
    assert bool(jnp.all(jax.vmap(solve)(rates).ok))

    def obs(value):
        return jnp.mean(solve(value).aux["flow"], axis=0)[:k_dir]

    if transform == "vmap_primal":
        run, arg = jax.jit(jax.vmap(obs)), rates
    elif transform == "vmap_jvp":
        tangents = jnp.eye(STATE_DIM, dtype=rate.dtype)[:k_dir]

        def run_vmap_jvp(value):
            _, pushforward = jax.linearize(obs, value)
            return jax.vmap(pushforward)(tangents)

        run, arg = jax.jit(run_vmap_jvp), rate
    else:
        run, arg = jax.jit(jax.jacrev(obs)), rate

    compiled = run.lower(arg).compile()
    jax.block_until_ready(compiled(arg))
    benchmark.pedantic(
        lambda: jax.block_until_ready(compiled(arg)), rounds=8, iterations=3
    )
