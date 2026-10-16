import os

import jax
import jax.numpy as jnp
import pytest

from tinydiffeq import (
    RK4,
    DenseExponential,
    EulerMaruyama,
    IController,
    Rodas5P,
    Tsit5,
    solve_linear_ode,
    solve_ode,
    solve_sde,
    solve_semi_explicit_dae,
    solve_semi_explicit_sdae,
)


def _is_datacenter_gpu():
    dev = jax.devices()[0]
    if dev.platform != "gpu":
        return False
    stats = dev.memory_stats() or {}
    return stats.get("bytes_limit", 0) >= 40 * 1024**3


def benchmark_batch_size():
    override = os.environ.get("TINYDIFFEQ_BENCH_BATCH")
    if override is not None:
        return int(override)
    return 64 if _is_datacenter_gpu() else 16


def benchmark_direction_count():
    override = os.environ.get("TINYDIFFEQ_BENCH_DIRECTIONS")
    if override is not None:
        return int(override)
    return 32 if _is_datacenter_gpu() else 16


def make_state(kind):
    if kind == "vector64":
        return jnp.linspace(0.5, 1.5, 64)
    if kind == "tree64":
        return {
            "left": jnp.linspace(0.5, 1.0, 32),
            "right": jnp.linspace(1.0, 1.5, 32),
        }
    if kind == "vector16":
        return jnp.linspace(0.5, 1.5, 16)
    if kind == "tree16":
        return {"left": jnp.linspace(0.5, 1.0, 8), "right": jnp.linspace(1.0, 1.5, 8)}
    return jnp.asarray(1.0)


def tree_map(fn, tree):
    return jax.tree.map(fn, tree)


def tree_sum(tree):
    return sum(jnp.sum(leaf) for leaf in jax.tree.leaves(tree))


def _coupled_drift(state):
    return tree_map(
        lambda leaf: (
            -0.2 * leaf
            + 0.05 * (jnp.roll(leaf, 1) - 2.0 * leaf + jnp.roll(leaf, -1))
            - 0.02 * leaf**2
        ),
        state,
    )


def make_solve(method, initial):
    if method == "rk4":
        return lambda x: (
            solve_ode(
                _coupled_drift,
                RK4(),
                0.0,
                1.0,
                x,
                dt_0=1 / 64,
                max_steps=64,
                has_aux=False,
            ).xs
        )
    if method == "tsit5":
        return lambda x: (
            solve_ode(
                _coupled_drift,
                Tsit5(),
                0.0,
                1.0,
                x,
                dt_0=0.05,
                controller=IController(),
                max_steps=64,
                has_aux=False,
            ).xs
        )
    if method == "rodas5p":
        return lambda x: (
            solve_ode(
                _coupled_drift,
                Rodas5P(),
                0.0,
                1.0,
                x,
                dt_0=0.1,
                controller=IController(),
                max_steps=32,
                has_aux=False,
            ).xs
        )
    if method == "expm":
        flat0 = jnp.concatenate([jnp.ravel(leaf) for leaf in jax.tree.leaves(initial)])
        d = flat0.shape[0]
        base_A = (
            -0.5 * jnp.eye(d, dtype=flat0.dtype)
            + 0.1 * jnp.eye(d, k=1, dtype=flat0.dtype)
            + 0.1 * jnp.eye(d, k=-1, dtype=flat0.dtype)
        )

        def solve_expm(x):
            return solve_linear_ode(
                base_A,
                DenseExponential(),
                0.0,
                1.0,
                x,
            ).xs

        return solve_expm
    if method == "em":
        return lambda x: (
            solve_sde(
                _coupled_drift,
                lambda state: tree_map(lambda leaf: 0.1 * jnp.ones_like(leaf), state),
                EulerMaruyama(),
                0.0,
                1.0,
                x,
                key=jax.random.key(7),
                n_steps=64,
                has_aux=False,
            ).xs
        )

    z_0 = tree_map(lambda leaf: 0.9 * leaf, initial)

    def constraint(y, z):
        y_flat = jnp.concatenate([jnp.ravel(leaf) for leaf in jax.tree.leaves(y)])
        z_flat = jnp.concatenate([jnp.ravel(leaf) for leaf in jax.tree.leaves(z)])
        return z_flat - y_flat - 0.05 * y_flat**2

    if method == "sdae":
        return lambda y: (
            solve_semi_explicit_sdae(
                lambda state, z: _coupled_drift(z),
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
                has_aux=False,
                has_algebraic_aux=False,
            ).ys
        )

    return lambda y: (
        solve_semi_explicit_dae(
            lambda state, z: _coupled_drift(z),
            constraint,
            Rodas5P() if method == "rodas5p_dae" else Tsit5(),
            0.0,
            1.0,
            y,
            z_0,
            dt_0=0.1,
            controller=IController(),
            max_steps=32,
            has_aux=False,
            has_algebraic_aux=False,
        ).ys
    )


def build_transformed_runner(method, state_kind, transform):
    initial = make_state(state_kind)
    solve = make_solve(method, initial)
    batch = benchmark_batch_size()
    k_dir = benchmark_direction_count()
    tangent = tree_map(jnp.ones_like, initial)

    if transform == "primal":
        return jax.jit(solve), initial
    if transform == "vmap_primal":
        scales = jnp.linspace(0.9, 1.1, batch, dtype=jax.tree.leaves(initial)[0].dtype)
        batch_init = jax.vmap(lambda s: tree_map(lambda leaf: s * leaf, initial))(
            scales
        )
        return jax.jit(jax.vmap(solve)), batch_init
    if transform == "jvp":
        return jax.jit(lambda x: jax.jvp(solve, (x,), (tangent,))), initial
    if transform == "vmap_jvp":
        scales = jnp.linspace(0.5, 1.5, k_dir, dtype=jax.tree.leaves(initial)[0].dtype)
        tangents = jax.vmap(
            lambda s: tree_map(lambda leaf: s * jnp.ones_like(leaf), initial)
        )(scales)

        def run_vmap_jvp(x):
            _, pushforward = jax.linearize(solve, x)
            return jax.vmap(pushforward)(tangents)

        return jax.jit(run_vmap_jvp), initial
    if transform == "vjp":
        return (
            jax.jit(lambda x: jax.grad(lambda value: tree_sum(solve(value)))(x)),
            initial,
        )
    out_sample = jax.eval_shape(solve, initial)
    scales = jnp.linspace(0.5, 1.5, k_dir, dtype=jax.tree.leaves(initial)[0].dtype)
    cotangents = jax.vmap(
        lambda s: tree_map(
            lambda leaf: s * jnp.ones(leaf.shape, dtype=leaf.dtype), out_sample
        )
    )(scales)

    def run_vmap_vjp(x):
        _, pullback = jax.vjp(solve, x)
        return jax.vmap(lambda ct: pullback(ct)[0])(cotangents)

    return jax.jit(run_vmap_vjp), initial


@pytest.mark.parametrize(
    "method",
    ["rk4", "tsit5", "rodas5p", "expm", "em", "dae", "rodas5p_dae", "sdae"],
)
@pytest.mark.parametrize("state_kind", ["vector64", "tree64"])
@pytest.mark.parametrize(
    "transform",
    ["primal", "vmap_primal", "jvp", "vmap_jvp", "vjp", "vmap_vjp"],
)
def test_steady_state(benchmark, method, state_kind, transform):
    run, arg = build_transformed_runner(method, state_kind, transform)
    compiled = run.lower(arg).compile()
    jax.block_until_ready(compiled(arg))
    benchmark.pedantic(
        lambda: jax.block_until_ready(compiled(arg)), rounds=12, iterations=5
    )
