import argparse
import os
import statistics
import timeit

import jax
import jax.numpy as jnp
import numpy as np

from tinydiffeq import solve_bvp


def _is_datacenter_gpu():
    dev = jax.devices()[0]
    if dev.platform != "gpu":
        return False
    stats = dev.memory_stats() or {}
    return stats.get("bytes_limit", 0) >= 40 * 1024**3


def default_batch_size():
    override = os.environ.get("TINYDIFFEQ_BENCH_BATCH")
    if override is not None:
        return int(override)
    return 64 if _is_datacenter_gpu() else 16


def default_direction_count():
    override = os.environ.get("TINYDIFFEQ_BENCH_DIRECTIONS")
    if override is not None:
        return int(override)
    return 32 if _is_datacenter_gpu() else 8


def shock_problem(max_nodes, dtype):
    eps = 1e-3

    def fun(t, y):
        return jnp.array(
            [
                y[1],
                -(
                    t * y[1]
                    + eps * jnp.pi**2 * jnp.cos(jnp.pi * t)
                    + jnp.pi * t * jnp.sin(jnp.pi * t)
                )
                / eps,
            ],
            dtype=y.dtype,
        )

    def bc(ya, yb):
        return jnp.array([ya[0] + 2.0, yb[0]], dtype=ya.dtype)

    t = jnp.linspace(-1.0, 1.0, 9, dtype=dtype)
    y_0 = jnp.zeros((9, 2), dtype=dtype)
    tol = 1e-3
    jitted = jax.jit(
        lambda tt, yy: solve_bvp(fun, bc, tt, yy, tol=tol, max_nodes=max_nodes)
    )

    def solve():
        return jitted(t, y_0)

    def scipy_solve():
        from scipy.integrate import solve_bvp as scipy_solve_bvp

        def np_fun(x, y):
            return np.vstack(
                (
                    y[1],
                    -(
                        x * y[1]
                        + eps * np.pi**2 * np.cos(np.pi * x)
                        + np.pi * x * np.sin(np.pi * x)
                    )
                    / eps,
                )
            )

        return scipy_solve_bvp(
            np_fun,
            lambda ya, yb: np.array([ya[0] + 2.0, yb[0]]),
            np.linspace(-1, 1, 9),
            np.zeros((2, 9)),
        )

    return solve, scipy_solve


def coupled_bvp_suite(max_nodes, batch, n_directions, dtype):
    n_pairs = 4
    dim = 2 * n_pairs
    t = jnp.linspace(0.0, 1.0, 17, dtype=dtype)
    y_0 = jnp.zeros((17, dim), dtype=dtype)
    tol = 1e-4 if dtype == jnp.float32 else 1e-6
    rates = jnp.linspace(1.0, 2.5, n_pairs, dtype=dtype)
    k1, k2 = jax.random.split(jax.random.key(0))
    proj_in = jax.random.normal(k1, (n_pairs, n_directions), dtype=dtype) / jnp.sqrt(
        jnp.asarray(n_directions, dtype=dtype)
    )
    proj_out = jax.random.normal(k2, (n_directions, dim), dtype=dtype) / jnp.sqrt(
        jnp.asarray(dim, dtype=dtype)
    )
    p_0 = jnp.ones((n_directions,), dtype=dtype) * 0.2
    p_batch = p_0[None, :] + jnp.linspace(-0.05, 0.05, batch, dtype=dtype)[:, None]

    def fun(t_val, y, z, args, p):
        u = y[0::2]
        v = y[1::2]
        shift = proj_in @ p
        du = v
        dv = (
            rates * u
            + 0.15 * (jnp.roll(u, 1) - 2.0 * u + jnp.roll(u, -1))
            + 0.05 * u**3
            + shift * jnp.sin(jnp.pi * t_val)
        )
        out = jnp.empty_like(y)
        out = out.at[0::2].set(du)
        out = out.at[1::2].set(dv)
        return out

    def bc(ya, yb, z, args, p):
        shift = proj_in @ p
        return jnp.concatenate([ya[0::2] - (1.0 + 0.2 * shift), yb[0::2]])

    def solve_obs(p):
        sol = solve_bvp(fun, bc, t, y_0, p=p, tol=tol, max_nodes=max_nodes)
        midpoint_and_end = sol.y[0] + sol.y[8] + sol.y[16]
        return proj_out @ midpoint_and_end

    tangent = jnp.ones_like(p_0)
    primal_fn = jax.jit(solve_obs)
    vmap_primal_fn = jax.jit(jax.vmap(solve_obs))
    jvp_fn = jax.jit(lambda p: jax.jvp(solve_obs, (p,), (tangent,)))
    vmap_jvp_fn = jax.jit(jax.jacfwd(solve_obs))
    vjp_fn = jax.jit(jax.grad(lambda p: jnp.sum(solve_obs(p) ** 2)))
    vmap_vjp_fn = jax.jit(jax.jacrev(solve_obs))
    vmap_grad_fn = jax.jit(jax.vmap(jax.grad(lambda p: jnp.sum(solve_obs(p) ** 2))))

    return [
        ("coupled8_primal", lambda: primal_fn(p_0)),
        (f"coupled8_vmap_primal_b{batch}", lambda: vmap_primal_fn(p_batch)),
        ("coupled8_jvp", lambda: jvp_fn(p_0)),
        (f"coupled8_vmap_jvp_k{n_directions}", lambda: vmap_jvp_fn(p_0)),
        ("coupled8_vjp", lambda: vjp_fn(p_0)),
        (f"coupled8_vmap_vjp_k{n_directions}", lambda: vmap_vjp_fn(p_0)),
        (f"coupled8_vmap_grad_b{batch}", lambda: vmap_grad_fn(p_batch)),
    ]


def cold_and_warm_ms(run, repeat):
    def timed():
        jax.block_until_ready(jax.tree.leaves(run()))

    cold_ms = 1e3 * timeit.timeit(timed, number=1)
    warm_ms = 1e3 * statistics.median(timeit.repeat(timed, repeat=repeat, number=1))
    return cold_ms, warm_ms


def cache_audit(max_nodes):
    def fun(t, y, z, args, p):
        return jnp.array([y[1], p * y[0]])

    def bc(ya, yb, z, args, p):
        return jnp.array([ya[0] - 1.0, yb[0]])

    for index, (p, span) in enumerate([(1.0, 1.0), (2.5, 1.5)]):
        print(f"=== SOLVE {index} p={p} ===", flush=True)
        t = jnp.linspace(0.0, span, 5)
        sol = solve_bvp(fun, bc, t, jnp.zeros((5, 2)), p=p, max_nodes=max_nodes)
        jax.block_until_ready(sol.y)
        print(f"=== DONE {index} status={int(sol.status)} ===", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-nodes", nargs="+", type=int, default=[64, 128])
    parser.add_argument(
        "--dtypes",
        nargs="+",
        default=None,
        choices=["float32", "float64"],
    )
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--directions", type=int, default=None)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--cache-audit", action="store_true")
    args = parser.parse_args()
    if args.dtypes is None:
        args.dtypes = (
            ["float32"]
            if os.environ.get("JAX_PLATFORMS") == "metal"
            else ["float32", "float64"]
        )

    if args.cache_audit or "float64" in args.dtypes:
        jax.config.update("jax_enable_x64", True)

    if args.cache_audit:
        cache_audit(args.max_nodes[0])
        return

    batch = default_batch_size() if args.batch is None else args.batch
    directions = (
        default_direction_count() if args.directions is None else args.directions
    )
    dtypes = {"float32": jnp.float32, "float64": jnp.float64}
    print("dtype,case,max_nodes,cold_ms,run_ms,scipy_ms")
    for dtype_name in args.dtypes:
        dtype = dtypes[dtype_name]
        for max_nodes in args.max_nodes:
            solve, scipy_solve = shock_problem(max_nodes, dtype)
            cold_ms, run_ms = cold_and_warm_ms(solve, args.repeat)
            scipy_ms = 1e3 * statistics.median(
                timeit.repeat(scipy_solve, repeat=args.repeat, number=1)
            )
            print(
                f"{dtype_name},shock,{max_nodes},"
                f"{cold_ms:.1f},{run_ms:.3f},{scipy_ms:.3f}"
            )
            for case_name, fn in coupled_bvp_suite(max_nodes, batch, directions, dtype):
                cold_ms, run_ms = cold_and_warm_ms(fn, args.repeat)
                print(
                    f"{dtype_name},{case_name},{max_nodes},{cold_ms:.1f},{run_ms:.3f},"
                )


if __name__ == "__main__":
    main()
