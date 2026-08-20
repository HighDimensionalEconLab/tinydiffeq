"""Compile and steady-state timings for solve_bvp, with scipy references.

Every case calls the solve through a jitted wrapper: cold is
compile-plus-first-run, run is the warm steady state.
"""

import argparse
import statistics
import timeit

import jax
import jax.numpy as jnp
import numpy as np

from tinydiffeq import solve_bvp

jax.config.update("jax_enable_x64", True)


def exp_problem(max_nodes):
    def fun(t, y):
        return jnp.array([y[1], y[0]])

    def bc(ya, yb):
        return jnp.array([ya[0] - 1.0, yb[0]])

    t = jnp.linspace(0.0, 1.0, 5)
    y_0 = jnp.zeros((5, 2))
    jitted = jax.jit(lambda tt, yy: solve_bvp(fun, bc, tt, yy, max_nodes=max_nodes))

    def solve():
        return jitted(t, y_0)

    def scipy_solve():
        from scipy.integrate import solve_bvp as scipy_solve_bvp

        return scipy_solve_bvp(
            lambda x, y: np.vstack((y[1], y[0])),
            lambda ya, yb: np.array([ya[0] - 1.0, yb[0]]),
            np.linspace(0, 1, 5),
            np.zeros((2, 5)),
        )

    return solve, scipy_solve


def shock_problem(max_nodes):
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
            ]
        )

    def bc(ya, yb):
        return jnp.array([ya[0] + 2.0, yb[0]])

    t = jnp.linspace(-1.0, 1.0, 5)
    y_0 = jnp.zeros((5, 2))
    jitted = jax.jit(lambda tt, yy: solve_bvp(fun, bc, tt, yy, max_nodes=max_nodes))

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
            np.linspace(-1, 1, 5),
            np.zeros((2, 5)),
        )

    return solve, scipy_solve


def eigen_gradient(max_nodes):
    def fun(t, y, z, args, p):
        return jnp.array([y[1], -(z[0] ** 2) * p * y[0]])

    def bc(ya, yb, z, args, p):
        return jnp.array([ya[0], yb[0], ya[1] - z[0]])

    t = jnp.linspace(0.0, 1.0, 9)
    y_0 = jnp.stack([jnp.sin(jnp.pi * t), jnp.pi * jnp.cos(jnp.pi * t)], axis=1)

    def objective(p):
        return solve_bvp(
            fun,
            bc,
            t,
            y_0,
            jnp.array([3.0]),
            p=p,
            tol=1e-6,
            max_nodes=max_nodes,
        ).z[0]

    gradient = jax.jit(jax.grad(objective))

    def solve():
        return gradient(1.0)

    return solve, None


def vmapped_exp(max_nodes, batch):
    def fun(t, y, z, args, p):
        return jnp.array([y[1], y[0]])

    def bc(ya, yb, z, args, p):
        return jnp.array([ya[0] - p, yb[0]])

    t = jnp.linspace(0.0, 1.0, 5)
    y_0 = jnp.zeros((5, 2))
    p_values = jnp.linspace(0.5, 2.0, batch)

    def one(p):
        return solve_bvp(fun, bc, t, y_0, p=p, max_nodes=max_nodes)

    batched = jax.jit(lambda ps: jax.vmap(one)(ps))

    def solve():
        return batched(p_values)

    return solve, None


def cold_and_warm_ms(run, repeat):
    def timed():
        jax.block_until_ready(jax.tree.leaves(run()))

    # Each factory builds fresh callables, so the first call in this process
    # traces and compiles: compile-plus-first-run, reported separately from
    # the warm steady state.
    cold_ms = 1e3 * timeit.timeit(timed, number=1)
    warm_ms = 1e3 * statistics.median(timeit.repeat(timed, repeat=repeat, number=1))
    return cold_ms, warm_ms


def cache_audit(max_nodes):
    """Two solves with changed data leaves; run under
    JAX_EXPLAIN_CACHE_MISSES=1 JAX_LOG_COMPILES=1 and count events after the
    second marker — the target is zero."""

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
    parser.add_argument("--max-nodes", nargs="+", type=int, default=[32, 128])
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--cache-audit", action="store_true")
    args = parser.parse_args()

    if args.cache_audit:
        cache_audit(args.max_nodes[0])
        return

    print("case,max_nodes,cold_ms,run_ms,scipy_ms")
    for max_nodes in args.max_nodes:
        for name, factory in (("exp", exp_problem), ("shock", shock_problem)):
            solve, scipy_solve = factory(max_nodes)
            cold_ms, run_ms = cold_and_warm_ms(solve, args.repeat)
            scipy_ms = 1e3 * statistics.median(
                timeit.repeat(scipy_solve, repeat=args.repeat, number=1)
            )
            print(f"{name},{max_nodes},{cold_ms:.1f},{run_ms:.3f},{scipy_ms:.3f}")
        solve, _ = eigen_gradient(max_nodes)
        cold_ms, run_ms = cold_and_warm_ms(solve, args.repeat)
        print(f"eigen_grad,{max_nodes},{cold_ms:.1f},{run_ms:.3f},")
        solve, _ = vmapped_exp(max_nodes, args.batch)
        cold_ms, run_ms = cold_and_warm_ms(solve, args.repeat)
        print(f"vmap_exp_b{args.batch},{max_nodes},{cold_ms:.1f},{run_ms:.3f},")


if __name__ == "__main__":
    main()
