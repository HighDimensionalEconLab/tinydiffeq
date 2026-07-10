"""Cold JAX compilation timings for the pytree benchmark matrix."""

import argparse
import statistics
import timeit
from functools import partial

import jax
import jax.numpy as jnp

from benchmarks.test_pytree_overhead import (
    make_solve,
    make_state,
    tree_map,
    tree_sum,
)


def transformed_solve(method, state_kind, transform):
    initial = make_state(state_kind)
    solve = make_solve(method, initial)
    tangent = tree_map(jnp.ones_like, initial)
    if transform == "primal":
        run = jax.jit(solve)
    elif transform == "jvp":
        run = jax.jit(lambda x: jax.jvp(solve, (x,), (tangent,)))
    else:
        run = jax.jit(lambda x: jax.grad(lambda value: tree_sum(solve(value)))(x))
    return run, initial


def compile_once(run, initial):
    jax.clear_caches()
    run.lower(initial).compile()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--methods", nargs="+", default=["rk4", "tsit5", "em", "dae", "sdae"]
    )
    parser.add_argument("--states", nargs="+", default=["scalar", "vector16", "tree16"])
    parser.add_argument("--transforms", nargs="+", default=["primal", "jvp", "vjp"])
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()

    print("method,state,transform,median_compile_seconds")
    for method in args.methods:
        for state_kind in args.states:
            for transform in args.transforms:
                run, initial = transformed_solve(method, state_kind, transform)
                samples = timeit.repeat(
                    partial(compile_once, run, initial),
                    repeat=args.repeat,
                    number=1,
                )
                print(
                    f"{method},{state_kind},{transform},"
                    f"{statistics.median(samples):.6f}"
                )


if __name__ == "__main__":
    main()
