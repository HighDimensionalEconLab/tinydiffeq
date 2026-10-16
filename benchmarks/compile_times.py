import argparse
import statistics
import timeit
from functools import partial

import jax

from benchmarks.test_pytree_overhead import build_transformed_runner


def compile_once(run, arg):
    jax.clear_caches()
    run.lower(arg).compile()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["rk4", "tsit5", "rodas5p", "expm", "em", "dae", "rodas5p_dae", "sdae"],
    )
    parser.add_argument("--states", nargs="+", default=["vector64", "tree64"])
    parser.add_argument(
        "--transforms",
        nargs="+",
        default=["primal", "vmap_primal", "jvp", "vmap_jvp", "vjp", "vmap_vjp"],
    )
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()

    print("method,state,transform,median_compile_seconds")
    for method in args.methods:
        for state_kind in args.states:
            for transform in args.transforms:
                run, arg = build_transformed_runner(method, state_kind, transform)
                samples = timeit.repeat(
                    partial(compile_once, run, arg),
                    repeat=args.repeat,
                    number=1,
                )
                print(
                    f"{method},{state_kind},{transform},"
                    f"{statistics.median(samples):.6f}"
                )


if __name__ == "__main__":
    main()
