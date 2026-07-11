"""Compare direct and cached-linearization ODE JVP/VJP strategies."""

from __future__ import annotations

import argparse
import statistics
import timeit

import jax
import jax.numpy as jnp

from tinydiffeq import Tsit5, solve_ode


def synchronize(value):
    jax.block_until_ready(value)


def median_microseconds(function, *arguments, repeat, number):
    synchronize(function(*arguments))
    samples = timeit.repeat(
        lambda: synchronize(function(*arguments)), number=number, repeat=repeat
    )
    return 1e6 * statistics.median(samples) / number


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directions", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--number", type=int, default=32)
    args = parser.parse_args()

    dtype = jnp.float64 if jax.config.x64_enabled else jnp.float32
    grid = jnp.arange(args.size, dtype=dtype)
    x_0 = jnp.sin(2 * jnp.asarray(jnp.pi, dtype) * grid / args.size)
    p = jnp.asarray(-0.2, dtype)

    def field(x, t, unused_args, parameter):
        return parameter * x + 0.1 * (jnp.roll(x, 1) - 2 * x + jnp.roll(x, -1))

    def endpoint(x):
        return solve_ode(
            field,
            Tsit5(),
            0.0,
            1.0,
            x,
            p=p,
            dt_0=jnp.asarray(1 / args.steps, dtype),
            max_steps=args.steps,
        ).xs

    primal, pushforward = jax.linearize(endpoint, x_0)
    synchronize(primal)
    cached_pushforward = jax.jit(jax.vmap(pushforward))
    _, pullback = jax.vjp(endpoint, x_0)
    cached_pullback = jax.jit(jax.vmap(lambda cotangent: pullback(cotangent)[0]))

    print("directions,vmap_jvp_us,cached_pushforward_us,vmap_vjp_us,cached_pullback_us")
    for count in args.directions:
        if count > args.size:
            raise ValueError("each direction count must not exceed --size")
        tangents = jnp.eye(args.size, dtype=dtype)[:count]

        @jax.jit
        def batched_jvp(x, vectors):
            return jax.vmap(lambda vector: jax.jvp(endpoint, (x,), (vector,))[1])(
                vectors
            )

        @jax.jit
        def batched_vjp(x, vectors):
            def one(vector):
                _, one_pullback = jax.vjp(endpoint, x)
                return one_pullback(vector)[0]

            return jax.vmap(one)(vectors)

        jvp_time = median_microseconds(
            batched_jvp,
            x_0,
            tangents,
            repeat=args.repeat,
            number=args.number,
        )
        cached_time = median_microseconds(
            cached_pushforward,
            tangents,
            repeat=args.repeat,
            number=args.number,
        )
        vjp_time = median_microseconds(
            batched_vjp,
            x_0,
            tangents,
            repeat=args.repeat,
            number=args.number,
        )
        pullback_time = median_microseconds(
            cached_pullback,
            tangents,
            repeat=args.repeat,
            number=args.number,
        )
        print(
            f"{count},{jvp_time:.2f},{cached_time:.2f},"
            f"{vjp_time:.2f},{pullback_time:.2f}"
        )


if __name__ == "__main__":
    main()
