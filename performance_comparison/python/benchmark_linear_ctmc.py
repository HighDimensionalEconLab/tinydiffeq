"""Large matrix-free linear-CTMC primal, VJP, and ensemble benchmarks."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
import timeit
from pathlib import Path

import jax
import jax.numpy as jnp

import tinydiffeq as td


def synchronize(value):
    jax.block_until_ready(value)


def measure(function, arguments, quick):
    start = time.perf_counter()
    compiled = jax.jit(function).lower(*arguments).compile()
    compile_seconds = time.perf_counter() - start
    synchronize(compiled(*arguments))
    number = 1
    target = 0.1 if quick else 0.5

    def invoke():
        synchronize(compiled(*arguments))

    while number < 1024 and timeit.timeit(invoke, number=number) < target:
        number *= 2
    repeats = 5 if quick else 15
    samples = [
        elapsed / number
        for elapsed in timeit.repeat(invoke, number=number, repeat=repeats)
    ]
    q1, _, q3 = statistics.quantiles(sorted(samples), n=4, method="inclusive")
    return {
        "median_seconds": statistics.median(samples),
        "iqr_seconds": q3 - q1,
        "compile_seconds": compile_seconds,
        "iterations_per_sample": number,
    }


def make_problem(dtype, num_states):
    indices = jnp.arange(num_states, dtype=dtype)
    denominator = jnp.asarray(max(num_states - 1, 1), dtype)
    rates = jnp.asarray(0.1, dtype) + jnp.asarray(0.9, dtype) * indices / denominator
    initial = 1 + jnp.sin(jnp.asarray(0.017, dtype) * indices) ** 2
    initial = initial / jnp.sum(initial)
    cotangent = jnp.cos(jnp.asarray(0.013, dtype) * indices) + jnp.asarray(
        0.2, dtype
    ) * jnp.sin(jnp.asarray(0.007, dtype) * indices)
    return rates, initial, cotangent


def forward_action(rates, state):
    flux = rates * state
    return jnp.roll(flux, 1) - flux


def adjoint_action(rates, state):
    return rates * (jnp.roll(state, -1) - state)


def endpoint(rates, initial, method, horizon):
    return td.solve_linear_ode(
        lambda state: forward_action(rates, state),
        method,
        jnp.asarray(0.0, initial.dtype),
        horizon,
        initial,
    ).xs


def adjoint_endpoint(rates, cotangent, method, horizon):
    return td.solve_linear_ode(
        lambda state: adjoint_action(rates, state),
        method,
        jnp.asarray(0.0, cotangent.dtype),
        horizon,
        cotangent,
    ).xs


def ensemble_initials(initial, batch_size):
    indices = jnp.arange(initial.size, dtype=initial.dtype)
    phases = jnp.arange(batch_size, dtype=initial.dtype)[:, None]
    perturbation = 1 + jnp.asarray(0.1, initial.dtype) * jnp.sin(
        jnp.asarray(0.011, initial.dtype) * indices[None]
        + jnp.asarray(0.37, initial.dtype) * phases
    )
    values = initial[None] * perturbation
    return values / jnp.sum(values, axis=1, keepdims=True)


def benchmark_case(dtype, dtype_name, num_states, quick, reorthogonalization_passes):
    rates, initial, cotangent = make_problem(dtype, num_states)
    horizon = jnp.asarray(10.0, dtype)
    tolerance = 1e-5 if dtype == jnp.float32 else 1e-10
    method = td.KrylovExponential(
        krylov_dim=30,
        num_substeps=2,
        reorthogonalization_passes=reorthogonalization_passes,
        rtol=tolerance,
        atol=tolerance * 1e-2,
    )
    adaptive_method = td.AdaptiveKrylovExponential(
        krylov_dim=30,
        max_steps=128,
        reorthogonalization_passes=reorthogonalization_passes,
        rtol=tolerance,
        atol=tolerance * 1e-2,
    )

    def primal(current_rates, current_initial):
        return endpoint(current_rates, current_initial, method, horizon)

    def value_and_vjp(current_rates, current_initial, current_cotangent):
        value, pullback = jax.vjp(
            lambda state: endpoint(current_rates, state, method, horizon),
            current_initial,
        )
        return value, pullback(current_cotangent)[0]

    def value_and_jvp(current_rates, current_initial, current_tangent):
        return jax.jvp(
            lambda state: endpoint(current_rates, state, method, horizon),
            (current_initial,),
            (current_tangent,),
        )

    def handcoded_value_and_jvp(current_rates, current_initial, current_tangent):
        solution, tangent = td.jvp_linear_ode(
            lambda state: forward_action(current_rates, state),
            method,
            jnp.asarray(0.0, current_initial.dtype),
            horizon,
            current_initial,
            current_tangent,
        )
        return solution.xs, tangent

    def handcoded_value_and_vjp(current_rates, current_initial, current_cotangent):
        solution, gradient = td.vjp_linear_ode(
            lambda state: forward_action(current_rates, state),
            method,
            jnp.asarray(0.0, current_initial.dtype),
            horizon,
            current_initial,
            current_cotangent,
        )
        return solution.xs, gradient

    def manual_value_and_vjp(current_rates, current_initial, current_cotangent):
        return (
            endpoint(current_rates, current_initial, method, horizon),
            adjoint_endpoint(current_rates, current_cotangent, method, horizon),
        )

    def adaptive_primal(current_rates, current_initial):
        return endpoint(current_rates, current_initial, adaptive_method, horizon)

    def adaptive_handcoded_value_and_vjp(
        current_rates, current_initial, current_cotangent
    ):
        solution, gradient = td.vjp_linear_ode(
            lambda state: forward_action(current_rates, state),
            adaptive_method,
            jnp.asarray(0.0, current_initial.dtype),
            horizon,
            current_initial,
            current_cotangent,
        )
        return solution.xs, gradient

    batch_size = 256 if num_states == 10_000 else 16
    batched_initial = ensemble_initials(initial, batch_size)

    def batched_primal(current_rates, current_initials):
        return jax.vmap(lambda state: endpoint(current_rates, state, method, horizon))(
            current_initials
        )

    primal_timing = measure(primal, (rates, initial), quick)
    jvp_timing = measure(value_and_jvp, (rates, initial, cotangent), quick)
    handcoded_jvp_timing = measure(
        handcoded_value_and_jvp, (rates, initial, cotangent), quick
    )
    vjp_timing = measure(value_and_vjp, (rates, initial, cotangent), quick)
    handcoded_vjp_timing = measure(
        handcoded_value_and_vjp, (rates, initial, cotangent), quick
    )
    manual_vjp_timing = measure(
        manual_value_and_vjp, (rates, initial, cotangent), quick
    )
    batched_timing = measure(batched_primal, (rates, batched_initial), quick)
    adaptive_primal_timing = measure(adaptive_primal, (rates, initial), quick)
    adaptive_vjp_timing = measure(
        adaptive_handcoded_value_and_vjp, (rates, initial, cotangent), quick
    )
    adaptive_probe = td.solve_linear_ode(
        lambda state: forward_action(rates, state),
        adaptive_method,
        jnp.asarray(0.0, dtype),
        horizon,
        initial,
    )

    value, gradient = jax.jit(value_and_vjp)(rates, initial, cotangent)
    _, manual_gradient = jax.jit(manual_value_and_vjp)(rates, initial, cotangent)
    relative_vjp_error = jnp.linalg.norm(gradient - manual_gradient) / jnp.linalg.norm(
        manual_gradient
    )
    return {
        "states": num_states,
        "dtype": dtype_name,
        "horizon": 10.0,
        "krylov_dim": 30,
        "num_substeps": 2,
        "tolerance": tolerance,
        "reorthogonalization_passes": reorthogonalization_passes,
        "batch_size": batch_size,
        "primal": primal_timing,
        "value_and_jvp_ad": jvp_timing,
        "value_and_jvp_handcoded": handcoded_jvp_timing,
        "value_and_vjp_ad": vjp_timing,
        "value_and_vjp_handcoded": handcoded_vjp_timing,
        "value_and_vjp_manual": manual_vjp_timing,
        "batched_primal": batched_timing,
        "adaptive_primal": adaptive_primal_timing,
        "adaptive_value_and_vjp_handcoded": adaptive_vjp_timing,
        "adaptive_accepted_steps": int(adaptive_probe.num_accepted),
        "adaptive_ok": bool(adaptive_probe.ok),
        "mass_error": float(jnp.abs(jnp.sum(value) - 1)),
        "objective": float(jnp.vdot(cotangent, value)),
        "vjp_norm": float(jnp.linalg.norm(gradient)),
        "relative_vjp_error_vs_adjoint_action": float(relative_vjp_error),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--states", type=int)
    parser.add_argument("--dtype", choices=("float32", "float64"))
    parser.add_argument("--passes", type=int, choices=(1, 2), default=2)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    sizes = (10_000, 100_000) if arguments.quick else (10_000, 100_000, 1_000_000)
    results = []
    for dtype, dtype_name in ((jnp.float32, "float32"), (jnp.float64, "float64")):
        if arguments.dtype is not None and dtype_name != arguments.dtype:
            continue
        for num_states in sizes:
            if arguments.states is not None and num_states != arguments.states:
                continue
            result = benchmark_case(
                dtype,
                dtype_name,
                num_states,
                arguments.quick,
                arguments.passes,
            )
            results.append(result)
            vjp_microseconds = 1e6 * result["value_and_vjp_ad"]["median_seconds"]
            print(
                f"tiny {dtype_name} N={num_states} "
                f"primal={1e6 * result['primal']['median_seconds']:.2f} us "
                f"value+VJP={vjp_microseconds:.2f} us "
                f"vmap(B={result['batch_size']})="
                f"{1e6 * result['batched_primal']['median_seconds']:.2f} us",
                flush=True,
            )
    payload = {
        "metadata": {
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "jax": jax.__version__,
            "python": platform.python_version(),
            "quick": arguments.quick,
        },
        "results": results,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
