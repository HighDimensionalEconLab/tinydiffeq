"""Reproducible deterministic Markov-distribution benchmarks."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
import timeit
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import tinydiffeq as td


@dataclass(frozen=True)
class Case:
    problem: str
    states: int
    points: int
    batch: int
    dtype: str
    method: str


def synchronize(value):
    jax.block_until_ready(value)


def measure(function, argument, quick):
    start = time.perf_counter()
    compiled = jax.jit(function).lower(argument).compile()
    compile_seconds = time.perf_counter() - start
    synchronize(compiled(argument))
    number = 1
    target = 0.03 if quick else 0.15

    def invoke():
        synchronize(compiled(argument))

    while number < 4096 and timeit.timeit(invoke, number=number) < target:
        number *= 2
    samples = [
        elapsed / number
        for elapsed in timeit.repeat(invoke, number=number, repeat=3 if quick else 9)
    ]
    q1, _, q3 = statistics.quantiles(sorted(samples), n=4, method="inclusive")
    return {
        "median_seconds": statistics.median(samples),
        "iqr_seconds": q3 - q1,
        "compile_seconds": compile_seconds,
        "iterations_per_sample": number,
    }


def transition_matrix(num_states, dtype):
    rng = np.random.default_rng(num_states)
    values = rng.uniform(0.1, 1.0, (num_states, num_states))
    values /= values.sum(axis=1, keepdims=True)
    return jnp.asarray(values, dtype)


def generator_matrix(num_states, dtype):
    rng = np.random.default_rng(num_states + 10_000)
    rates = rng.uniform(0.1, 1.0, num_states)
    values = np.zeros((num_states, num_states))
    indices = np.arange(num_states)
    values[indices, indices] = -rates
    values[indices, (indices + 1) % num_states] = rates
    return jnp.asarray(values, dtype)


def initial_distributions(case, dtype):
    values = np.random.default_rng(case.states + case.batch).uniform(
        0.1, 1.0, (case.batch, case.states)
    )
    values /= values.sum(axis=1, keepdims=True)
    return jnp.asarray(values, dtype)


def build(case):
    dtype = jnp.float32 if case.dtype == "float32" else jnp.float64
    initial = initial_distributions(case, dtype)
    if case.problem.startswith("dtmc"):
        chain = td.DiscreteMarkovChain(transition_matrix(case.states, dtype))
        method = {
            "power": td.MatrixPowerMarkov(),
            "sequential": td.SequentialMarkov(),
            "associative": td.AssociativeMarkov(),
        }[case.method]
        save_at = (
            td.SaveAt(t_1=True)
            if case.problem == "dtmc_endpoint"
            else td.SaveAt(steps=True)
        )

        def one(distribution):
            return td.forecast_markov_chain(
                chain,
                distribution,
                num_steps=case.points,
                method=method,
                save_at=save_at,
            ).probabilities

        return jax.vmap(one), initial, None

    generator = generator_matrix(case.states, dtype)
    dense_chain = td.ContinuousTimeMarkovChain(generator)
    if case.method == "dense":
        chain = dense_chain
        method = td.DenseExponential()
    else:
        split = case.states // 2
        rates = -jnp.diag(generator)

        def action(probabilities):
            flat = jnp.concatenate([probabilities["left"], probabilities["right"]])
            flux = rates * flat
            result = jnp.roll(flux, 1) - flux
            return {"left": result[:split], "right": result[split:]}

        chain = td.MatrixFreeContinuousTimeMarkovChain(action)
        method = td.KrylovExponential(krylov_dim=min(30, case.states), num_substeps=2)
        initial = {"left": initial[:, :split], "right": initial[:, split:]}
    times = jnp.linspace(jnp.asarray(0.0, dtype), jnp.asarray(1.0, dtype), case.points)
    save_at = td.SaveAt(t_1=True) if case.points == 1 else td.SaveAt(ts=times)

    def one(distribution):
        return td.forecast_continuous_time_markov_chain(
            chain,
            jnp.asarray(0.0, dtype),
            jnp.asarray(1.0, dtype),
            distribution,
            method=method,
            save_at=save_at,
        ).probabilities

    reference_initial = (
        jnp.concatenate([initial["left"], initial["right"]], axis=1)
        if case.method == "krylov_pytree"
        else initial
    )

    def reference(distribution):
        return td.forecast_continuous_time_markov_chain(
            dense_chain,
            0.0,
            1.0,
            distribution,
            save_at=save_at,
        ).probabilities

    return jax.vmap(one), initial, (jax.vmap(reference), reference_initial)


def cases(quick):
    states = (8, 32, 128)
    batches = (1, 256) if quick else (1, 256, 4096)
    dtypes = ("float32", "float64")
    selected = []
    for num_states in states:
        for batch in batches:
            for dtype in dtypes:
                selected.extend(
                    [
                        Case("dtmc_endpoint", num_states, 1024, batch, dtype, "power"),
                        Case(
                            "dtmc_endpoint",
                            num_states,
                            1024,
                            batch,
                            dtype,
                            "sequential",
                        ),
                        Case("ctmc_endpoint", num_states, 1, batch, dtype, "dense"),
                        Case(
                            "ctmc_endpoint",
                            num_states,
                            1,
                            batch,
                            dtype,
                            "krylov_pytree",
                        ),
                    ]
                )
    for dtype in dtypes:
        selected.extend(
            [
                Case("dtmc_path", 8, 1024, 256, dtype, "sequential"),
                Case("dtmc_path", 8, 1024, 256, dtype, "associative"),
                Case("ctmc_grid", 32, 33, 1, dtype, "dense"),
                Case("ctmc_grid", 32, 33, 1, dtype, "krylov_pytree"),
            ]
        )
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--problem")
    parser.add_argument("--states", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    results = []
    selected = cases(arguments.quick)
    if arguments.problem is not None:
        selected = [case for case in selected if case.problem == arguments.problem]
    if arguments.states is not None:
        selected = [case for case in selected if case.states == arguments.states]
    if arguments.batch is not None:
        selected = [case for case in selected if case.batch == arguments.batch]
    for case in selected:
        function, argument, reference = build(case)
        timing = measure(function, argument, arguments.quick)
        maximum_dense_disagreement = None
        if reference is not None and case.method == "krylov_pytree":
            reference_function, reference_argument = reference
            actual = function(argument)
            actual_flat = jnp.concatenate([actual["left"], actual["right"]], axis=-1)
            expected = reference_function(reference_argument)
            maximum_dense_disagreement = float(jnp.max(jnp.abs(actual_flat - expected)))
        timing.update(
            case=asdict(case),
            maximum_dense_disagreement=maximum_dense_disagreement,
        )
        results.append(timing)
        print(
            f"{case.problem:13} K={case.states:<3} N={case.points:<4} "
            f"B={case.batch:<4} {case.dtype:7} {case.method:15} "
            f"{timing['median_seconds'] * 1e6:10.2f} us",
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
