"""Reproducible CPU/GPU primal benchmarks for finite-state Markov simulation."""

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
    kind: str
    states: int
    length: int
    batch: int
    dtype: str
    method: str


def synchronize(value):
    jax.block_until_ready(value)


def measure(function, argument, quick):
    start = time.perf_counter()
    compiled = jax.jit(function).lower(argument).compile()
    compile_seconds = time.perf_counter() - start
    start = time.perf_counter()
    synchronize(compiled(argument))
    first_execute_seconds = time.perf_counter() - start
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
    ordered = sorted(samples)
    q1, _, q3 = statistics.quantiles(ordered, n=4, method="inclusive")
    return {
        "median_seconds": statistics.median(samples),
        "iqr_seconds": q3 - q1,
        "compile_seconds": compile_seconds,
        "first_execute_seconds": first_execute_seconds,
        "iterations_per_sample": number,
    }


def transition_matrix(num_states, dtype):
    generator = np.random.default_rng(num_states)
    values = generator.uniform(0.1, 1.0, (num_states, num_states))
    values /= values.sum(axis=1, keepdims=True)
    return jnp.asarray(values, dtype)


def generator_matrix(num_states, dtype):
    generator = np.random.default_rng(num_states + 10_000)
    off_diagonal = generator.uniform(0.0, 1.0, (num_states, num_states))
    np.fill_diagonal(off_diagonal, 0.0)
    off_diagonal /= np.maximum(off_diagonal.sum(axis=1, keepdims=True), 1e-12)
    off_diagonal *= generator.uniform(0.5, 2.0, (num_states, 1))
    values = off_diagonal.copy()
    np.fill_diagonal(values, -off_diagonal.sum(axis=1))
    return jnp.asarray(values, dtype)


def build(case):
    dtype = jnp.float32 if case.dtype == "float32" else jnp.float64
    method = {
        "sequential": td.SequentialMarkov(),
        "sequential_unroll16": td.SequentialMarkov(unroll=16),
        "associative": td.AssociativeMarkov(),
    }[case.method]
    start = time.perf_counter()
    if case.kind == "discrete":
        chain = td.DiscreteMarkovChain(transition_matrix(case.states, dtype))
    else:
        chain = td.ContinuousTimeMarkovChain(generator_matrix(case.states, dtype))
    preparation_seconds = time.perf_counter() - start
    keys = jax.random.split(jax.random.key(1701), case.batch)

    if case.kind == "discrete":

        def one(key):
            return td.simulate_markov_chain(
                chain,
                jnp.int32(0),
                key=key,
                num_steps=case.length,
                method=method,
                save_at=td.SaveAt(steps=True),
            ).xs
    else:

        def one(key):
            return td.simulate_continuous_time_markov_chain(
                chain,
                jnp.asarray(0.0, dtype),
                jnp.asarray(case.length / 2, dtype),
                jnp.int32(0),
                key=key,
                max_jumps=case.length,
                method=method,
                save_at=td.SaveAt(steps=True),
            ).xs

    return jax.vmap(one), keys, preparation_seconds


def cases(quick):
    if quick:
        main_cases = [
            Case(kind, 8, 1024, batch, dtype, method)
            for kind in ("discrete", "continuous")
            for batch in (256, 4096)
            for dtype in ("float32", "float64")
            for method in ("sequential", "sequential_unroll16", "associative")
        ]
        sciml_match = [
            Case("continuous", 2, 64, 256, dtype, method)
            for dtype in ("float32", "float64")
            for method in ("sequential", "sequential_unroll16", "associative")
        ]
        return main_cases + sciml_match
    return [
        Case(kind, states, length, batch, dtype, method)
        for kind in ("discrete", "continuous")
        for states in (4, 8, 32, 128)
        for length in (64, 256, 1024)
        for batch in (1, 256, 4096)
        for dtype in ("float32", "float64")
        for method in ("sequential", "sequential_unroll16", "associative")
        if not (states == 128 and length == 1024 and batch == 4096)
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--kind", choices=["discrete", "continuous"])
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    selected = cases(arguments.quick)
    if arguments.kind is not None:
        selected = [case for case in selected if case.kind == arguments.kind]
    results = []
    for case in selected:
        function, keys, preparation_seconds = build(case)
        result = measure(function, keys, arguments.quick)
        result.update(
            case=asdict(case),
            preparation_seconds=preparation_seconds,
        )
        results.append(result)
        print(
            f"{case.kind:10} K={case.states:<3} T={case.length:<4} "
            f"B={case.batch:<4} {case.dtype:7} {case.method:19} "
            f"{result['median_seconds'] * 1e6:10.2f} us",
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
