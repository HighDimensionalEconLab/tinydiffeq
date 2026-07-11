"""Reproducible tinydiffeq/Diffrax endpoint benchmarks.

Compilation is excluded from steady-state timings. Every timed JAX call blocks on
its full output. Run through ``../run.sh`` so backend selection happens before JAX
is imported.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import time
import timeit
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import diffrax as dfx
import jax
import jax.numpy as jnp
import numpy as np

import tinydiffeq as td

jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class Case:
    equation: str
    size: int
    method: str
    stepping: str
    controller: str
    dtype: str
    transform: str
    batch: int = 1


DTYPE = {"float32": jnp.float32, "float64": jnp.float64}
TOLERANCES = {
    "float32": [(1e-4, 1e-6, "common")],
    "float64": [(1e-4, 1e-6, "common"), (1e-7, 1e-9, "precision")],
}


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[2],
    ).stdout.strip()


def _git_dirty() -> bool:
    return bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[2],
        ).stdout
    )


def _sync(value: Any) -> None:
    jax.block_until_ready(value)


def _measure(run: Callable[[Any], Any], argument: Any, quick: bool) -> dict[str, Any]:
    start = time.perf_counter()
    compiled = jax.jit(run).lower(argument).compile()
    compile_seconds = time.perf_counter() - start
    start = time.perf_counter()
    _sync(compiled(argument))
    first_execute_seconds = time.perf_counter() - start

    def invoke() -> None:
        _sync(compiled(argument))

    number = 1
    target = 0.03 if quick else 0.15
    while number < 4096:
        elapsed = timeit.timeit(invoke, number=number)
        if elapsed >= target:
            break
        number *= 2
    repeats = 3 if quick else 9
    samples = [
        value / number for value in timeit.repeat(invoke, number=number, repeat=repeats)
    ]
    ordered = sorted(samples)
    q1, _, q3 = statistics.quantiles(ordered, n=4, method="inclusive")
    return {
        "median_seconds": statistics.median(samples),
        "iqr_seconds": q3 - q1,
        "samples_seconds": samples,
        "iterations_per_sample": number,
        "compile_seconds": compile_seconds,
        "first_execute_seconds": first_execute_seconds,
    }


def _transform(
    solve: Callable[[jax.Array], jax.Array], initial: jax.Array, transform: str
) -> Callable[[jax.Array], Any]:
    tangent = jnp.ones_like(initial)
    if transform == "primal":
        return solve
    if transform == "jvp":
        return lambda value: jax.jvp(solve, (value,), (tangent,))
    return lambda value: jax.value_and_grad(lambda value: jnp.sum(solve(value)))(value)


def _ode_config(equation: str, dtype: Any) -> dict[str, Any]:
    if equation == "ode_scalar":
        return {
            "initial": jnp.asarray(1.0, dtype),
            "t_0": jnp.asarray(0.0, dtype),
            "t_1": jnp.asarray(10.0, dtype),
            "n_steps": 64,
            "dt_adaptive": jnp.asarray(0.1, dtype),
            "max_steps": 512,
        }
    n = 256
    grid = jnp.arange(n, dtype=dtype)
    return {
        "initial": jnp.sin(2 * jnp.asarray(jnp.pi, dtype) * grid / n),
        "t_0": jnp.asarray(0.0, dtype),
        "t_1": jnp.asarray(1.0, dtype),
        "n_steps": 128,
        "dt_adaptive": jnp.asarray(0.01, dtype),
        "max_steps": 512,
    }


def _ode_field(equation: str) -> Callable[..., jax.Array]:
    if equation == "ode_scalar":
        return lambda x, t, args, p: p * x

    def field(x, t, args, p):
        return p * x + 0.1 * (jnp.roll(x, 1) - 2 * x + jnp.roll(x, -1))

    return field


def _tiny_controller(name: str, rtol: float, atol: float, dtype: Any, t_1: Any):
    dt_min = 10 * jnp.finfo(dtype).eps * jnp.maximum(1, jnp.abs(t_1))
    kwargs = dict(
        rtol=rtol,
        atol=atol,
        dt_min=float(dt_min),
        dt_max=float("inf"),
        safety=0.9,
        factor_min=0.2,
        factor_max=5.0,
    )
    if name == "i":
        return td.IController(**kwargs)
    return td.PIController(p_coeff=0.4, i_coeff=0.3, **kwargs)


def _diffrax_controller(name: str, rtol: float, atol: float, dtype: Any, t_1: Any):
    dt_min = 10 * jnp.finfo(dtype).eps * jnp.maximum(1, jnp.abs(t_1))
    coefficients = (0.0, 1.0) if name == "i" else (0.4, 0.3)
    return dfx.PIDController(
        rtol=jnp.asarray(rtol, dtype),
        atol=jnp.asarray(atol, dtype),
        pcoeff=coefficients[0],
        icoeff=coefficients[1],
        dcoeff=0,
        dtmin=dt_min,
        dtmax=jnp.asarray(jnp.inf, dtype),
        factormin=0.2,
        factormax=5.0,
        safety=0.9,
        norm=lambda value: jnp.max(jnp.abs(value)),
    )


def _tiny_ode_solve(case: Case, rtol: float, atol: float):
    dtype = DTYPE[case.dtype]
    config = _ode_config(case.equation, dtype)
    field = _ode_field(case.equation)
    solver = {
        "euler": td.Euler(),
        "rk4": td.RK4(),
        "tsit5": td.Tsit5(),
        "rodas5p": td.Rodas5P(),
    }[case.method]
    adaptive = case.stepping == "adaptive"
    controller = (
        _tiny_controller(case.controller, rtol, atol, dtype, config["t_1"])
        if adaptive
        else td.ConstantStepSize()
    )
    dt = (
        config["dt_adaptive"]
        if adaptive
        else (config["t_1"] - config["t_0"]) / config["n_steps"]
    )
    max_steps = config["max_steps"] if adaptive else config["n_steps"]

    def solve(initial):
        return td.solve_ode(
            field,
            solver,
            config["t_0"],
            config["t_1"],
            initial,
            p=jnp.asarray(-0.2, dtype),
            dt_0=dt,
            controller=controller,
            max_steps=max_steps,
        ).xs

    def stats(initial):
        sol = td.solve_ode(
            field,
            solver,
            config["t_0"],
            config["t_1"],
            initial,
            p=jnp.asarray(-0.2, dtype),
            dt_0=dt,
            controller=controller,
            max_steps=max_steps,
        )
        return {"accepted_steps": int(sol.num_accepted), "ok": bool(sol.ok)}

    return solve, stats, config["initial"], float(dt), max_steps


def _diffrax_ode_solve(case: Case, rtol: float, atol: float):
    dtype = DTYPE[case.dtype]
    config = _ode_config(case.equation, dtype)
    field = _ode_field(case.equation)
    if case.method == "rk4":
        raise NotImplementedError("Diffrax has no public classic RK4 solver")
    if case.method == "rodas5p":
        raise NotImplementedError("Diffrax has no Rodas5P implementation")
    solver = {"euler": dfx.Euler(), "tsit5": dfx.Tsit5()}[case.method]
    adaptive = case.stepping == "adaptive"
    controller = (
        _diffrax_controller(case.controller, rtol, atol, dtype, config["t_1"])
        if adaptive
        else dfx.ConstantStepSize()
    )
    dt = (
        config["dt_adaptive"]
        if adaptive
        else (config["t_1"] - config["t_0"]) / config["n_steps"]
    )
    max_steps = config["max_steps"] if adaptive else config["n_steps"]
    adjoint = (
        dfx.ForwardMode()
        if case.transform == "jvp"
        else dfx.RecursiveCheckpointAdjoint()
    )
    term = dfx.ODETerm(lambda t, x, p: field(x, t, None, p))

    def full(initial):
        return dfx.diffeqsolve(
            term,
            solver,
            t0=config["t_0"],
            t1=config["t_1"],
            dt0=dt,
            y0=initial,
            args=jnp.asarray(-0.2, dtype),
            stepsize_controller=controller,
            saveat=dfx.SaveAt(t1=True),
            adjoint=adjoint,
            max_steps=max_steps,
            throw=False,
        )

    def solve(initial):
        return full(initial).ys[0]

    def stats(initial):
        sol = full(initial)
        return {
            "accepted_steps": int(sol.stats["num_accepted_steps"]),
            "rejected_steps": int(sol.stats["num_rejected_steps"]),
            "ok": bool(sol.result == dfx.RESULTS.successful),
        }

    return solve, stats, config["initial"], float(dt), max_steps


def _tiny_dae_solve(case: Case, rtol: float, atol: float):
    dtype = DTYPE[case.dtype]
    size = case.size
    initial = (
        jnp.asarray(1.0, dtype)
        if size == 1
        else jnp.linspace(0.8, 1.2, size, dtype=dtype)
    )
    t_0, t_1 = jnp.asarray(0.0, dtype), jnp.asarray(1.0, dtype)
    n_steps = 64
    adaptive = case.stepping == "adaptive"
    dt = jnp.asarray(0.05, dtype) if adaptive else (t_1 - t_0) / n_steps
    max_steps = 256 if adaptive else n_steps
    solver = {
        "rk4": td.RK4(),
        "tsit5": td.Tsit5(),
        "rodas5p": td.Rodas5P(),
    }[case.method]
    controller = (
        _tiny_controller(case.controller, rtol, atol, dtype, t_1)
        if adaptive
        else td.ConstantStepSize()
    )

    def differential(y, z, t, args, p):
        return p * z

    def constraint(y, z, t, args, p):
        return z + 0.1 * (z**3 - y**3) - y

    def full(value):
        return td.solve_semi_explicit_dae(
            differential,
            constraint,
            solver,
            t_0,
            t_1,
            value,
            value,
            p=jnp.asarray(-0.2, dtype),
            dt_0=dt,
            controller=controller,
            max_steps=max_steps,
        )

    def solve(value):
        sol = full(value)
        return jnp.concatenate([jnp.ravel(sol.ys), jnp.ravel(sol.zs)])

    def stats(value):
        sol = full(value)
        return {"accepted_steps": int(sol.num_accepted), "ok": bool(sol.ok)}

    return solve, stats, initial, float(dt), max_steps


def _sde_initial(batch: int, dtype: Any) -> jax.Array:
    return jnp.linspace(0.8, 1.2, batch, dtype=dtype)


def _tiny_sde_solve(case: Case):
    dtype = DTYPE[case.dtype]
    initial = _sde_initial(case.batch, dtype)
    keys = jax.random.split(jax.random.key(1729), case.batch)
    n_steps = 128

    def single(value, key):
        return td.solve_sde(
            lambda x, t, args, p: p[0] * x,
            lambda x, t, args, p: p[1] * x,
            td.EulerMaruyama(),
            jnp.asarray(0.0, dtype),
            jnp.asarray(1.0, dtype),
            value,
            p=jnp.asarray([-0.2, 0.1], dtype),
            key=key,
            n_steps=n_steps,
        ).xs

    solve = jax.vmap(single, in_axes=(0, 0))
    return (
        solve,
        lambda value: {"accepted_steps": n_steps, "ok": True},
        initial,
        1 / n_steps,
        n_steps,
        keys,
    )


def _diffrax_sde_solve(case: Case):
    dtype = DTYPE[case.dtype]
    initial = _sde_initial(case.batch, dtype)
    keys = jax.random.split(jax.random.key(1729), case.batch)
    n_steps = 128
    dt = jnp.asarray(1 / n_steps, dtype)
    # UnsafeBrownianPath is intentionally used to match fixed-grid EM without
    # virtual-tree overhead. DirectAdjoint is the Diffrax transform supporting
    # both this Brownian path and forward/reverse discrete AD.
    adjoint = dfx.DirectAdjoint()

    def single(value, key):
        increments = jnp.sqrt(dt) * jax.random.normal(key, (n_steps,), dtype=dtype)
        brownian = dfx.LinearInterpolation(
            ts=jnp.linspace(
                jnp.asarray(0.0, dtype), jnp.asarray(1.0, dtype), n_steps + 1
            ),
            ys=jnp.concatenate([jnp.zeros((1,), dtype), jnp.cumsum(increments)]),
        )
        terms = dfx.MultiTerm(
            dfx.ODETerm(lambda t, x, p: p[0] * x),
            dfx.ControlTerm(lambda t, x, p: p[1] * x, brownian),
        )
        sol = dfx.diffeqsolve(
            terms,
            dfx.Euler(),
            t0=jnp.asarray(0.0, dtype),
            t1=jnp.asarray(1.0, dtype),
            dt0=dt,
            y0=value,
            args=jnp.asarray([-0.2, 0.1], dtype),
            stepsize_controller=dfx.ConstantStepSize(),
            saveat=dfx.SaveAt(t1=True),
            adjoint=adjoint,
            max_steps=n_steps,
            throw=False,
        )
        return sol.ys[0]

    solve = jax.vmap(single, in_axes=(0, 0))
    return (
        solve,
        lambda value: {"accepted_steps": n_steps, "ok": True},
        initial,
        1 / n_steps,
        n_steps,
        keys,
    )


def _tiny_sdae_solve(case: Case):
    dtype = DTYPE[case.dtype]
    initial = _sde_initial(case.batch, dtype)
    keys = jax.random.split(jax.random.key(1729), case.batch)
    n_steps = 128

    def single(value, key):
        sol = td.solve_semi_explicit_sdae(
            lambda y, z, t, args, p: p[0] * z,
            lambda y, z, t, args, p: p[1] * z,
            lambda y, z, t, args, p: z + 0.1 * (z**3 - y**3) - y,
            td.EulerMaruyama(),
            jnp.asarray(0.0, dtype),
            jnp.asarray(1.0, dtype),
            value,
            value,
            p=jnp.asarray([-0.2, 0.1], dtype),
            key=key,
            n_steps=n_steps,
        )
        return jnp.stack([sol.ys, sol.zs])

    solve = jax.vmap(single, in_axes=(0, 0))
    return (
        solve,
        lambda value: {"accepted_steps": n_steps, "ok": True},
        initial,
        1 / n_steps,
        n_steps,
        keys,
    )


def _wrap_keys(solve, keys):
    return lambda value: solve(value, keys)


def _controller_equivalence(dtype_name: str, controller_name: str) -> dict[str, Any]:
    """Compare the accepted scalar Tsit5 meshes under the requested policy."""
    dtype = DTYPE[dtype_name]
    rtol, atol, _ = TOLERANCES[dtype_name][0]
    t_0 = jnp.asarray(0.0, dtype)
    t_1 = jnp.asarray(10.0, dtype)
    dt_0 = jnp.asarray(0.1, dtype)
    initial = jnp.asarray(1.0, dtype)
    max_steps = 512
    tiny = td.solve_ode(
        lambda x, t, args, p: p * x,
        td.Tsit5(),
        t_0,
        t_1,
        initial,
        p=jnp.asarray(-0.2, dtype),
        dt_0=dt_0,
        controller=_tiny_controller(controller_name, rtol, atol, dtype, t_1),
        max_steps=max_steps,
        save_at=td.SaveAt(steps=True),
    )
    term = dfx.ODETerm(lambda t, x, p: p * x)
    other = dfx.diffeqsolve(
        term,
        dfx.Tsit5(),
        t0=t_0,
        t1=t_1,
        dt0=dt_0,
        y0=initial,
        args=jnp.asarray(-0.2, dtype),
        stepsize_controller=_diffrax_controller(
            controller_name, rtol, atol, dtype, t_1
        ),
        saveat=dfx.SaveAt(t0=True, steps=True),
        max_steps=max_steps,
        throw=False,
    )
    tiny_times = np.asarray(tiny.ts)[np.asarray(tiny.accepted)]
    other_times = np.asarray(other.ts)
    other_times = other_times[np.isfinite(other_times)]
    equivalent = len(tiny_times) == len(other_times) and np.allclose(
        tiny_times, other_times, rtol=8 * np.finfo(np.dtype(dtype_name)).eps, atol=0
    )
    return {
        "dtype": dtype_name,
        "controller": controller_name,
        "equivalent": bool(equivalent),
        "tinydiffeq_accepted": len(tiny_times) - 1,
        "diffrax_accepted": int(other.stats["num_accepted_steps"]),
        "max_mesh_difference": (
            float(np.max(np.abs(tiny_times - other_times)))
            if len(tiny_times) == len(other_times)
            else None
        ),
    }


def _run_case(library: str, case: Case, rtol: float, atol: float, quick: bool):
    if case.equation.startswith("ode"):
        builder = _tiny_ode_solve if library == "tinydiffeq" else _diffrax_ode_solve
        solve, stats, initial, dt, max_steps = builder(case, rtol, atol)
    elif case.equation.startswith("dae"):
        if library != "tinydiffeq":
            raise NotImplementedError("no equivalent semi-explicit DAE interface")
        solve, stats, initial, dt, max_steps = _tiny_dae_solve(case, rtol, atol)
    elif case.equation == "sde_ensemble":
        builder = _tiny_sde_solve if library == "tinydiffeq" else _diffrax_sde_solve
        solve_with_keys, stats, initial, dt, max_steps, keys = builder(case)
        solve = _wrap_keys(solve_with_keys, keys)
    else:
        if library != "tinydiffeq":
            raise NotImplementedError("no equivalent explicit projected SDAE interface")
        solve_with_keys, stats, initial, dt, max_steps, keys = _tiny_sdae_solve(case)
        solve = _wrap_keys(solve_with_keys, keys)

    transformed = _transform(solve, initial, case.transform)
    result = _measure(transformed, initial, quick)
    primal = jax.jit(solve)(initial)
    _sync(primal)
    if case.equation == "ode_scalar":
        reference = np.exp(-2.0)
        error = float(np.max(np.abs(np.asarray(primal) - reference)))
    else:
        error = None
    result.update(
        {
            "library": library,
            "case": asdict(case),
            "rtol": rtol,
            "atol": atol,
            "dt_0": dt,
            "max_steps": max_steps,
            "stats": stats(initial),
            "absolute_error": error,
        }
    )
    return result


def _cases(quick: bool) -> list[Case]:
    dtypes = ["float32", "float64"]
    transforms = ["primal", "jvp", "vjp"]
    cases: list[Case] = []
    for dtype in dtypes:
        for transform in transforms:
            for equation, size in [("ode_scalar", 1), ("ode_vector", 256)]:
                for method in ["euler", "rk4", "tsit5", "rodas5p"]:
                    cases.append(
                        Case(equation, size, method, "fixed", "none", dtype, transform)
                    )
                for controller in ["i", "pi"]:
                    cases.append(
                        Case(
                            equation,
                            size,
                            "tsit5",
                            "adaptive",
                            controller,
                            dtype,
                            transform,
                        )
                    )
                cases.append(
                    Case(
                        equation,
                        size,
                        "rodas5p",
                        "adaptive",
                        "i",
                        dtype,
                        transform,
                    )
                )
            for size in [1, 32]:
                cases.append(
                    Case(
                        f"dae_{'scalar' if size == 1 else 'vector'}",
                        size,
                        "rk4",
                        "fixed",
                        "none",
                        dtype,
                        transform,
                    )
                )
                cases.append(
                    Case(
                        f"dae_{'scalar' if size == 1 else 'vector'}",
                        size,
                        "rodas5p",
                        "fixed",
                        "none",
                        dtype,
                        transform,
                    )
                )
                cases.append(
                    Case(
                        f"dae_{'scalar' if size == 1 else 'vector'}",
                        size,
                        "rodas5p",
                        "adaptive",
                        "i",
                        dtype,
                        transform,
                    )
                )
                cases.append(
                    Case(
                        f"dae_{'scalar' if size == 1 else 'vector'}",
                        size,
                        "tsit5",
                        "adaptive",
                        "i",
                        dtype,
                        transform,
                    )
                )
            for batch in [1, 256, 16384]:
                cases.append(
                    Case(
                        "sde_ensemble",
                        1,
                        "em",
                        "fixed",
                        "none",
                        dtype,
                        transform,
                        batch,
                    )
                )
                cases.append(
                    Case(
                        "sdae_ensemble",
                        1,
                        "em",
                        "fixed",
                        "none",
                        dtype,
                        transform,
                        batch,
                    )
                )
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--library", choices=["tinydiffeq", "diffrax", "both"], default="both"
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--method")
    parser.add_argument(
        "--subset",
        choices=[
            "ode",
            "ode_scalar",
            "ode_vector",
            "dae",
            "dae_scalar",
            "dae_vector",
            "sde_ensemble",
            "sdae_ensemble",
        ],
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    libraries = ["tinydiffeq", "diffrax"] if args.library == "both" else [args.library]
    results = []
    unavailable = []
    cases = _cases(args.quick)
    if args.subset is not None:
        cases = [
            case
            for case in cases
            if case.equation == args.subset
            or (args.subset in {"ode", "dae"} and case.equation.startswith(args.subset))
        ]
    if args.method is not None:
        cases = [case for case in cases if case.method == args.method]
    for case in cases:
        for rtol, atol, tolerance_name in TOLERANCES[case.dtype]:
            if case.stepping != "adaptive" and tolerance_name != "common":
                continue
            for library in libraries:
                try:
                    record = _run_case(library, case, rtol, atol, args.quick)
                    record["tolerance_name"] = tolerance_name
                    results.append(record)
                    print(
                        f"{library:12} {case.equation:14} {case.method:5} "
                        f"{case.transform:6} {case.dtype:7} b={case.batch:<5} "
                        f"{record['median_seconds'] * 1e6:10.2f} us",
                        flush=True,
                    )
                except (
                    NotImplementedError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                ) as exc:
                    unavailable.append(
                        {
                            "library": library,
                            "case": asdict(case),
                            "rtol": rtol,
                            "atol": atol,
                            "reason": str(exc),
                        }
                    )
    payload = {
        "metadata": {
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "jax": jax.__version__,
            "diffrax": getattr(dfx, "__version__", "unknown"),
            "tinydiffeq_git": _git_sha(),
            "tinydiffeq_worktree_dirty": _git_dirty(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "quick": args.quick,
            "x64_enabled": bool(jax.config.x64_enabled),
            "environment": {
                name: os.environ.get(name)
                for name in ["JAX_PLATFORMS", "CUDA_VISIBLE_DEVICES", "XLA_FLAGS"]
            },
        },
        "controller_equivalence": [
            _controller_equivalence(dtype, controller)
            for dtype in ["float32", "float64"]
            for controller in ["i", "pi"]
        ],
        "results": results,
        "unavailable": unavailable,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
