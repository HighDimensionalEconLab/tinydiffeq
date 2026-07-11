"""Correctness and transformation checks for the Python comparison harness."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from benchmark import (
    Case,
    _diffrax_ode_solve,
    _diffrax_sde_solve,
    _tiny_dae_solve,
    _tiny_ode_solve,
    _tiny_sdae_solve,
    _tiny_sde_solve,
    _wrap_keys,
)

jax.config.update("jax_enable_x64", True)


def ready(value):
    return jax.block_until_ready(value)


def finite(tree):
    return all(np.all(np.isfinite(np.asarray(leaf))) for leaf in jax.tree.leaves(tree))


def validate_ode(dtype_name):
    dtype = np.dtype(dtype_name)
    tolerance = 2e-5 if dtype_name == "float32" else 2e-12
    records = []
    for equation, size in [("ode_scalar", 1), ("ode_vector", 256)]:
        for method in ["euler", "tsit5"]:
            case = Case(equation, size, method, "fixed", "none", dtype_name, "primal")
            tiny, _, initial, _, _ = _tiny_ode_solve(case, 1e-4, 1e-6)
            other, _, _, _, _ = _diffrax_ode_solve(case, 1e-4, 1e-6)
            jvp_case = Case(equation, size, method, "fixed", "none", dtype_name, "jvp")
            other_jvp_solve, _, _, _, _ = _diffrax_ode_solve(jvp_case, 1e-4, 1e-6)
            tiny_value = ready(jax.jit(tiny)(initial))
            other_value = ready(jax.jit(other)(initial))
            np.testing.assert_allclose(
                tiny_value, other_value, rtol=tolerance, atol=tolerance
            )
            tangent = jnp.ones_like(initial)
            tiny_jvp = ready(
                jax.jit(
                    lambda x, solve=tiny, direction=tangent: jax.jvp(
                        solve, (x,), (direction,)
                    )[1]
                )(initial)
            )
            other_jvp = ready(
                jax.jit(
                    lambda x, solve=other_jvp_solve, direction=tangent: jax.jvp(
                        solve, (x,), (direction,)
                    )[1]
                )(initial)
            )
            np.testing.assert_allclose(
                tiny_jvp, other_jvp, rtol=tolerance, atol=tolerance
            )
            assert finite((tiny_value, other_value, tiny_jvp, other_jvp))
            records.append(
                {
                    "equation": equation,
                    "method": method,
                    "dtype": dtype.name,
                    "max_endpoint_difference": float(
                        np.max(np.abs(np.asarray(tiny_value) - np.asarray(other_value)))
                    ),
                }
            )
    return records


def validate_dae(dtype_name):
    records = []
    for method in ["tsit5", "rodas5p"]:
        case = Case("dae_vector", 32, method, "adaptive", "i", dtype_name, "primal")
        solve, stats, initial, _, _ = _tiny_dae_solve(case, 1e-4, 1e-6)
        value = ready(jax.jit(solve)(initial))
        tangent = ready(
            jax.jit(
                lambda x, solve=solve: jax.jvp(solve, (x,), (jnp.ones_like(x),))[1]
            )(initial)
        )
        cotangent = ready(
            jax.jit(jax.grad(lambda x, solve=solve: jnp.sum(solve(x))))(initial)
        )
        assert stats(initial)["ok"]
        assert finite((value, tangent, cotangent))
        y, z = np.split(np.asarray(value), 2)
        residual = z + 0.1 * (z**3 - y**3) - y
        threshold = 2e-4 if dtype_name == "float32" else 2e-8
        assert np.max(np.abs(residual)) < threshold
        records.append(
            {
                "dtype": dtype_name,
                "method": method,
                "max_constraint_residual": float(np.max(np.abs(residual))),
            }
        )
    return records


def validate_stochastic(dtype_name):
    case = Case("sde_ensemble", 1, "em", "fixed", "none", dtype_name, "primal", 256)
    tiny_with_keys, _, initial, _, _, tiny_keys = _tiny_sde_solve(case)
    other_with_keys, _, _, _, _, other_keys = _diffrax_sde_solve(case)
    tiny = _wrap_keys(tiny_with_keys, tiny_keys)
    other = _wrap_keys(other_with_keys, other_keys)
    tiny_value = ready(jax.jit(tiny)(initial))
    replay = ready(jax.jit(tiny)(initial))
    other_value = ready(jax.jit(other)(initial))
    np.testing.assert_array_equal(tiny_value, replay)
    tolerance = 3e-5 if dtype_name == "float32" else 3e-12
    np.testing.assert_allclose(tiny_value, other_value, rtol=tolerance, atol=tolerance)

    sdae_case = Case(
        "sdae_ensemble", 1, "em", "fixed", "none", dtype_name, "primal", 256
    )
    sdae_with_keys, _, sdae_initial, _, _, keys = _tiny_sdae_solve(sdae_case)
    sdae = _wrap_keys(sdae_with_keys, keys)
    sdae_value = ready(jax.jit(sdae)(sdae_initial))
    sdae_jvp = ready(
        jax.jit(lambda x: jax.jvp(sdae, (x,), (jnp.ones_like(x),))[1])(sdae_initial)
    )
    sdae_vjp = ready(jax.jit(jax.grad(lambda x: jnp.sum(sdae(x))))(sdae_initial))
    assert finite((tiny_value, other_value, sdae_value, sdae_jvp, sdae_vjp))
    return {
        "dtype": dtype_name,
        "max_sde_cross_library_difference": float(
            np.max(np.abs(np.asarray(tiny_value) - np.asarray(other_value)))
        ),
        "independent_paths": int(np.unique(np.asarray(tiny_value)).size),
    }


def main():
    payload = {
        "backend": jax.default_backend(),
        "ode": sum((validate_ode(dtype) for dtype in ["float32", "float64"]), []),
        "dae": sum((validate_dae(dtype) for dtype in ["float32", "float64"]), []),
        "stochastic": [validate_stochastic(dtype) for dtype in ["float32", "float64"]],
    }
    output = (
        Path(__file__).resolve().parents[1]
        / "results"
        / f"validation_{payload['backend']}.json"
    )
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"validated {payload['backend']} comparisons -> {output}")


if __name__ == "__main__":
    main()
