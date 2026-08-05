"""Train the kernels growth policies and export their weights for benchmarks.

Runs the kernels package's neoclassical and stochastic growth solvers
in-process (the package keeps trained nnx modules in memory only), extracts
the MLP weights and structural parameters as plain arrays, verifies a pure-jnp
reimplementation of the policy forward pass against the nnx module, and writes
`benchmarks/policies/<model>.npz` consumed by `gpu_trajectories.py`.

Run from the kernels venv (it owns flax and the model modules):

    uv run --project /Users/jlperla/GitHub/kernels python \
        benchmarks/export_growth_policies.py
"""

import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, "/Users/jlperla/GitHub/kernels")

import jax
import jax.numpy as jnp
import numpy as np

OUTPUT_DIR = Path(__file__).parent / "policies"


def unit_softplus(raw):
    log_two = jnp.log(jnp.asarray(2.0, dtype=raw.dtype))
    return jax.nn.softplus(2.0 * log_two * raw) / log_two


def extract(policy, structural):
    weights = {}
    for index, layer in enumerate(policy.layers):
        weights[f"layer{index}_kernel"] = np.asarray(layer.kernel.value)
        weights[f"layer{index}_bias"] = np.asarray(layer.bias.value)
    weights["head_kernel"] = np.asarray(policy.head.kernel.value)
    weights["head_bias"] = np.asarray(policy.head.bias.value)
    metadata = {
        "num_layers": len(policy.layers),
        "homothetic": bool(policy.homothetic),
        "log_z_scale": float(getattr(policy, "log_z_scale", 0.0)),
        "log_k_scale": float(getattr(policy, "log_k_scale", 0.0)),
        "k_scale": float(getattr(policy, "k_scale", 1.0)),
        "structural": {
            key: (
                np.asarray(value).item()
                if np.asarray(value).ndim == 0
                else np.asarray(value).tolist()
            )
            for key, value in dataclasses.asdict(structural).items()
        },
    }
    return weights, metadata


def pure_policy(weights, metadata, gelu_approximate):
    num_layers = metadata["num_layers"]
    delta = metadata["structural"]["delta"]

    def forward(log_state):
        log_z, log_k = log_state
        if metadata["homothetic"]:
            h = jnp.atleast_1d(log_k - log_z)
            anchor = delta * jnp.exp(log_k)
        else:
            h = jnp.stack(
                [log_z - metadata["log_z_scale"], log_k - metadata["log_k_scale"]]
            )
            anchor = delta * metadata["k_scale"]
        for index in range(num_layers):
            h = jax.nn.gelu(
                h @ weights[f"layer{index}_kernel"] + weights[f"layer{index}_bias"],
                approximate=gelu_approximate,
            )
        raw = h @ weights["head_kernel"] + weights["head_bias"]
        return anchor * unit_softplus(raw)[0]

    return forward


def verify_and_save(name, policy, structural):
    weights, metadata = extract(policy, structural)
    states = jnp.stack(
        [
            0.3 * jax.random.normal(jax.random.key(0), (256,)),
            0.3 * jax.random.normal(jax.random.key(1), (256,)),
        ],
        axis=1,
    )
    reference = jax.vmap(lambda s: policy(s, structural))(states)
    gelu_approximate = None
    for candidate in (True, False):
        ours = jax.vmap(pure_policy(weights, metadata, candidate))(states)
        gap = float(jnp.max(jnp.abs(ours - reference)))
        if gap < 1e-10:
            gelu_approximate = candidate
            break
    if gelu_approximate is None:
        raise AssertionError(f"{name}: pure-jnp policy mismatch, max gap {gap}")
    metadata["gelu_approximate"] = gelu_approximate
    OUTPUT_DIR.mkdir(exist_ok=True)
    np.savez(OUTPUT_DIR / f"{name}.npz", **weights)
    (OUTPUT_DIR / f"{name}.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        f"{name}: exported {sum(w.size for w in weights.values())} params, "
        f"gelu_approximate={gelu_approximate}, parity gap < 1e-10"
    )


def main():
    from neoclassical_growth_nn_recursive import neoclassical_growth_nn_recursive

    results, extras = neoclassical_growth_nn_recursive(use_float64=True)
    verify_and_save("neoclassical_growth", extras["model"], extras["parameters"])

    from stochastic_growth_nn_recursive import stochastic_growth_nn_recursive

    results, extras = stochastic_growth_nn_recursive(use_float64=True)
    verify_and_save("stochastic_growth", extras["model"], extras["parameters"])


if __name__ == "__main__":
    main()
