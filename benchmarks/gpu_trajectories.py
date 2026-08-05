"""Trajectory-ensemble benchmarks for the kernels use case.

Measures vmapped, jitted solves over B trajectories with per-trajectory x_0 —
SRA1/EulerMaruyama SDEs with explicit or key-drawn noise, and fixed-step RK4
ODEs — in primal mode and as jit(grad) of a scalar residual with respect to
(x_0, p, noise). n_steps defaults include 31 and 127 so both a scan XLA may
unroll and one it will not are covered. Beyond the synthetic OU/MLP drifts,
`--drifts neoclassical stochastic-growth` load trained kernels-package
investment policies (see export_growth_policies.py) so the measured workload
is the real recursive-NN growth model, with the policy weights as the
differentiable parameters.

`--remat` sweeps the reverse-mode memory/compute trade (the axis where
diffrax's default RecursiveCheckpointAdjoint differs from a plain scan):
`none` is JAX's native O(n)-memory scan rule, `solve` wraps the whole
per-trajectory solve in jax.checkpoint, and `chunked` is a script-local
sqrt(n)-nested-scan checkpoint over the raw solver steps (endpoint-only,
explicit noise; verified against the library solve before timing). A chunked
win motivates a real checkpointing option in the package.

Run on a GPU node (or CPU for a smoke test):

    python -m benchmarks.gpu_trajectories --batch 1000 10000 --n-steps 31 127 1024
    python -m benchmarks.gpu_trajectories --drifts ou mlp --modes grad --remat chunked

Writes <output>.json (raw records) and <output>.md (summary table).
"""

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np

HIDDEN = 32
SIGMA = 0.1
THETA = 0.7
TARGET_SECONDS_PER_REPEAT = 0.05
MAX_ITERS = 1000
POLICY_DIR = Path(__file__).parent / "policies"
GROWTH_DRIFTS = {
    "neoclassical": "neoclassical_growth",
    "stochastic-growth": "stochastic_growth",
}
GROWTH_LOG_Z_VARIANCE = 0.012564
GROWTH_LOG_K_VARIANCE = 0.01


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--solvers", nargs="+", default=["sra1", "em", "rk4"])
    parser.add_argument("--batch", nargs="+", type=int, default=[100, 1000, 10000])
    parser.add_argument("--dim", nargs="+", type=int, default=[1, 8])
    parser.add_argument(
        "--n-steps", nargs="+", type=int, default=[31, 127, 1024], dest="n_steps"
    )
    parser.add_argument("--dtypes", nargs="+", default=["float32"])
    parser.add_argument(
        "--drifts",
        nargs="+",
        default=["ou"],
        choices=["ou", "mlp", "neoclassical", "stochastic-growth"],
        help="ou/mlp are synthetic; neoclassical (RK4 only) and "
        "stochastic-growth (SDE solvers only) load trained kernels policies "
        "from benchmarks/policies/ (see export_growth_policies.py)",
    )
    parser.add_argument(
        "--noise-modes",
        nargs="+",
        default=["explicit"],
        choices=["explicit", "key"],
        dest="noise_modes",
    )
    parser.add_argument("--save", nargs="+", default=["t1"], choices=["t1", "steps"])
    parser.add_argument("--modes", nargs="+", default=["primal", "grad"])
    parser.add_argument(
        "--remat",
        default="none",
        choices=["none", "solve", "chunked"],
        help="reverse-mode memory policy: plain scan, jax.checkpoint around "
        "the per-trajectory solve, or a sqrt(n)-chunked nested scan",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--matmul-precision",
        default=None,
        choices=["default", "tensorfloat32", "highest"],
        dest="matmul_precision",
    )
    parser.add_argument(
        "--scan-unroll",
        type=int,
        default=1,
        dest="scan_unroll",
        help="forwarded to the solvers' unroll= argument",
    )
    parser.add_argument(
        "--output",
        default="benchmarks/results/gpu_trajectories",
        help="prefix for the .json and .md outputs",
    )
    return parser.parse_args()


def unit_softplus(raw):
    log_two = jnp.log(jnp.asarray(2.0, dtype=raw.dtype))
    return jax.nn.softplus(2.0 * log_two * raw) / log_two


def load_growth_model(drift_kind, dtype):
    """Build (drift, diffusion, weights) from an exported kernels policy.

    The drift evaluates the trained investment-policy MLP at the (log z,
    log k) state, exactly as the kernels package does; the weights pytree is
    the differentiable `p` so grad mode measures the real training shape.
    """
    name = GROWTH_DRIFTS[drift_kind]
    metadata = json.loads((POLICY_DIR / f"{name}.json").read_text())
    raw = np.load(POLICY_DIR / f"{name}.npz")
    weights = {key: jnp.asarray(raw[key], dtype) for key in raw.files}
    structural = metadata["structural"]
    num_layers = metadata["num_layers"]
    homothetic = metadata["homothetic"]
    log_z_scale = metadata["log_z_scale"]
    log_k_scale = metadata["log_k_scale"]
    k_scale = metadata["k_scale"]
    approximate = metadata["gelu_approximate"]
    delta = structural["delta"]

    def policy(log_state, w):
        log_z, log_k = log_state
        if homothetic:
            h = jnp.atleast_1d(log_k - log_z)
            anchor = delta * jnp.exp(log_k)
        else:
            h = jnp.stack([log_z - log_z_scale, log_k - log_k_scale])
            anchor = delta * k_scale
        for index in range(num_layers):
            h = jax.nn.gelu(
                h @ w[f"layer{index}_kernel"] + w[f"layer{index}_bias"],
                approximate=approximate,
            )
        raw_head = h @ w["head_kernel"] + w["head_bias"]
        return anchor * unit_softplus(raw_head)[0]

    if drift_kind == "neoclassical":
        g = structural.get("g", 0.0)

        def drift(x, t, args, w):
            log_k = x[1]
            investment = policy(x, w)
            return jnp.stack(
                [
                    jnp.asarray(g, x.dtype),
                    investment / jnp.exp(log_k) - delta,
                ]
            )

        return drift, None, weights

    eta = structural["eta"]
    log_z_bar = structural["log_z_bar"]
    sigma_z = structural["sigma_z"]

    def drift(x, t, args, w):
        log_z, log_k = x
        investment = policy(x, w)
        return jnp.stack(
            [
                eta * (log_z_bar - log_z),
                investment / jnp.exp(log_k) - delta,
            ]
        )

    def diffusion(x, t, args, w):
        return jnp.stack([jnp.asarray(sigma_z, x.dtype), jnp.zeros((), x.dtype)])

    return drift, diffusion, weights


def growth_initial_states(batch, dtype):
    key_z, key_k = jax.random.split(jax.random.key(7))
    log_z = GROWTH_LOG_Z_VARIANCE**0.5 * jax.random.normal(key_z, (batch,), dtype)
    log_k = GROWTH_LOG_K_VARIANCE**0.5 * jax.random.normal(key_k, (batch,), dtype)
    return jnp.stack([log_z, log_k], axis=1)


def ou_drift(x, t, args, p):
    return -p * x


def mlp_drift(x, t, args, p):
    return jnp.tanh(x @ p["w1"] + p["b1"]) @ p["w2"] + p["b2"]


def additive_diffusion(x, t, args, p):
    return jnp.full_like(x, SIGMA)


def make_parameters(drift_kind, dim, dtype, key):
    if drift_kind == "ou":
        return jnp.asarray(THETA, dtype)
    keys = jax.random.split(key, 4)
    scale_1 = 1.0 / dim**0.5
    scale_2 = 1.0 / HIDDEN**0.5
    return {
        "w1": scale_1 * jax.random.normal(keys[0], (dim, HIDDEN), dtype),
        "b1": jnp.zeros((HIDDEN,), dtype),
        "w2": scale_2 * jax.random.normal(keys[1], (HIDDEN, dim), dtype),
        "b2": jnp.zeros((dim,), dtype),
    }


def chunk_counts(n):
    """Split n steps into (n_chunks, chunk) with n_chunks the largest divisor
    of n at most sqrt(n); prime n degenerates to one whole-solve chunk."""
    best = 1
    divisor = 1
    while divisor * divisor <= n:
        if n % divisor == 0:
            best = divisor
        divisor += 1
    return best, n // best


def chunked_sde_endpoint(solver, drift, diffusion, n, dt, dtype, unroll):
    n_chunks, chunk = chunk_counts(n)
    times = (jnp.arange(n, dtype=dtype) * dt).reshape(n_chunks, chunk)

    def one(x_0, p, noise):
        def g_drift(x, t):
            return drift(x, t, None, p)

        def g_diffusion(x, t):
            return diffusion(x, t, None, p)

        def inner(x, inputs):
            t, w = inputs
            return solver.step(g_drift, g_diffusion, t, x, dt, w, identity), None

        def chunk_body(x, inputs):
            x, _ = jax.lax.scan(inner, x, inputs, unroll=unroll)
            return x, None

        noise_chunks = jax.tree.map(
            lambda leaf: leaf.reshape((n_chunks, chunk) + leaf.shape[1:]), noise
        )
        x, _ = jax.lax.scan(jax.checkpoint(chunk_body), x_0, (times, noise_chunks))
        return x

    return one


def chunked_rk4_endpoint(solver, drift, n, dt, dtype, unroll):
    n_chunks, chunk = chunk_counts(n)
    times = (jnp.arange(n, dtype=dtype) * dt).reshape(n_chunks, chunk)

    def one(x_0, p):
        def g(x, t):
            return drift(x, t, None, p)

        def inner(x, t):
            x_1, _, _ = solver.step_fixed(g, t, x, dt, None, identity)
            return x_1, None

        def chunk_body(x, ts):
            x, _ = jax.lax.scan(inner, x, ts, unroll=unroll)
            return x, None

        x, _ = jax.lax.scan(jax.checkpoint(chunk_body), x_0, times)
        return x

    return one


def identity(x):
    return x


def build_case(case, dtype, remat, unroll=1):
    """Return (fn, inputs) where jit(fn)(*inputs) runs the whole ensemble."""
    save_at = SaveAt(steps=True) if case["save"] == "steps" else SaveAt(t_1=True)
    n = case["n_steps"]
    batch = case["batch"]
    dim = case["dim"]
    if case["drift"] in GROWTH_DRIFTS:
        drift, diffusion, p = load_growth_model(case["drift"], dtype)
        x_0s = growth_initial_states(batch, dtype)
    else:
        drift = ou_drift if case["drift"] == "ou" else mlp_drift
        diffusion = additive_diffusion
        p = make_parameters(case["drift"], dim, dtype, jax.random.key(0))
        x_0s = jnp.linspace(0.5, 2.0, batch * dim, dtype=dtype).reshape(batch, dim)
    solver = {"sra1": SRA1(), "em": EulerMaruyama(), "rk4": RK4()}[case["solver"]]
    dt = jnp.asarray(1.0 / n, dtype)

    if case["solver"] == "rk4":
        if remat == "chunked":
            one = chunked_rk4_endpoint(solver, drift, n, dt, dtype, unroll)
        else:

            def one(x_0, p):
                return solve_ode(
                    drift,
                    solver,
                    0.0,
                    1.0,
                    x_0,
                    p=p,
                    dt_0=1.0 / n,
                    max_steps=n,
                    save_at=save_at,
                    has_aux=False,
                    unroll=unroll,
                ).xs

            if remat == "solve":
                one = jax.checkpoint(one)

        if case["mode"] == "primal":

            def fn(x_0s, p):
                return jax.vmap(lambda x_0: one(x_0, p))(x_0s)

            return fn, (x_0s, p)

        def loss(x_0s, p):
            return jnp.sum(jax.vmap(lambda x_0: one(x_0, p))(x_0s) ** 2)

        return jax.grad(loss, argnums=(0, 1)), (x_0s, p)

    keys = jax.random.split(jax.random.key(1), batch)
    if case["noise"] == "explicit":
        noise = jax.vmap(
            lambda k: solver.sample_noise(jnp.zeros((dim,), dtype), k, n, dt, dtype)
        )(keys)
        if remat == "chunked":
            one = chunked_sde_endpoint(solver, drift, diffusion, n, dt, dtype, unroll)
        else:

            def one(x_0, p, w):
                return solve_sde(
                    drift,
                    diffusion,
                    solver,
                    0.0,
                    1.0,
                    x_0,
                    p=p,
                    noise=w,
                    n_steps=n,
                    save_at=save_at,
                    has_aux=False,
                    unroll=unroll,
                ).xs

            if remat == "solve":
                one = jax.checkpoint(one)

        if case["mode"] == "primal":

            def fn(x_0s, p, noise):
                return jax.vmap(lambda x_0, w: one(x_0, p, w))(x_0s, noise)

            return fn, (x_0s, p, noise)

        def loss(x_0s, p, noise):
            return jnp.sum(jax.vmap(lambda x_0, w: one(x_0, p, w))(x_0s, noise) ** 2)

        return jax.grad(loss, argnums=(0, 1, 2)), (x_0s, p, noise)

    def one(x_0, p, key):
        return solve_sde(
            drift,
            diffusion,
            solver,
            0.0,
            1.0,
            x_0,
            p=p,
            key=key,
            n_steps=n,
            save_at=save_at,
            has_aux=False,
            unroll=unroll,
        ).xs

    if remat == "solve":
        one = jax.checkpoint(one)

    if case["mode"] == "primal":

        def fn(x_0s, p, keys):
            return jax.vmap(lambda x_0, k: one(x_0, p, k))(x_0s, keys)

        return fn, (x_0s, p, keys)

    def loss(x_0s, p, keys):
        return jnp.sum(jax.vmap(lambda x_0, k: one(x_0, p, k))(x_0s, keys) ** 2)

    return jax.grad(loss, argnums=(0, 1)), (x_0s, p, keys)


def verify_chunked(case, dtype):
    """The chunked path bypasses the solve functions; require endpoint parity
    before timing it so a silent mismatch cannot produce wrong conclusions."""
    primal = dict(case, mode="primal")
    plain_fn, plain_inputs = build_case(primal, dtype, remat="none")
    chunk_fn, chunk_inputs = build_case(primal, dtype, remat="chunked")
    plain = jax.jit(plain_fn)(*plain_inputs)
    chunked = jax.jit(chunk_fn)(*chunk_inputs)
    tolerance = 200 * float(jnp.finfo(dtype).eps)
    if not jnp.allclose(plain, chunked, atol=tolerance, rtol=tolerance):
        gap = float(jnp.max(jnp.abs(plain - chunked)))
        raise AssertionError(f"chunked endpoint mismatch for {case}: max |diff|={gap}")


def time_case(fn, inputs, repeats):
    compiled = jax.jit(fn)

    def call():
        return jax.block_until_ready(compiled(*inputs))

    start = time.perf_counter()
    call()
    compile_seconds = time.perf_counter() - start
    single = time.perf_counter()
    call()
    single = time.perf_counter() - single
    iters = max(1, min(MAX_ITERS, int(TARGET_SECONDS_PER_REPEAT / max(single, 1e-9))))
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(iters):
            call()
        best = min(best, (time.perf_counter() - start) / iters)
    return best, compile_seconds


def case_records(args, flush):
    dtypes = {"float32": jnp.float32, "float64": jnp.float64}
    records = []
    cases = [
        dict(
            zip(
                ("solver", "drift", "batch", "dim", "n_steps", "dtype", "save", "mode"),
                values,
                strict=True,
            )
        )
        for values in itertools.product(
            args.solvers,
            args.drifts,
            args.batch,
            args.dim,
            args.n_steps,
            args.dtypes,
            args.save,
            args.modes,
        )
    ]
    for case in cases:
        if case["drift"] in GROWTH_DRIFTS:
            # Growth drifts bind to their model class: the neoclassical model
            # is the deterministic ODE, stochastic growth is the SDE, and the
            # state is fixed at (log z, log k).
            if (case["drift"] == "neoclassical") != (case["solver"] == "rk4"):
                continue
            if case["dim"] != args.dim[0]:
                continue
            case = dict(case, dim=2)
        noise_modes = ["none"] if case["solver"] == "rk4" else args.noise_modes
        for noise_mode in noise_modes:
            case = dict(case, noise=noise_mode)
            if args.remat == "chunked" and (
                case["save"] != "t1" or noise_mode == "key"
            ):
                print(f"skip {case}: chunked remat is endpoint/explicit-noise only")
                continue
            dtype = dtypes[case["dtype"]]
            try:
                if args.remat == "chunked":
                    verify_chunked(case, dtype)
                fn, inputs = build_case(case, dtype, args.remat, args.scan_unroll)
                seconds, compile_seconds = time_case(fn, inputs, args.repeats)
            except Exception as error:
                # OOM is a measurement (the memory wall), not a crash; anything
                # else still fails the run loudly.
                if "RESOURCE_EXHAUSTED" not in str(error):
                    raise
                record = dict(case, remat=args.remat, oom=True)
                records.append(record)
                flush(records)
                print(
                    f"{case['solver']:>4} {case['drift']:>3} "
                    f"B={case['batch']:<6} d={case['dim']:<3} "
                    f"n={case['n_steps']:<5} {case['dtype']} "
                    f"noise={noise_mode:<8} save={case['save']:<5} "
                    f"{case['mode']:<6}       OOM",
                    flush=True,
                )
                continue
            record = dict(
                case,
                remat=args.remat,
                oom=False,
                seconds_per_call=seconds,
                trajectories_per_second=case["batch"] / seconds,
                compile_seconds=compile_seconds,
            )
            records.append(record)
            flush(records)
            print(
                f"{case['solver']:>4} {case['drift']:>3} B={case['batch']:<6} "
                f"d={case['dim']:<3} n={case['n_steps']:<5} {case['dtype']} "
                f"noise={noise_mode:<8} save={case['save']:<5} "
                f"{case['mode']:<6} {seconds * 1e3:9.3f} ms/call "
                f"({record['trajectories_per_second']:.3g} traj/s)",
                flush=True,
            )
    return records


def write_outputs(records, args, device):
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "device": device,
        "backend": jax.default_backend(),
        "jax_version": jax.__version__,
        "matmul_precision": args.matmul_precision,
        "scan_unroll": args.scan_unroll,
        "remat": args.remat,
    }
    output.with_suffix(".json").write_text(
        json.dumps({"config": header, "records": records}, indent=2) + "\n"
    )
    lines = [
        f"# gpu_trajectories — {device} ({jax.default_backend()}, "
        f"jax {jax.__version__})",
        "",
        f"matmul_precision={args.matmul_precision}, "
        f"scan_unroll={args.scan_unroll}, remat={args.remat}",
        "",
        "| solver | drift | B | d | n_steps | dtype | noise | save | mode "
        "| ms/call | traj/s | compile s |",
        "|---|---|---:|---:|---:|---|---|---|---|---:|---:|---:|",
    ]
    for r in records:
        if r.get("oom"):
            timing = "| OOM | — | — |"
        else:
            timing = (
                f"| {r['seconds_per_call'] * 1e3:.3f} "
                f"| {r['trajectories_per_second']:.3g} "
                f"| {r['compile_seconds']:.2f} |"
            )
        lines.append(
            f"| {r['solver']} | {r['drift']} | {r['batch']} | {r['dim']} "
            f"| {r['n_steps']} | {r['dtype']} | {r['noise']} | {r['save']} "
            f"| {r['mode']} {timing}"
        )
    output.with_suffix(".md").write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    global jax, jnp, SRA1, EulerMaruyama, RK4, SaveAt, solve_ode, solve_sde
    import jax
    import jax.numpy as jnp

    if "float64" in args.dtypes:
        jax.config.update("jax_enable_x64", True)
    if args.matmul_precision is not None:
        jax.config.update("jax_default_matmul_precision", args.matmul_precision)
    from tinydiffeq import RK4, SRA1, EulerMaruyama, SaveAt, solve_ode, solve_sde

    device = jax.devices()[0].device_kind
    print(f"device: {device} ({jax.default_backend()}), jax {jax.__version__}")
    records = case_records(
        args, flush=lambda records: write_outputs(records, args, device)
    )
    write_outputs(records, args, device)
    print(f"done: {Path(args.output).with_suffix('.json')}")


if __name__ == "__main__":
    main()
