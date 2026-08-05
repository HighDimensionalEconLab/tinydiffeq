# Vulcan L40S: fixed-step SDE/ODE trajectory ensembles (2026-08-04)

Environment: NVIDIA L40S (48 GB), jax 0.9.1 + CUDA-12 plugin (Alliance
wheelhouse), Python 3.11, float32, TF32 matmul default. Workload:
`benchmarks/gpu_trajectories.py` with the **trained kernels investment
policies** (`benchmarks/policies/`, exported by `export_growth_policies.py`) —
the neoclassical growth ODE under RK4 and the stochastic growth SDE
(additive OU noise in log z) under SRA1/EulerMaruyama. Per-trajectory `x_0`
and explicit presampled `noise=`, vmapped and jitted; grad mode is
`jit(grad)` of a scalar loss with respect to `(x_0, policy weights, noise)`.
Raw records: `vulcan-l40s-*.json` in this directory.

## Headline numbers (kernels scale: B ≤ 32, `SaveAt(steps=True)`)

Milliseconds per whole-ensemble call, `unroll=1` → `unroll=8`:

| model / solver | B | n_steps | primal | grad |
|---|---:|---:|---:|---:|
| stochastic growth / SRA1 | 32 | 63 | 1.35 → 1.09 | 8.03 → 3.43 |
| stochastic growth / SRA1 | 32 | 255 | 5.14 → 4.07 | 32.0 → 13.3 |
| stochastic growth / EM | 32 | 255 | 3.07 → 2.19 | 18.7 → 7.1 |
| neoclassical / RK4 | 32 | 63 | 2.11 → 1.62 | 14.3 → 6.0 |
| neoclassical / RK4 | 32 | 255 | 8.25 → 6.13 | 57.1 → 25.1 |

Cost is set almost entirely by `n_steps` (sequential scan latency), not by
batch: times are flat in B from 1 to 32 — and in primal mode flat to
**B = 1024** (SRA1 n=255: 4.49 ms at B=16 vs 5.22 ms at B=1024). Forward
ensembles of hundreds of trajectories are effectively free; the grad-mode
per-LM-iteration cost at the kernels training shape (B=16, n=63, steps
saved, SRA1) is ~8 ms rolled and **~3.4 ms with `unroll=8`**.

## Findings

1. **`lax.scan` unroll is the one big lever: 2.0–3.1× faster grad, 1.2–1.7×
   primal** at B ≤ 32 (per-iteration dispatch overhead dominates these small
   kernels; the scan bodies are pure math — matmuls, elementwise, and their
   transposes; no branches or bookkeeping to strip). Most of the gain is at
   `unroll=4`; 8 adds a little more; compile time grows with the unroll
   factor. Promoted to a real `unroll=` argument on `solve_sde` and
   fixed-step `solve_ode`; `vulcan-l40s-g-unroll8-promoted` confirms the
   argument reproduces the measured numbers.
2. **Explicit `noise=` is free**: identical timing to in-solve `key=` draws
   in every configuration, so first-class noise costs nothing.
3. **`SaveAt(steps=True)` is free at this scale** (≤ 5% over endpoint-only),
   so the collocation rollout shape carries no memory-traffic penalty.
4. **`jax_default_matmul_precision="highest"` costs 15–35%** on these
   width-32 policy matmuls versus the TF32 default. Use it for
   reproducibility when needed, knowingly.
5. **CUDA-graph command-buffer capture of the while loop
   (`--xla_gpu_enable_command_buffer=...,WHILE`) had zero effect** — timings
   identical to baseline. Unroll is the only dispatch-overhead remedy that
   worked.
6. **The reverse-mode memory wall is real but far from kernels scale**: the
   plain-scan grad tape OOMs the 48 GB card at B=10,000 × n=1024 (SRA1,
   ~45 GiB requested; `vulcan-l40s-a-baseline` records the failure). At
   B ≤ 32 the tape is tens of MB. The benchmark's `--remat chunked` mode
   (√n nested-scan checkpoint, endpoint-parity-verified) is the demonstrated
   remedy if ensembles ever grow to B·n ≳ 10⁷ in grad mode; a package
   `checkpoint_every=` option was considered and deliberately not added.

## Follow-up: vmapped adaptive solves (CPU measurement, same session)

A batched-predicate `lax.cond` lowers under `vmap` to a both-branches
select, so vmapped **adaptive** solves used to execute every `max_steps`
attempt slot: a B=32 Tsit5 solve of the neoclassical policy model that
needs 8 attempts took 2.6 ms at `max_steps=64` but 47.5 ms at
`max_steps=1024` (CPU). Gating the skip conds on a scalar `unvmap_all`
predicate (batching rule reduces over the batch axis; the diffrax
`unvmap_any` trick) makes the vmapped primal budget-invariant: 0.5 ms at
every budget, a 95× win at `max_steps=1024`. Reverse mode improved 3.9×
(129 → 33 ms) but still scales with the budget through the scan's stacked
per-slot residuals — keep `max_steps` realistic when differentiating
adaptive solves.

Validated on the L40S (same probe, pre-gate vs gated, ms per call):

| B | max_steps | primal before → after | grad before → after |
|---:|---:|---|---|
| 32 | 64 | 2.05 → 0.87 (2.4×) | 9.0 → 5.7 (1.6×) |
| 32 | 256 | 7.82 → 1.16 (6.7×) | 36.2 → 15.4 (2.3×) |
| 32 | 1024 | 30.7 → 2.50 (12.3×) | 144.3 → 54.7 (2.6×) |
| 256 | 64 | 2.71 → 1.00 (2.7×) | — |
| 256 | 1024 | 40.9 → 2.59 (15.8×) | — |

## Reproduce

```bash
python -m benchmarks.gpu_trajectories \
  --drifts neoclassical stochastic-growth --solvers sra1 em rk4 \
  --batch 1 4 16 32 --dim 1 --n-steps 31 63 127 255 \
  --modes primal grad --save t1 steps --output <prefix>
# knobs: --scan-unroll 8 | --matmul-precision highest | --noise-modes key
```
