# Failed implicit-AD integration benchmark comparison

## Environment and revisions

- Date: 2026-07-21
- Host: `dh4.econ.ubc.ca`, Apple M4 Max, arm64, macOS 26.5.2
- Python: CPython 3.13.2
- JAX/backend: JAX 0.10.2 on CPU
- Baseline: tinydiffeq `7de3396c736d6809755fb3fa789a5e674eebe92d`
  with nlls-gram `bf22f01`
- Post-change: the tinydiffeq commit containing this report with nlls-gram
  `f75a7bfc0b84aab6f6008c54d30f70ce68dfdae3`
- Runtime regression gate: a slowdown greater than `max(5%, 1 us)` in a
  successful JVP/VJP case

The baseline and post-change JSON files were produced by the full 115-case
pytest-benchmark suite. Because short microbenchmarks on this host showed
substantial unrelated variation, the changed DAE/SDAE/aux paths were also
measured with 15 rounds of 100 blocked dispatches. Gate decisions use these
higher-stability measurements.

## Higher-stability runtime results

The 39-case root/aux matrix had no gate crossings. Representative AD results
are below; negative changes are improvements.

| Case | Transform | Baseline (us) | Post (us) | Change |
| --- | --- | ---: | ---: | ---: |
| DAE scalar | JVP | 31.935 | 32.046 | +0.35% |
| DAE scalar | VJP | 52.809 | 51.555 | -2.38% |
| DAE vector16 | JVP | 148.572 | 142.267 | -4.24% |
| DAE vector16 | VJP | 573.171 | 560.020 | -2.29% |
| DAE tree16 | JVP | 183.834 | 185.392 | +0.85% |
| DAE tree16 | VJP | 677.593 | 676.075 | -0.22% |
| Rodas5P DAE scalar | JVP | 44.296 | 36.226 | -18.22% |
| Rodas5P DAE scalar | VJP | 50.241 | 44.443 | -11.54% |
| SDAE scalar | JVP | 81.113 | 79.583 | -1.89% |
| SDAE scalar | VJP | 85.711 | 85.245 | -0.54% |

The full short-round JSON comparison had 11 apparent crossings: nine aux/grid
cases, scalar Euler-Maruyama primal, and tree Tsit5 primal. No DAE/SDAE root
case crossed. Repeating the nine changed-path aux/grid cases at 15 by 100
dispatches removed every JVP/VJP crossing:

| Case | Transform | Baseline (us) | Post (us) | Change | Gate |
| --- | --- | ---: | ---: | ---: | --- |
| aux 8 steps, 8 queries | primal | 55.882 | 55.019 | -1.54% | pass |
| aux 128 steps, 8 queries | primal | 462.615 | 460.960 | -0.36% | pass |
| aux 32 steps, 32 queries | primal | 129.065 | 136.682 | +5.90% | primal-only crossing |
| aux 128 steps, 128 queries | primal | 461.463 | 458.000 | -0.75% | pass |
| aux 8 steps, 128 queries | JVP | 68.364 | 70.695 | +3.41% | pass |
| no aux, auto, 8 by 8 | primal | 47.335 | 49.216 | +3.97% | pass |
| no aux, auto, 128 by 128 | VJP | 696.938 | 714.709 | +2.55% | pass |
| no aux, explicit, 8 by 8 | primal | 47.100 | 47.717 | +1.31% | pass |
| no aux, explicit, 8 steps, 128 queries | VJP | 67.086 | 67.905 | +1.22% | pass |

The isolated 32-by-32 primal crossing is 7.62 us and is not monotone in problem
size: both 128-step primal cases were slightly faster. It does not affect the
JVP/VJP gate and was not accompanied by a changed-path AD regression.

## Cold compilation

Rodas5P DAE and SDAE compile times were effectively flat. Repeated-stage Tsit5
DAE compilation added 6.98 to 28.95 ms depending on state and transform. Six
of nine DAE entries crossed 5%; the three scalar entries remained below 5%.

| DAE state | Transform | Baseline (s) | Post (s) | Change |
| --- | --- | ---: | ---: | ---: |
| scalar | primal | 0.124265 | 0.129940 | +4.57% |
| scalar | JVP | 0.209886 | 0.218616 | +4.16% |
| scalar | VJP | 0.338350 | 0.354601 | +4.80% |
| vector16 | primal | 0.133185 | 0.140160 | +5.24% |
| vector16 | JVP | 0.244740 | 0.268966 | +9.90% |
| vector16 | VJP | 0.398900 | 0.419466 | +5.16% |
| tree16 | primal | 0.155509 | 0.163628 | +5.22% |
| tree16 | JVP | 0.297332 | 0.313207 | +5.34% |
| tree16 | VJP | 0.510382 | 0.539334 | +5.67% |

The added compilation work is the inactive-lane-safe input selection required
before each nlls call. Its tangent rule was consolidated into one pytree-level
operation; this removed the successful VJP runtime regression seen in the
initial implementation. The only unrelated cold-compilation crossing was RK4
scalar primal (20.15 ms to 22.05 ms), consistent with run-to-run noise.

## JVP strategy benchmark

The ODE-only strategy benchmark is not on the modified DAE code path. VJP and
cached-pullback timings improved for every direction count. One uncached,
eight-direction JVP measurement increased by 7.45%, while its 16-direction
measurement improved by 2.09%; this is not a systematic changed-path
regression.

## Commands and artifacts

```bash
JAX_PLATFORMS=cpu uv run --group benchmark pytest benchmarks \
  --benchmark-only \
  --benchmark-json=benchmarks/results/2026-07-21_failed-ad-integration-post.json
JAX_PLATFORMS=cpu uv run --group benchmark python -m benchmarks.compile_times
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
  uv run python -m benchmarks.jvp_strategies
```

The baseline/post JSON, cold-compilation CSVs, JVP-strategy CSVs, stable matrix
CSVs, and targeted raw-crossing CSV are stored alongside this report.

## Published nlls-gram 2.4.0 verification

The release candidate was retested after replacing the local editable nlls-gram
source with the published PyPI wheel. The lockfile resolves `nlls-gram` 2.4.0
from the registry while the tinydiffeq package metadata retains the public
requirement `nlls-gram>=2.4.0`.

The full released-dependency benchmark completed all 115 cases. Its only raw
short-round crossing was the unrelated Tsit5 tree-state primal case (21.869 to
23.608 us); no JVP or VJP case crossed the gate.

The corresponding 39-case higher-stability root/aux matrix initially showed
one 5.74% sample increase for the DAE tree-state JVP (185.392 to 196.033 us).
Seven immediate repetitions of that exact 15-round by 100-dispatch measurement
ranged from 174.412 to 182.578 us, all faster than the reference. It was thus a
transient timing sample rather than a released-package regression. Every other
stable JVP/VJP case passed on the first release run.

Release verification artifacts:

- `2026-07-21_nlls-2.4.0-release.json`
- `2026-07-21_nlls-2.4.0-release-stable.csv`
- `2026-07-21_nlls-2.4.0-release-tree-jvp-repeat.csv`
