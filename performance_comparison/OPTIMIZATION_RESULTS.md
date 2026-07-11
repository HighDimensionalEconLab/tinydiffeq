# ODE loop optimization follow-up

Targeted post-compilation medians measured on the same local CPU and RTX 3090
as `CURRENT_RESULTS.md`. The fixed rows use 128 identical steps for the
256-state periodic diffusion problem. Adaptive rows use the common
`rtol=1e-4`, `atol=1e-6`, `dt_0=0.01`, and `max_steps=512` configuration; both
libraries accepted the same number of steps. Every timed JAX result was
synchronized, compilation was excluded, and nine samples were used.

| Backend/case | Previous tinydiffeq | Optimized tinydiffeq | Diffrax | Change |
|---|---:|---:|---:|---:|
| CPU RK4 fixed, float32 | 533.69 µs | 546.94 µs | — | 0.98× |
| CPU RK4 fixed, float64 | 661.41 µs | 569.27 µs | — | 1.16× |
| CPU Tsit5 fixed, float32 | 1,730.04 µs | 1,267.95 µs | 549.03 µs | 1.36× |
| CPU Tsit5 fixed, float64 | 1,850.71 µs | 1,368.91 µs | 668.95 µs | 1.35× |
| GPU Tsit5 adaptive/I, float32 | 9,219.59 µs | 1,265.63 µs | 970.69 µs | 7.28× |
| GPU Tsit5 adaptive/I, float64 | 10,595.13 µs | 1,313.59 µs | 975.68 µs | 8.07× |
| GPU Tsit5 adaptive/PI, float32 | 10,665.16 µs | 1,576.57 µs | 1,594.27 µs | 6.77× |
| GPU Tsit5 adaptive/PI, float64 | 12,422.36 µs | 1,653.93 µs | 1,400.59 µs | 7.51× |

Adaptive chunking removes most padded-tail GPU work while retaining bounded
reverse-mode-compatible scans. The fixed specialization removes controller
bookkeeping and lets fixed Tsit5 omit its unused embedded error estimate. RK4
is effectively unchanged in float32; its remaining CPU gap is stage-kernel
cost rather than adaptive or embedded-error overhead.
