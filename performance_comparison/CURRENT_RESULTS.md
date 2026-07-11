# Current tinydiffeq performance comparison

This snapshot reports median post-compilation endpoint latency. JAX calls synchronize
the complete result on every timed invocation; Julia calls are warmed before
`BenchmarkTools` sampling. Ratios in another library's cell are relative to
tinydiffeq only when both occupy the same row. An em dash means that combination was
not available or not measured.

The targeted [ODE loop optimization follow-up](OPTIMIZATION_RESULTS.md)
supersedes the affected fixed RK4/Tsit5 CPU and adaptive Tsit5 GPU rows below.

## Environment

| Result file | Runtime metadata |
|---|---|
| `python_cpu_rodas.json` | backend=cpu, devices=['cpu:0'], jax=0.10.2, diffrax=0.7.2, tinydiffeq_git=23fa371742eef93055269807e10d74d403a6b0d9, python=3.13.13, platform=Linux-6.17.0-35-generic-x86_64-with-glibc2.39, quick=True, x64_enabled=True |
| `python_gpu_rodas.json` | backend=gpu, devices=['cuda:0'], jax=0.10.2, diffrax=0.7.2, tinydiffeq_git=23fa371742eef93055269807e10d74d403a6b0d9, python=3.13.13, platform=Linux-6.17.0-35-generic-x86_64-with-glibc2.39, quick=True, x64_enabled=True |
| `julia_cpu_rodas.json` | backend=cpu, julia=1.12.6, packages={'StochasticDiffEq': '7.1.1', 'SciMLSensitivity': '7.113.0', 'SciMLBase': '3.31.0'}, threads=1, platform=x86_64-linux-gnu, quick=True, blas_threads=1 |
| `python_cpu_dae.json` | backend=cpu, devices=['cpu:0'], jax=0.10.2, diffrax=0.7.2, tinydiffeq_git=23fa371742eef93055269807e10d74d403a6b0d9, python=3.13.13, platform=Linux-6.17.0-35-generic-x86_64-with-glibc2.39, quick=True, x64_enabled=True |
| `python_gpu_dae.json` | backend=gpu, devices=['cuda:0'], jax=0.10.2, diffrax=0.7.2, tinydiffeq_git=23fa371742eef93055269807e10d74d403a6b0d9, python=3.13.13, platform=Linux-6.17.0-35-generic-x86_64-with-glibc2.39, quick=True, x64_enabled=True |
| `julia_cpu_dae.json` | backend=cpu, julia=1.12.6, packages={'StochasticDiffEq': '7.1.1', 'SciMLSensitivity': '7.113.0', 'SciMLBase': '3.31.0'}, threads=1, platform=x86_64-linux-gnu, quick=True, blas_threads=1 |
| `python_cpu_sde_large.json` | backend=cpu, devices=['cpu:0'], jax=0.10.2, diffrax=0.7.2, tinydiffeq_git=23fa371742eef93055269807e10d74d403a6b0d9, python=3.13.13, platform=Linux-6.17.0-35-generic-x86_64-with-glibc2.39, quick=True, x64_enabled=True |
| `python_gpu_sde_large.json` | backend=gpu, devices=['cuda:0'], jax=0.10.2, diffrax=0.7.2, tinydiffeq_git=23fa371742eef93055269807e10d74d403a6b0d9, python=3.13.13, platform=Linux-6.17.0-35-generic-x86_64-with-glibc2.39, quick=True, x64_enabled=True |
| `julia_cpu_sde_large.json` | backend=cpu, julia=1.12.6, packages={'StochasticDiffEq': '7.1.1', 'SciMLSensitivity': '7.113.0', 'SciMLBase': '3.31.0'}, threads=1, platform=x86_64-linux-gnu, quick=True, blas_threads=1 |
| `julia_cpu_sdae.json` | backend=cpu, julia=1.12.6, packages={'StochasticDiffEq': '7.1.1', 'SciMLSensitivity': '7.113.0', 'SciMLBase': '3.31.0'}, threads=1, platform=x86_64-linux-gnu, quick=True, blas_threads=1 |
| `julia_cpu_vjp.json` | backend=cpu, julia=1.12.6, packages={'StochasticDiffEq': '7.1.1', 'SciMLSensitivity': '7.113.0', 'SciMLBase': '3.31.0'}, threads=1, platform=x86_64-linux-gnu, quick=True, blas_threads=1 |
| `python_cpu.json` | backend=cpu, devices=['cpu:0'], jax=0.10.2, diffrax=0.7.2, tinydiffeq_git=23fa371742eef93055269807e10d74d403a6b0d9, python=3.13.13, platform=Linux-6.17.0-35-generic-x86_64-with-glibc2.39, quick=True, x64_enabled=True |
| `python_gpu.json` | backend=gpu, devices=['cuda:0'], jax=0.10.2, diffrax=0.7.2, tinydiffeq_git=23fa371742eef93055269807e10d74d403a6b0d9, python=3.13.13, platform=Linux-6.17.0-35-generic-x86_64-with-glibc2.39, quick=True, x64_enabled=True |
| `julia_cpu.json` | backend=cpu, julia=1.12.6, packages={'StochasticDiffEq': '7.1.1', 'SciMLSensitivity': '7.113.0', 'SciMLBase': '3.31.0'}, threads=1, platform=x86_64-linux-gnu, quick=True, blas_threads=1 |
| `julia_gpu.json` | backend=cuda, julia=1.12.6, packages={'CUDA': '6.2.1', 'DiffEqGPU': '3.15.2'}, device=CUDACore.CuDevice(0), quick=True |

The CPU runs use one Julia thread and one BLAS thread. GPU results use the local RTX
3090. Compilation and first execution are excluded from the main timing cells but
remain in the raw JSON. The Rodas5P supplemental files were measured from the current
working tree on top of the recorded base commit; future runs record the dirty-worktree
flag explicitly.

All recorded Julia call wrappers had a concrete inferred return type. The validation
pass checks deterministic endpoints and JVPs across the JAX libraries, DAE constraint
residuals plus JVP/VJP finiteness, stochastic replay, independent paths, and SDE/SDAE
pathwise derivatives:

| Backend | ODE checks | DAE checks | Stochastic checks |
|---|---:|---:|---:|
| cpu | 8 passed | 4 passed | 2 passed |
| gpu | 8 passed | 4 passed | 2 passed |

JAX VJPs differentiate the discrete solve. SciML VJP cells are shown only when
`ReverseDiffAdjoint` succeeds, so they have the same discrete-sensitivity semantics;
its default continuous-adjoint timings are deliberately excluded.

## Hyperparameter equivalence

Fixed rows use the identical interval, endpoint-only output, method, dtype, and step
count: scalar ODE 64 steps on `[0, 10]`, 256-state ODE 128 steps on `[0, 1]`, DAE
64 steps on `[0, 1]`, and SDE/SDAE ensembles 128 steps on `[0, 1]`. JAX adaptive
rows share `dt_0`, tolerance, attempt budget, safety 0.9, factor limits `[0.2, 5]`,
and max-norm scaling. SciML adaptive rows share `dt_0`, tolerance, and budget but
retain SciML's native controller.

| Dtype | Controller | Exact mesh match | Accepted steps (tiny/Diffrax) | Max mesh difference |
|---|---|---:|---:|---:|
| float32 | i | no | 6/6 | 0.0347433 |
| float32 | pi | no | 10/10 | 0.23175 |
| float64 | i | no | 6/6 | 0.0292134 |
| float64 | pi | no | 10/10 | 0.359179 |

Because the adaptive meshes do not match exactly, every adaptive result below is a
**same-tolerance native-controller comparison**, not an exact hyperparameter match.
Float32 and Float64 both use the common `rtol=1e-4`, `atol=1e-6` row; Float64 also
has a precision row at `rtol=1e-7`, `atol=1e-9`.

## CPU: exact fixed-step ODE methods

| Case | tinydiffeq | Diffrax | SciML |
|---|---:|---:|---:|
| ode scalar n=1; euler; fixed; float32; jvp; common | 18.07 µs (IQR 1.70) | 50.62 µs (IQR 9.87; 2.80×) | 2.73 µs (IQR 0.11; 0.15×) |
| ode scalar n=1; euler; fixed; float32; primal; common | 11.92 µs (IQR 0.27) | 43.47 µs (IQR 0.19; 3.65×) | 2.62 µs (IQR 0.13; 0.22×) |
| ode scalar n=1; euler; fixed; float32; vjp; common | 17.10 µs (IQR 0.70) | 217.07 µs (IQR 1.10; 12.69×) | — |
| ode scalar n=1; euler; fixed; float64; jvp; common | 14.83 µs (IQR 0.07) | 48.47 µs (IQR 0.08; 3.27×) | 2.65 µs (IQR 0.05; 0.18×) |
| ode scalar n=1; euler; fixed; float64; primal; common | 11.89 µs (IQR 0.11) | 47.48 µs (IQR 0.79; 3.99×) | 2.67 µs (IQR 0.12; 0.22×) |
| ode scalar n=1; euler; fixed; float64; vjp; common | 17.31 µs (IQR 0.08) | 222.21 µs (IQR 1.27; 12.84×) | — |
| ode scalar n=1; rk4; fixed; float32; jvp; common | 15.71 µs (IQR 1.46) | — | 2.90 µs (IQR 0.09; 0.18×) |
| ode scalar n=1; rk4; fixed; float32; primal; common | 12.08 µs (IQR 0.07) | — | 2.90 µs (IQR 0.12; 0.24×) |
| ode scalar n=1; rk4; fixed; float32; vjp; common | 17.65 µs (IQR 0.16) | — | — |
| ode scalar n=1; rk4; fixed; float64; jvp; common | 15.42 µs (IQR 0.36) | — | 3.02 µs (IQR 0.09; 0.20×) |
| ode scalar n=1; rk4; fixed; float64; primal; common | 12.59 µs (IQR 0.73) | — | 2.95 µs (IQR 0.09; 0.23×) |
| ode scalar n=1; rk4; fixed; float64; vjp; common | 18.43 µs (IQR 0.01) | — | — |
| ode scalar n=1; rodas5p; fixed; float32; jvp; common | 671.73 µs (IQR 2.61) | — | 31.18 µs (IQR 0.98; 0.05×) |
| ode scalar n=1; rodas5p; fixed; float32; primal; common | 359.94 µs (IQR 15.61) | — | 12.96 µs (IQR 0.49; 0.04×) |
| ode scalar n=1; rodas5p; fixed; float32; vjp; common | 723.30 µs (IQR 3.62) | — | — |
| ode scalar n=1; rodas5p; fixed; float64; jvp; common | 710.25 µs (IQR 2.47) | — | 32.44 µs (IQR 0.98; 0.05×) |
| ode scalar n=1; rodas5p; fixed; float64; primal; common | 359.26 µs (IQR 0.37) | — | 12.94 µs (IQR 0.14; 0.04×) |
| ode scalar n=1; rodas5p; fixed; float64; vjp; common | 746.73 µs (IQR 7.32) | — | — |
| ode scalar n=1; tsit5; fixed; float32; jvp; common | 19.26 µs (IQR 1.87) | 55.84 µs (IQR 5.12; 2.90×) | 3.19 µs (IQR 0.03; 0.17×) |
| ode scalar n=1; tsit5; fixed; float32; primal; common | 12.88 µs (IQR 0.11) | 48.83 µs (IQR 0.74; 3.79×) | 3.32 µs (IQR 0.06; 0.26×) |
| ode scalar n=1; tsit5; fixed; float32; vjp; common | 18.78 µs (IQR 0.21) | 294.93 µs (IQR 5.55; 15.71×) | — |
| ode scalar n=1; tsit5; fixed; float64; jvp; common | 15.68 µs (IQR 0.26) | 70.18 µs (IQR 0.18; 4.47×) | 3.54 µs (IQR 0.10; 0.23×) |
| ode scalar n=1; tsit5; fixed; float64; primal; common | 13.55 µs (IQR 0.66) | 48.46 µs (IQR 0.99; 3.58×) | 3.40 µs (IQR 0.08; 0.25×) |
| ode scalar n=1; tsit5; fixed; float64; vjp; common | 19.24 µs (IQR 0.34) | 316.18 µs (IQR 0.51; 16.43×) | — |
| ode vector n=256; euler; fixed; float32; jvp; common | 71.62 µs (IQR 7.70) | 132.83 µs (IQR 9.14; 1.85×) | 59.01 µs (IQR 0.25; 0.82×) |
| ode vector n=256; euler; fixed; float32; primal; common | 43.28 µs (IQR 4.04) | 102.10 µs (IQR 0.68; 2.36×) | 48.27 µs (IQR 0.26; 1.12×) |
| ode vector n=256; euler; fixed; float32; vjp; common | 129.62 µs (IQR 17.89) | 2,329.62 µs (IQR 340.95; 17.97×) | 103,406.99 µs (IQR 1,010.14; 797.77×) |
| ode vector n=256; euler; fixed; float64; jvp; common | 66.43 µs (IQR 0.16) | 140.53 µs (IQR 3.17; 2.12×) | 86.56 µs (IQR 0.37; 1.30×) |
| ode vector n=256; euler; fixed; float64; primal; common | 48.16 µs (IQR 2.41) | 106.85 µs (IQR 0.06; 2.22×) | 53.90 µs (IQR 0.26; 1.12×) |
| ode vector n=256; euler; fixed; float64; vjp; common | 119.73 µs (IQR 2.64) | 2,164.65 µs (IQR 78.88; 18.08×) | 120,670.40 µs (IQR 2,496.96; 1007.88×) |
| ode vector n=256; rk4; fixed; float32; jvp; common | 1,036.18 µs (IQR 12.48) | — | 172.79 µs (IQR 0.76; 0.17×) |
| ode vector n=256; rk4; fixed; float32; primal; common | 533.69 µs (IQR 26.01) | — | 136.96 µs (IQR 0.49; 0.26×) |
| ode vector n=256; rk4; fixed; float32; vjp; common | 1,162.39 µs (IQR 4.50) | — | 568,532.86 µs (IQR 0.00; 489.11×) |
| ode vector n=256; rk4; fixed; float64; jvp; common | 1,066.78 µs (IQR 21.83) | — | 280.35 µs (IQR 1.06; 0.26×) |
| ode vector n=256; rk4; fixed; float64; primal; common | 661.41 µs (IQR 79.01) | — | 146.24 µs (IQR 1.11; 0.22×) |
| ode vector n=256; rk4; fixed; float64; vjp; common | 1,258.73 µs (IQR 11.81) | — | 553,378.82 µs (IQR 0.00; 439.63×) |
| ode vector n=256; rodas5p; fixed; float32; jvp; common | 114,807.67 µs (IQR 637.80) | — | 119,820.26 µs (IQR 1,453.55; 1.04×) |
| ode vector n=256; rodas5p; fixed; float32; primal; common | 85,374.45 µs (IQR 15.81) | — | 47,700.09 µs (IQR 173.42; 0.56×) |
| ode vector n=256; rodas5p; fixed; float32; vjp; common | 123,717.00 µs (IQR 1,365.48) | — | — |
| ode vector n=256; rodas5p; fixed; float64; jvp; common | 180,365.17 µs (IQR 212.77) | — | 198,304.44 µs (IQR 1,313.98; 1.10×) |
| ode vector n=256; rodas5p; fixed; float64; primal; common | 117,072.51 µs (IQR 981.88) | — | 71,240.44 µs (IQR 443.50; 0.61×) |
| ode vector n=256; rodas5p; fixed; float64; vjp; common | 191,493.30 µs (IQR 2,230.55) | — | — |
| ode vector n=256; tsit5; fixed; float32; jvp; common | 4,374.27 µs (IQR 591.62) | 1,020.68 µs (IQR 91.52; 0.23×) | 283.33 µs (IQR 1.10; 0.06×) |
| ode vector n=256; tsit5; fixed; float32; primal; common | 1,730.04 µs (IQR 34.17) | 529.75 µs (IQR 6.73; 0.31×) | 202.14 µs (IQR 0.36; 0.12×) |
| ode vector n=256; tsit5; fixed; float32; vjp; common | 2,512.28 µs (IQR 55.16) | 4,803.16 µs (IQR 697.43; 1.91×) | 1,316,610.91 µs (IQR 0.00; 524.07×) |
| ode vector n=256; tsit5; fixed; float64; jvp; common | 3,674.05 µs (IQR 72.91) | 1,129.90 µs (IQR 36.94; 0.31×) | 474.38 µs (IQR 2.16; 0.13×) |
| ode vector n=256; tsit5; fixed; float64; primal; common | 1,850.71 µs (IQR 18.15) | 638.30 µs (IQR 1.12; 0.34×) | 222.64 µs (IQR 0.69; 0.12×) |
| ode vector n=256; tsit5; fixed; float64; vjp; common | 2,992.54 µs (IQR 9.75) | 5,926.09 µs (IQR 96.84; 1.98×) | 1,609,369.13 µs (IQR 0.00; 537.79×) |

## CPU: adaptive Tsit5 with native controller behavior

| Case | tinydiffeq | Diffrax | SciML |
|---|---:|---:|---:|
| ode scalar n=1; rodas5p; adaptive/i; float32; jvp; common | 122.67 µs (IQR 0.55; 5 accepted) | — | — |
| ode scalar n=1; rodas5p; adaptive/i; float32; primal; common | 84.14 µs (IQR 0.85; 5 accepted) | — | — |
| ode scalar n=1; rodas5p; adaptive/i; float32; vjp; common | 1,691.66 µs (IQR 10.14; 5 accepted) | — | — |
| ode scalar n=1; rodas5p; adaptive/i; float64; jvp; common | 126.37 µs (IQR 1.78; 5 accepted) | — | — |
| ode scalar n=1; rodas5p; adaptive/i; float64; jvp; precision | 186.68 µs (IQR 0.33; 11 accepted) | — | — |
| ode scalar n=1; rodas5p; adaptive/i; float64; primal; common | 88.69 µs (IQR 0.12; 5 accepted) | — | — |
| ode scalar n=1; rodas5p; adaptive/i; float64; primal; precision | 124.36 µs (IQR 2.13; 11 accepted) | — | — |
| ode scalar n=1; rodas5p; adaptive/i; float64; vjp; common | 1,743.44 µs (IQR 39.75; 5 accepted) | — | — |
| ode scalar n=1; rodas5p; adaptive/i; float64; vjp; precision | 1,855.33 µs (IQR 26.43; 11 accepted) | — | — |
| ode scalar n=1; rodas5p; adaptive/native; float32; jvp; common | — | — | 6.76 µs (IQR 0.34) |
| ode scalar n=1; rodas5p; adaptive/native; float32; primal; common | — | — | 3.43 µs (IQR 0.95) |
| ode scalar n=1; rodas5p; adaptive/native; float64; jvp; common | — | — | 5.17 µs (IQR 0.05) |
| ode scalar n=1; rodas5p; adaptive/native; float64; jvp; precision | — | — | 10.49 µs (IQR 0.50) |
| ode scalar n=1; rodas5p; adaptive/native; float64; primal; common | — | — | 2.41 µs (IQR 0.02) |
| ode scalar n=1; rodas5p; adaptive/native; float64; primal; precision | — | — | 4.69 µs (IQR 0.03) |
| ode scalar n=1; tsit5; adaptive/i; float32; jvp; common | 20.72 µs (IQR 1.18; 6 accepted) | 56.58 µs (IQR 8.48; 2.73×; 6 accepted/0 rejected) | — |
| ode scalar n=1; tsit5; adaptive/i; float32; primal; common | 16.36 µs (IQR 0.10; 6 accepted) | 47.93 µs (IQR 1.86; 2.93×; 6 accepted/0 rejected) | — |
| ode scalar n=1; tsit5; adaptive/i; float32; vjp; common | 27.71 µs (IQR 1.19; 6 accepted) | 90.88 µs (IQR 12.79; 3.28×; 6 accepted/0 rejected) | — |
| ode scalar n=1; tsit5; adaptive/i; float64; jvp; common | 20.64 µs (IQR 0.32; 6 accepted) | 54.26 µs (IQR 0.27; 2.63×; 6 accepted/0 rejected) | — |
| ode scalar n=1; tsit5; adaptive/i; float64; jvp; precision | 20.69 µs (IQR 0.36; 14 accepted) | 57.75 µs (IQR 0.25; 2.79×; 14 accepted/0 rejected) | — |
| ode scalar n=1; tsit5; adaptive/i; float64; primal; common | 16.57 µs (IQR 2.10; 6 accepted) | 52.67 µs (IQR 0.86; 3.18×; 6 accepted/0 rejected) | — |
| ode scalar n=1; tsit5; adaptive/i; float64; primal; precision | 17.21 µs (IQR 0.09; 14 accepted) | 55.04 µs (IQR 2.11; 3.20×; 14 accepted/0 rejected) | — |
| ode scalar n=1; tsit5; adaptive/i; float64; vjp; common | 28.87 µs (IQR 0.21; 6 accepted) | 93.35 µs (IQR 0.22; 3.23×; 6 accepted/0 rejected) | — |
| ode scalar n=1; tsit5; adaptive/i; float64; vjp; precision | 29.40 µs (IQR 0.48; 14 accepted) | 120.39 µs (IQR 0.15; 4.10×; 14 accepted/0 rejected) | — |
| ode scalar n=1; tsit5; adaptive/native; float32; jvp; common | — | — | 1.37 µs (IQR 0.15) |
| ode scalar n=1; tsit5; adaptive/native; float32; primal; common | — | — | 1.16 µs (IQR 0.10) |
| ode scalar n=1; tsit5; adaptive/native; float64; jvp; common | — | — | 1.37 µs (IQR 0.05) |
| ode scalar n=1; tsit5; adaptive/native; float64; jvp; precision | — | — | 2.46 µs (IQR 0.06) |
| ode scalar n=1; tsit5; adaptive/native; float64; primal; common | — | — | 1.25 µs (IQR 0.09) |
| ode scalar n=1; tsit5; adaptive/native; float64; primal; precision | — | — | 2.16 µs (IQR 0.05) |
| ode scalar n=1; tsit5; adaptive/pi; float32; jvp; common | 22.36 µs (IQR 0.21; 10 accepted) | 61.59 µs (IQR 5.59; 2.75×; 10 accepted/0 rejected) | — |
| ode scalar n=1; tsit5; adaptive/pi; float32; primal; common | 17.29 µs (IQR 0.14; 10 accepted) | 52.67 µs (IQR 3.18; 3.05×; 10 accepted/0 rejected) | — |
| ode scalar n=1; tsit5; adaptive/pi; float32; vjp; common | 33.28 µs (IQR 3.32; 10 accepted) | 111.11 µs (IQR 2.14; 3.34×; 10 accepted/0 rejected) | — |
| ode scalar n=1; tsit5; adaptive/pi; float64; jvp; common | 21.59 µs (IQR 0.06; 10 accepted) | 56.96 µs (IQR 0.02; 2.64×; 10 accepted/0 rejected) | — |
| ode scalar n=1; tsit5; adaptive/pi; float64; jvp; precision | 22.39 µs (IQR 0.14; 20 accepted) | 61.58 µs (IQR 0.05; 2.75×; 19 accepted/0 rejected) | — |
| ode scalar n=1; tsit5; adaptive/pi; float64; primal; common | 18.38 µs (IQR 1.21; 10 accepted) | 72.56 µs (IQR 7.64; 3.95×; 10 accepted/0 rejected) | — |
| ode scalar n=1; tsit5; adaptive/pi; float64; primal; precision | 19.11 µs (IQR 3.25; 20 accepted) | 59.73 µs (IQR 0.09; 3.13×; 19 accepted/0 rejected) | — |
| ode scalar n=1; tsit5; adaptive/pi; float64; vjp; common | 30.29 µs (IQR 0.53; 10 accepted) | 108.42 µs (IQR 0.14; 3.58×; 10 accepted/0 rejected) | — |
| ode scalar n=1; tsit5; adaptive/pi; float64; vjp; precision | 30.59 µs (IQR 0.31; 20 accepted) | 141.52 µs (IQR 1.72; 4.63×; 19 accepted/0 rejected) | — |
| ode vector n=256; rodas5p; adaptive/i; float32; jvp; common | 4,048.37 µs (IQR 10.54; 4 accepted) | — | — |
| ode vector n=256; rodas5p; adaptive/i; float32; primal; common | 2,760.43 µs (IQR 17.97; 4 accepted) | — | — |
| ode vector n=256; rodas5p; adaptive/i; float32; vjp; common | 57,778.35 µs (IQR 3,210.04; 4 accepted) | — | — |
| ode vector n=256; rodas5p; adaptive/i; float64; jvp; common | 6,016.51 µs (IQR 239.31; 4 accepted) | — | — |
| ode vector n=256; rodas5p; adaptive/i; float64; jvp; precision | 6,298.16 µs (IQR 118.04; 4 accepted) | — | — |
| ode vector n=256; rodas5p; adaptive/i; float64; primal; common | 3,825.78 µs (IQR 19.73; 4 accepted) | — | — |
| ode vector n=256; rodas5p; adaptive/i; float64; primal; precision | 3,538.15 µs (IQR 6.65; 4 accepted) | — | — |
| ode vector n=256; rodas5p; adaptive/i; float64; vjp; common | 102,696.40 µs (IQR 1,279.93; 4 accepted) | — | — |
| ode vector n=256; rodas5p; adaptive/i; float64; vjp; precision | 98,418.87 µs (IQR 1,632.61; 4 accepted) | — | — |
| ode vector n=256; rodas5p; adaptive/native; float32; jvp; common | — | — | 9,304.54 µs (IQR 83.82) |
| ode vector n=256; rodas5p; adaptive/native; float32; primal; common | — | — | 3,891.25 µs (IQR 39.97) |
| ode vector n=256; rodas5p; adaptive/native; float64; jvp; common | — | — | 6,453.05 µs (IQR 43.44) |
| ode vector n=256; rodas5p; adaptive/native; float64; jvp; precision | — | — | 9,500.73 µs (IQR 41.00) |
| ode vector n=256; rodas5p; adaptive/native; float64; primal; common | — | — | 1,978.28 µs (IQR 18.06) |
| ode vector n=256; rodas5p; adaptive/native; float64; primal; precision | — | — | 3,138.18 µs (IQR 22.19) |
| ode vector n=256; tsit5; adaptive/i; float32; jvp; common | 344.17 µs (IQR 5.91; 4 accepted) | 96.81 µs (IQR 17.85; 0.28×; 4 accepted/0 rejected) | — |
| ode vector n=256; tsit5; adaptive/i; float32; primal; common | 96.74 µs (IQR 8.65; 4 accepted) | 74.39 µs (IQR 0.17; 0.77×; 4 accepted/0 rejected) | — |
| ode vector n=256; tsit5; adaptive/i; float32; vjp; common | 644.71 µs (IQR 9.42; 4 accepted) | 274.50 µs (IQR 38.31; 0.43×; 4 accepted/0 rejected) | — |
| ode vector n=256; tsit5; adaptive/i; float64; jvp; common | 344.93 µs (IQR 4.17; 4 accepted) | 98.84 µs (IQR 1.64; 0.29×; 4 accepted/0 rejected) | — |
| ode vector n=256; tsit5; adaptive/i; float64; jvp; precision | 341.25 µs (IQR 0.96; 4 accepted) | 98.58 µs (IQR 0.13; 0.29×; 4 accepted/0 rejected) | — |
| ode vector n=256; tsit5; adaptive/i; float64; primal; common | 117.44 µs (IQR 1.17; 4 accepted) | 77.30 µs (IQR 0.44; 0.66×; 4 accepted/0 rejected) | — |
| ode vector n=256; tsit5; adaptive/i; float64; primal; precision | 125.90 µs (IQR 5.36; 4 accepted) | 79.53 µs (IQR 0.39; 0.63×; 4 accepted/0 rejected) | — |
| ode vector n=256; tsit5; adaptive/i; float64; vjp; common | 708.64 µs (IQR 0.43; 4 accepted) | 254.72 µs (IQR 3.40; 0.36×; 4 accepted/0 rejected) | — |
| ode vector n=256; tsit5; adaptive/i; float64; vjp; precision | 699.78 µs (IQR 7.68; 4 accepted) | 249.95 µs (IQR 3.41; 0.36×; 4 accepted/0 rejected) | — |
| ode vector n=256; tsit5; adaptive/native; float32; jvp; common | — | — | 20.48 µs (IQR 0.36) |
| ode vector n=256; tsit5; adaptive/native; float32; primal; common | — | — | 14.81 µs (IQR 1.05) |
| ode vector n=256; tsit5; adaptive/native; float32; vjp; common | — | — | 142,204.50 µs (IQR 4,356.71) |
| ode vector n=256; tsit5; adaptive/native; float64; jvp; common | — | — | 20.65 µs (IQR 0.28) |
| ode vector n=256; tsit5; adaptive/native; float64; jvp; precision | — | — | 35.11 µs (IQR 0.57) |
| ode vector n=256; tsit5; adaptive/native; float64; primal; common | — | — | 8.89 µs (IQR 0.13) |
| ode vector n=256; tsit5; adaptive/native; float64; primal; precision | — | — | 19.54 µs (IQR 1.58) |
| ode vector n=256; tsit5; adaptive/native; float64; vjp; common | — | — | 86,478.58 µs (IQR 508.76) |
| ode vector n=256; tsit5; adaptive/native; float64; vjp; precision | — | — | 132,758.43 µs (IQR 4,464.54) |
| ode vector n=256; tsit5; adaptive/pi; float32; jvp; common | 435.97 µs (IQR 52.88; 7 accepted) | 109.90 µs (IQR 0.18; 0.25×; 7 accepted/0 rejected) | — |
| ode vector n=256; tsit5; adaptive/pi; float32; primal; common | 215.57 µs (IQR 3.22; 7 accepted) | 88.51 µs (IQR 0.51; 0.41×; 7 accepted/0 rejected) | — |
| ode vector n=256; tsit5; adaptive/pi; float32; vjp; common | 1,022.03 µs (IQR 223.61; 7 accepted) | 354.94 µs (IQR 27.86; 0.35×; 7 accepted/0 rejected) | — |
| ode vector n=256; tsit5; adaptive/pi; float64; jvp; common | 439.54 µs (IQR 2.16; 6 accepted) | 124.81 µs (IQR 4.52; 0.28×; 6 accepted/0 rejected) | — |
| ode vector n=256; tsit5; adaptive/pi; float64; jvp; precision | 471.51 µs (IQR 1.50; 8 accepted) | 137.16 µs (IQR 2.08; 0.29×; 8 accepted/0 rejected) | — |
| ode vector n=256; tsit5; adaptive/pi; float64; primal; common | 237.78 µs (IQR 0.89; 6 accepted) | 93.40 µs (IQR 0.18; 0.39×; 6 accepted/0 rejected) | — |
| ode vector n=256; tsit5; adaptive/pi; float64; primal; precision | 246.88 µs (IQR 1.49; 8 accepted) | 108.12 µs (IQR 0.15; 0.44×; 8 accepted/0 rejected) | — |
| ode vector n=256; tsit5; adaptive/pi; float64; vjp; common | 789.74 µs (IQR 1.38; 6 accepted) | 339.45 µs (IQR 10.10; 0.43×; 6 accepted/0 rejected) | — |
| ode vector n=256; tsit5; adaptive/pi; float64; vjp; precision | 807.01 µs (IQR 19.33; 8 accepted) | 400.18 µs (IQR 10.46; 0.50×; 8 accepted/0 rejected) | — |

## CPU: DAE exact and capability rows

Fixed `Rodas5P` rows compare the same published method, equations, interval, dtype,
step count, and endpoint-only output against SciML's original
[`OrdinaryDiffEqRosenbrock`](https://github.com/SciML/OrdinaryDiffEq.jl/tree/master/lib/OrdinaryDiffEqRosenbrock)
implementation. Tsit5/RK4 rows use tinydiffeq's nonlinear root-restoring formulation;
adaptive Rodas5P rows use each implementation's native controller and are
same-tolerance rather than same-mesh comparisons. The DAE initial values are already
consistent; tinydiffeq still performs its documented LM consistency solve, while
SciML uses its native consistent-initialization policy.

| Case | tinydiffeq | Diffrax | SciML |
|---|---:|---:|---:|
| dae scalar n=1; rk4; fixed; float32; jvp; common | 855.81 µs (IQR 60.74) | — | — |
| dae scalar n=1; rk4; fixed; float32; primal; common | 527.09 µs (IQR 3.61) | — | — |
| dae scalar n=1; rk4; fixed; float32; vjp; common | 969.83 µs (IQR 3.02) | — | — |
| dae scalar n=1; rk4; fixed; float64; jvp; common | 1,195.30 µs (IQR 5.98) | — | — |
| dae scalar n=1; rk4; fixed; float64; primal; common | 857.64 µs (IQR 5.87) | — | — |
| dae scalar n=1; rk4; fixed; float64; vjp; common | 1,256.02 µs (IQR 12.17) | — | — |
| dae scalar n=1; rodas5p; adaptive/i; float32; jvp; common | 101.27 µs (IQR 0.55; 3 accepted) | — | — |
| dae scalar n=1; rodas5p; adaptive/i; float32; primal; common | 68.93 µs (IQR 0.97; 3 accepted) | — | — |
| dae scalar n=1; rodas5p; adaptive/i; float32; vjp; common | 2,225.01 µs (IQR 33.73; 3 accepted) | — | — |
| dae scalar n=1; rodas5p; adaptive/i; float64; jvp; common | 101.45 µs (IQR 1.10; 3 accepted) | — | — |
| dae scalar n=1; rodas5p; adaptive/i; float64; jvp; precision | 100.67 µs (IQR 0.74; 3 accepted) | — | — |
| dae scalar n=1; rodas5p; adaptive/i; float64; primal; common | 69.11 µs (IQR 0.26; 3 accepted) | — | — |
| dae scalar n=1; rodas5p; adaptive/i; float64; primal; precision | 70.30 µs (IQR 1.24; 3 accepted) | — | — |
| dae scalar n=1; rodas5p; adaptive/i; float64; vjp; common | 2,251.56 µs (IQR 55.27; 3 accepted) | — | — |
| dae scalar n=1; rodas5p; adaptive/i; float64; vjp; precision | 2,302.24 µs (IQR 88.19; 3 accepted) | — | — |
| dae scalar n=1; rodas5p; adaptive/native; float32; jvp; common | — | — | 46.20 µs (IQR 4.30) |
| dae scalar n=1; rodas5p; adaptive/native; float32; primal; common | — | — | 12.18 µs (IQR 0.43) |
| dae scalar n=1; rodas5p; adaptive/native; float64; jvp; common | — | — | 24.16 µs (IQR 1.50) |
| dae scalar n=1; rodas5p; adaptive/native; float64; jvp; precision | — | — | 36.04 µs (IQR 2.11) |
| dae scalar n=1; rodas5p; adaptive/native; float64; primal; common | — | — | 7.40 µs (IQR 0.11) |
| dae scalar n=1; rodas5p; adaptive/native; float64; primal; precision | — | — | 9.43 µs (IQR 0.40) |
| dae scalar n=1; rodas5p; fixed; float32; jvp; common | 829.28 µs (IQR 2.20) | — | 365.63 µs (IQR 9.91; 0.44×) |
| dae scalar n=1; rodas5p; fixed; float32; primal; common | 466.05 µs (IQR 1.74) | — | 94.02 µs (IQR 1.69; 0.20×) |
| dae scalar n=1; rodas5p; fixed; float32; vjp; common | 1,632.48 µs (IQR 33.13) | — | — |
| dae scalar n=1; rodas5p; fixed; float64; jvp; common | 846.09 µs (IQR 2.88) | — | 236.35 µs (IQR 2.92; 0.28×) |
| dae scalar n=1; rodas5p; fixed; float64; primal; common | 478.26 µs (IQR 1.96) | — | 89.90 µs (IQR 0.90; 0.19×) |
| dae scalar n=1; rodas5p; fixed; float64; vjp; common | 1,782.93 µs (IQR 26.97) | — | — |
| dae scalar n=1; tsit5; adaptive/i; float32; jvp; common | 166.63 µs (IQR 4.89; 3 accepted) | — | — |
| dae scalar n=1; tsit5; adaptive/i; float32; primal; common | 107.41 µs (IQR 1.63; 3 accepted) | — | — |
| dae scalar n=1; tsit5; adaptive/i; float32; vjp; common | 2,243.46 µs (IQR 190.41; 3 accepted) | — | — |
| dae scalar n=1; tsit5; adaptive/i; float64; jvp; common | 166.71 µs (IQR 0.78; 3 accepted) | — | — |
| dae scalar n=1; tsit5; adaptive/i; float64; jvp; precision | 177.16 µs (IQR 1.68; 3 accepted) | — | — |
| dae scalar n=1; tsit5; adaptive/i; float64; primal; common | 130.36 µs (IQR 1.26; 3 accepted) | — | — |
| dae scalar n=1; tsit5; adaptive/i; float64; primal; precision | 132.91 µs (IQR 2.08; 3 accepted) | — | — |
| dae scalar n=1; tsit5; adaptive/i; float64; vjp; common | 1,822.46 µs (IQR 38.89; 3 accepted) | — | — |
| dae scalar n=1; tsit5; adaptive/i; float64; vjp; precision | 1,835.94 µs (IQR 20.13; 3 accepted) | — | — |
| dae vector n=32; rk4; fixed; float32; jvp; common | 10,048.65 µs (IQR 66.84) | — | — |
| dae vector n=32; rk4; fixed; float32; primal; common | 9,371.44 µs (IQR 2,481.82) | — | — |
| dae vector n=32; rk4; fixed; float32; vjp; common | 11,505.88 µs (IQR 41.62) | — | — |
| dae vector n=32; rk4; fixed; float64; jvp; common | 17,565.39 µs (IQR 39.42) | — | — |
| dae vector n=32; rk4; fixed; float64; primal; common | 14,238.64 µs (IQR 37.08) | — | — |
| dae vector n=32; rk4; fixed; float64; vjp; common | 19,857.01 µs (IQR 76.57) | — | — |
| dae vector n=32; rodas5p; adaptive/i; float32; jvp; common | 369.76 µs (IQR 5.39; 3 accepted) | — | — |
| dae vector n=32; rodas5p; adaptive/i; float32; primal; common | 290.41 µs (IQR 32.07; 3 accepted) | — | — |
| dae vector n=32; rodas5p; adaptive/i; float32; vjp; common | 4,236.20 µs (IQR 13.21; 3 accepted) | — | — |
| dae vector n=32; rodas5p; adaptive/i; float64; jvp; common | 486.09 µs (IQR 8.80; 3 accepted) | — | — |
| dae vector n=32; rodas5p; adaptive/i; float64; jvp; precision | 473.15 µs (IQR 1.10; 3 accepted) | — | — |
| dae vector n=32; rodas5p; adaptive/i; float64; primal; common | 253.87 µs (IQR 2.24; 3 accepted) | — | — |
| dae vector n=32; rodas5p; adaptive/i; float64; primal; precision | 254.58 µs (IQR 2.12; 3 accepted) | — | — |
| dae vector n=32; rodas5p; adaptive/i; float64; vjp; common | 5,248.11 µs (IQR 38.00; 3 accepted) | — | — |
| dae vector n=32; rodas5p; adaptive/i; float64; vjp; precision | 5,209.96 µs (IQR 49.99; 3 accepted) | — | — |
| dae vector n=32; rodas5p; adaptive/native; float32; jvp; common | — | — | 435.38 µs (IQR 8.35) |
| dae vector n=32; rodas5p; adaptive/native; float32; primal; common | — | — | 281.94 µs (IQR 2.55) |
| dae vector n=32; rodas5p; adaptive/native; float64; jvp; common | — | — | 370.19 µs (IQR 2.63) |
| dae vector n=32; rodas5p; adaptive/native; float64; jvp; precision | — | — | 556.64 µs (IQR 4.62) |
| dae vector n=32; rodas5p; adaptive/native; float64; primal; common | — | — | 163.35 µs (IQR 2.60) |
| dae vector n=32; rodas5p; adaptive/native; float64; primal; precision | — | — | 230.88 µs (IQR 3.61) |
| dae vector n=32; rodas5p; fixed; float32; jvp; common | 4,550.44 µs (IQR 34.00) | — | 4,100.54 µs (IQR 55.12; 0.90×) |
| dae vector n=32; rodas5p; fixed; float32; primal; common | 2,900.43 µs (IQR 454.16) | — | 2,491.09 µs (IQR 5.96; 0.86×) |
| dae vector n=32; rodas5p; fixed; float32; vjp; common | 6,050.11 µs (IQR 152.79) | — | — |
| dae vector n=32; rodas5p; fixed; float64; jvp; common | 6,344.04 µs (IQR 101.61) | — | 5,983.95 µs (IQR 118.04; 0.94×) |
| dae vector n=32; rodas5p; fixed; float64; primal; common | 3,909.77 µs (IQR 19.68) | — | 2,885.63 µs (IQR 67.08; 0.74×) |
| dae vector n=32; rodas5p; fixed; float64; vjp; common | 8,387.97 µs (IQR 132.92) | — | — |
| dae vector n=32; tsit5; adaptive/i; float32; jvp; common | 1,220.17 µs (IQR 6.04; 3 accepted) | — | — |
| dae vector n=32; tsit5; adaptive/i; float32; primal; common | 952.40 µs (IQR 14.32; 3 accepted) | — | — |
| dae vector n=32; tsit5; adaptive/i; float32; vjp; common | 4,864.83 µs (IQR 105.11; 3 accepted) | — | — |
| dae vector n=32; tsit5; adaptive/i; float64; jvp; common | 1,867.25 µs (IQR 13.00; 3 accepted) | — | — |
| dae vector n=32; tsit5; adaptive/i; float64; jvp; precision | 1,818.45 µs (IQR 3.26; 3 accepted) | — | — |
| dae vector n=32; tsit5; adaptive/i; float64; primal; common | 1,561.48 µs (IQR 4.51; 3 accepted) | — | — |
| dae vector n=32; tsit5; adaptive/i; float64; primal; precision | 1,557.98 µs (IQR 4.16; 3 accepted) | — | — |
| dae vector n=32; tsit5; adaptive/i; float64; vjp; common | 6,221.54 µs (IQR 120.06; 3 accepted) | — | — |
| dae vector n=32; tsit5; adaptive/i; float64; vjp; precision | 5,949.32 µs (IQR 40.71; 3 accepted) | — | — |

## CPU: SDE and SDAE ensembles

JAX ensemble rows compile a `vmap` over independent keys. SciML CPU uses
`EnsembleSerial` to keep the matched resource count at one thread.

| Case | tinydiffeq | Diffrax | SciML |
|---|---:|---:|---:|
| sdae ensemble n=1 B=256; em; fixed; float32; jvp; common | 23,249.95 µs (IQR 4,093.08) | — | — |
| sdae ensemble n=1 B=256; em; fixed; float32; primal; common | 13,381.57 µs (IQR 3,765.79) | — | — |
| sdae ensemble n=1 B=256; em; fixed; float32; vjp; common | 19,968.65 µs (IQR 2,247.74) | — | — |
| sdae ensemble n=1 B=256; em; fixed; float64; jvp; common | 24,574.29 µs (IQR 22.36) | — | — |
| sdae ensemble n=1 B=256; em; fixed; float64; primal; common | 18,384.19 µs (IQR 4.35) | — | — |
| sdae ensemble n=1 B=256; em; fixed; float64; vjp; common | 24,763.51 µs (IQR 142.17) | — | — |
| sdae ensemble n=1 B=256; implicit_em; fixed/native; float32; primal; common | — | — | 18,950.75 µs (IQR 501.83) |
| sdae ensemble n=1 B=256; implicit_em; fixed/native; float64; primal; common | — | — | 18,247.62 µs (IQR 57.87) |
| sde ensemble n=1 B=1; em; fixed; float32; jvp; common | 15.19 µs (IQR 0.07) | 290.10 µs (IQR 6.82; 19.09×) | — |
| sde ensemble n=1 B=256; em; fixed; float32; jvp; common | 249.04 µs (IQR 3.04) | 689.65 µs (IQR 4.61; 2.77×) | — |
| sde ensemble n=1 B=16384; em; fixed; float32; jvp; common | 14,388.34 µs (IQR 79.33) | 26,255.53 µs (IQR 1,350.83; 1.82×) | — |
| sde ensemble n=1 B=1; em; fixed; float32; primal; common | 13.23 µs (IQR 0.07) | 278.27 µs (IQR 1.07; 21.03×) | — |
| sde ensemble n=1 B=256; em; fixed; float32; primal; common | 267.76 µs (IQR 21.72) | 629.57 µs (IQR 13.74; 2.35×) | 2,654.90 µs (IQR 3.19; 9.92×) |
| sde ensemble n=1 B=16384; em; fixed; float32; primal; common | 13,981.31 µs (IQR 63.49) | 20,246.44 µs (IQR 769.76; 1.45×) | 209,545.84 µs (IQR 11,020.10; 14.99×) |
| sde ensemble n=1 B=1; em; fixed; float32; vjp; common | 17.73 µs (IQR 0.16) | 678.49 µs (IQR 4.01; 38.26×) | — |
| sde ensemble n=1 B=256; em; fixed; float32; vjp; common | 259.89 µs (IQR 3.52) | 1,211.45 µs (IQR 105.61; 4.66×) | — |
| sde ensemble n=1 B=16384; em; fixed; float32; vjp; common | 14,313.89 µs (IQR 63.08) | 27,740.40 µs (IQR 724.51; 1.94×) | — |
| sde ensemble n=1 B=1; em; fixed; float64; jvp; common | 27.31 µs (IQR 0.98) | 292.50 µs (IQR 6.92; 10.71×) | — |
| sde ensemble n=1 B=256; em; fixed; float64; jvp; common | 293.76 µs (IQR 6.77) | 716.38 µs (IQR 2.70; 2.44×) | — |
| sde ensemble n=1 B=16384; em; fixed; float64; jvp; common | 19,297.63 µs (IQR 208.11) | 25,494.55 µs (IQR 904.79; 1.32×) | — |
| sde ensemble n=1 B=1; em; fixed; float64; primal; common | 15.35 µs (IQR 0.22) | 279.87 µs (IQR 0.33; 18.24×) | — |
| sde ensemble n=1 B=256; em; fixed; float64; primal; common | 289.01 µs (IQR 4.19) | 652.72 µs (IQR 1.70; 2.26×) | 2,768.10 µs (IQR 3.28; 9.58×) |
| sde ensemble n=1 B=16384; em; fixed; float64; primal; common | 20,294.74 µs (IQR 444.17) | 25,549.61 µs (IQR 283.27; 1.26×) | 201,151.64 µs (IQR 0.00; 9.91×) |
| sde ensemble n=1 B=1; em; fixed; float64; vjp; common | 27.45 µs (IQR 3.61) | 754.30 µs (IQR 24.46; 27.48×) | — |
| sde ensemble n=1 B=256; em; fixed; float64; vjp; common | 295.62 µs (IQR 1.24) | 1,195.43 µs (IQR 10.37; 4.04×) | — |
| sde ensemble n=1 B=16384; em; fixed; float64; vjp; common | 19,948.51 µs (IQR 372.27) | 31,536.28 µs (IQR 880.38; 1.58×) | — |

## GPU results

| Case | tinydiffeq | Diffrax | SciML |
|---|---:|---:|---:|
| dae scalar n=1; rk4; fixed; float32; jvp; common | 46,001.50 µs (IQR 29.04) | — | — |
| dae scalar n=1; rk4; fixed; float32; primal; common | 36,404.42 µs (IQR 133.87) | — | — |
| dae scalar n=1; rk4; fixed; float32; vjp; common | 61,221.52 µs (IQR 481.69) | — | — |
| dae scalar n=1; rk4; fixed; float64; jvp; common | 69,863.89 µs (IQR 255.22) | — | — |
| dae scalar n=1; rk4; fixed; float64; primal; common | 57,403.32 µs (IQR 382.36) | — | — |
| dae scalar n=1; rk4; fixed; float64; vjp; common | 85,803.37 µs (IQR 206.66) | — | — |
| dae scalar n=1; rodas5p; adaptive/i; float32; jvp; common | 9,617.38 µs (IQR 16.17; 3 accepted) | — | — |
| dae scalar n=1; rodas5p; adaptive/i; float32; primal; common | 7,209.87 µs (IQR 26.27; 3 accepted) | — | — |
| dae scalar n=1; rodas5p; adaptive/i; float32; vjp; common | 53,694.49 µs (IQR 75.78; 3 accepted) | — | — |
| dae scalar n=1; rodas5p; adaptive/i; float64; jvp; common | 8,653.06 µs (IQR 11.92; 3 accepted) | — | — |
| dae scalar n=1; rodas5p; adaptive/i; float64; jvp; precision | 8,644.52 µs (IQR 5.81; 3 accepted) | — | — |
| dae scalar n=1; rodas5p; adaptive/i; float64; primal; common | 8,416.87 µs (IQR 52.79; 3 accepted) | — | — |
| dae scalar n=1; rodas5p; adaptive/i; float64; primal; precision | 8,404.96 µs (IQR 25.10; 3 accepted) | — | — |
| dae scalar n=1; rodas5p; adaptive/i; float64; vjp; common | 54,441.08 µs (IQR 29.22; 3 accepted) | — | — |
| dae scalar n=1; rodas5p; adaptive/i; float64; vjp; precision | 54,953.60 µs (IQR 82.52; 3 accepted) | — | — |
| dae scalar n=1; rodas5p; fixed; float32; jvp; common | 21,076.68 µs (IQR 25.19) | — | — |
| dae scalar n=1; rodas5p; fixed; float32; primal; common | 12,554.13 µs (IQR 23.68) | — | — |
| dae scalar n=1; rodas5p; fixed; float32; vjp; common | 36,142.51 µs (IQR 10.31) | — | — |
| dae scalar n=1; rodas5p; fixed; float64; jvp; common | 24,714.78 µs (IQR 16.25) | — | — |
| dae scalar n=1; rodas5p; fixed; float64; primal; common | 14,875.29 µs (IQR 239.10) | — | — |
| dae scalar n=1; rodas5p; fixed; float64; vjp; common | 43,837.51 µs (IQR 229.01) | — | — |
| dae scalar n=1; tsit5; adaptive/i; float32; jvp; common | 11,733.87 µs (IQR 18.53; 3 accepted) | — | — |
| dae scalar n=1; tsit5; adaptive/i; float32; primal; common | 10,961.63 µs (IQR 34.11; 3 accepted) | — | — |
| dae scalar n=1; tsit5; adaptive/i; float32; vjp; common | 60,855.71 µs (IQR 114.54; 3 accepted) | — | — |
| dae scalar n=1; tsit5; adaptive/i; float64; jvp; common | 11,601.48 µs (IQR 25.06; 3 accepted) | — | — |
| dae scalar n=1; tsit5; adaptive/i; float64; jvp; precision | 11,622.16 µs (IQR 16.91; 3 accepted) | — | — |
| dae scalar n=1; tsit5; adaptive/i; float64; primal; common | 12,781.72 µs (IQR 14.49; 3 accepted) | — | — |
| dae scalar n=1; tsit5; adaptive/i; float64; primal; precision | 12,720.47 µs (IQR 17.93; 3 accepted) | — | — |
| dae scalar n=1; tsit5; adaptive/i; float64; vjp; common | 63,386.50 µs (IQR 290.15; 3 accepted) | — | — |
| dae scalar n=1; tsit5; adaptive/i; float64; vjp; precision | 63,405.22 µs (IQR 91.41; 3 accepted) | — | — |
| dae vector n=32; rk4; fixed; float32; jvp; common | 88,749.00 µs (IQR 222.75) | — | — |
| dae vector n=32; rk4; fixed; float32; primal; common | 66,945.33 µs (IQR 111.48) | — | — |
| dae vector n=32; rk4; fixed; float32; vjp; common | 103,305.95 µs (IQR 192.25) | — | — |
| dae vector n=32; rk4; fixed; float64; jvp; common | 357,424.08 µs (IQR 263.21) | — | — |
| dae vector n=32; rk4; fixed; float64; primal; common | 330,253.74 µs (IQR 243.65) | — | — |
| dae vector n=32; rk4; fixed; float64; vjp; common | 367,008.29 µs (IQR 126.16) | — | — |
| dae vector n=32; rodas5p; adaptive/i; float32; jvp; common | 9,660.31 µs (IQR 20.08; 4 accepted) | — | — |
| dae vector n=32; rodas5p; adaptive/i; float32; primal; common | 10,261.78 µs (IQR 247.04; 4 accepted) | — | — |
| dae vector n=32; rodas5p; adaptive/i; float32; vjp; common | 56,007.55 µs (IQR 416.26; 4 accepted) | — | — |
| dae vector n=32; rodas5p; adaptive/i; float64; jvp; common | 11,015.83 µs (IQR 30.08; 3 accepted) | — | — |
| dae vector n=32; rodas5p; adaptive/i; float64; jvp; precision | 10,976.02 µs (IQR 107.25; 3 accepted) | — | — |
| dae vector n=32; rodas5p; adaptive/i; float64; primal; common | 7,987.84 µs (IQR 18.49; 3 accepted) | — | — |
| dae vector n=32; rodas5p; adaptive/i; float64; primal; precision | 7,993.90 µs (IQR 11.43; 3 accepted) | — | — |
| dae vector n=32; rodas5p; adaptive/i; float64; vjp; common | 57,043.56 µs (IQR 263.90; 3 accepted) | — | — |
| dae vector n=32; rodas5p; adaptive/i; float64; vjp; precision | 56,827.29 µs (IQR 267.88; 3 accepted) | — | — |
| dae vector n=32; rodas5p; fixed; float32; jvp; common | 35,112.29 µs (IQR 69.01) | — | — |
| dae vector n=32; rodas5p; fixed; float32; primal; common | 21,374.94 µs (IQR 94.62) | — | — |
| dae vector n=32; rodas5p; fixed; float32; vjp; common | 44,703.24 µs (IQR 57.40) | — | — |
| dae vector n=32; rodas5p; fixed; float64; jvp; common | 47,915.31 µs (IQR 56.61) | — | — |
| dae vector n=32; rodas5p; fixed; float64; primal; common | 28,668.15 µs (IQR 555.17) | — | — |
| dae vector n=32; rodas5p; fixed; float64; vjp; common | 65,154.14 µs (IQR 170.43) | — | — |
| dae vector n=32; tsit5; adaptive/i; float32; jvp; common | 16,554.11 µs (IQR 51.67; 3 accepted) | — | — |
| dae vector n=32; tsit5; adaptive/i; float32; primal; common | 14,437.79 µs (IQR 20.21; 3 accepted) | — | — |
| dae vector n=32; tsit5; adaptive/i; float32; vjp; common | 69,302.14 µs (IQR 219.32; 3 accepted) | — | — |
| dae vector n=32; tsit5; adaptive/i; float64; jvp; common | 38,511.17 µs (IQR 48.63; 3 accepted) | — | — |
| dae vector n=32; tsit5; adaptive/i; float64; jvp; precision | 38,490.66 µs (IQR 39.65; 3 accepted) | — | — |
| dae vector n=32; tsit5; adaptive/i; float64; primal; common | 38,104.86 µs (IQR 33.44; 3 accepted) | — | — |
| dae vector n=32; tsit5; adaptive/i; float64; primal; precision | 38,105.72 µs (IQR 20.34; 3 accepted) | — | — |
| dae vector n=32; tsit5; adaptive/i; float64; vjp; common | 93,988.88 µs (IQR 66.40; 3 accepted) | — | — |
| dae vector n=32; tsit5; adaptive/i; float64; vjp; precision | 94,840.14 µs (IQR 371.28; 3 accepted) | — | — |
| ode scalar n=1; euler; fixed; float32; jvp; common | 1,679.18 µs (IQR 2.48) | 1,344.03 µs (IQR 16.53; 0.80×) | — |
| ode scalar n=1; euler; fixed; float32; primal; common | 1,248.31 µs (IQR 60.88) | 1,660.99 µs (IQR 1.88; 1.33×) | — |
| ode scalar n=1; euler; fixed; float32; vjp; common | 2,367.69 µs (IQR 19.39) | 11,598.96 µs (IQR 4.40; 4.90×) | — |
| ode scalar n=1; euler; fixed; float64; jvp; common | 1,705.68 µs (IQR 5.10) | 1,351.58 µs (IQR 0.79; 0.79×) | — |
| ode scalar n=1; euler; fixed; float64; primal; common | 1,076.21 µs (IQR 8.08) | 1,662.30 µs (IQR 16.16; 1.54×) | — |
| ode scalar n=1; euler; fixed; float64; vjp; common | 2,458.39 µs (IQR 18.62) | 11,672.90 µs (IQR 6.25; 4.75×) | — |
| ode scalar n=1; rk4; fixed; float32; jvp; common | 1,670.42 µs (IQR 5.42) | — | — |
| ode scalar n=1; rk4; fixed; float32; primal; common | 1,008.70 µs (IQR 99.80) | — | — |
| ode scalar n=1; rk4; fixed; float32; vjp; common | 2,827.33 µs (IQR 61.03) | — | — |
| ode scalar n=1; rk4; fixed; float64; jvp; common | 1,739.78 µs (IQR 2.45) | — | — |
| ode scalar n=1; rk4; fixed; float64; primal; common | 1,455.18 µs (IQR 6.32) | — | — |
| ode scalar n=1; rk4; fixed; float64; vjp; common | 3,013.56 µs (IQR 22.53) | — | — |
| ode scalar n=1; rodas5p; adaptive/i; float32; jvp; common | 13,831.09 µs (IQR 16.84; 5 accepted) | — | — |
| ode scalar n=1; rodas5p; adaptive/i; float32; primal; common | 13,073.68 µs (IQR 47.08; 5 accepted) | — | — |
| ode scalar n=1; rodas5p; adaptive/i; float32; vjp; common | 38,168.88 µs (IQR 994.78; 5 accepted) | — | — |
| ode scalar n=1; rodas5p; adaptive/i; float64; jvp; common | 12,367.82 µs (IQR 83.08; 5 accepted) | — | — |
| ode scalar n=1; rodas5p; adaptive/i; float64; jvp; precision | 14,509.24 µs (IQR 84.84; 11 accepted) | — | — |
| ode scalar n=1; rodas5p; adaptive/i; float64; primal; common | 11,532.61 µs (IQR 33.95; 5 accepted) | — | — |
| ode scalar n=1; rodas5p; adaptive/i; float64; primal; precision | 12,686.58 µs (IQR 74.63; 11 accepted) | — | — |
| ode scalar n=1; rodas5p; adaptive/i; float64; vjp; common | 38,796.99 µs (IQR 508.88; 5 accepted) | — | — |
| ode scalar n=1; rodas5p; adaptive/i; float64; vjp; precision | 41,458.34 µs (IQR 181.45; 11 accepted) | — | — |
| ode scalar n=1; rodas5p; fixed; float32; jvp; common | 21,073.76 µs (IQR 151.47) | — | — |
| ode scalar n=1; rodas5p; fixed; float32; primal; common | 11,919.05 µs (IQR 74.21) | — | — |
| ode scalar n=1; rodas5p; fixed; float32; vjp; common | 26,385.90 µs (IQR 63.10) | — | — |
| ode scalar n=1; rodas5p; fixed; float64; jvp; common | 24,137.03 µs (IQR 390.16) | — | — |
| ode scalar n=1; rodas5p; fixed; float64; primal; common | 13,267.34 µs (IQR 202.64) | — | — |
| ode scalar n=1; rodas5p; fixed; float64; vjp; common | 36,111.05 µs (IQR 152.09) | — | — |
| ode scalar n=1; tsit5; adaptive/i; float32; jvp; common | 10,694.07 µs (IQR 101.04; 6 accepted) | 1,164.14 µs (IQR 1.60; 0.11×; 6 accepted/0 rejected) | — |
| ode scalar n=1; tsit5; adaptive/i; float32; primal; common | 11,624.10 µs (IQR 50.65; 6 accepted) | 1,274.35 µs (IQR 4.74; 0.11×; 6 accepted/0 rejected) | — |
| ode scalar n=1; tsit5; adaptive/i; float32; vjp; common | 21,640.95 µs (IQR 93.40; 6 accepted) | 4,215.09 µs (IQR 16.44; 0.19×; 6 accepted/0 rejected) | — |
| ode scalar n=1; tsit5; adaptive/i; float64; jvp; common | 8,899.31 µs (IQR 143.79; 6 accepted) | 1,187.39 µs (IQR 3.33; 0.13×; 6 accepted/0 rejected) | — |
| ode scalar n=1; tsit5; adaptive/i; float64; jvp; precision | 9,237.89 µs (IQR 86.47; 14 accepted) | 2,503.76 µs (IQR 7.95; 0.27×; 14 accepted/0 rejected) | — |
| ode scalar n=1; tsit5; adaptive/i; float64; primal; common | 9,209.43 µs (IQR 44.18; 6 accepted) | 1,277.94 µs (IQR 1.69; 0.14×; 6 accepted/0 rejected) | — |
| ode scalar n=1; tsit5; adaptive/i; float64; primal; precision | 9,099.77 µs (IQR 54.21; 14 accepted) | 2,664.50 µs (IQR 1.85; 0.29×; 14 accepted/0 rejected) | — |
| ode scalar n=1; tsit5; adaptive/i; float64; vjp; common | 23,036.70 µs (IQR 136.60; 6 accepted) | 4,371.20 µs (IQR 6.78; 0.19×; 6 accepted/0 rejected) | — |
| ode scalar n=1; tsit5; adaptive/i; float64; vjp; precision | 22,968.73 µs (IQR 259.30; 14 accepted) | 9,663.15 µs (IQR 12.97; 0.42×; 14 accepted/0 rejected) | — |
| ode scalar n=1; tsit5; adaptive/pi; float32; jvp; common | 10,781.22 µs (IQR 170.58; 10 accepted) | 1,790.68 µs (IQR 4.40; 0.17×; 10 accepted/0 rejected) | — |
| ode scalar n=1; tsit5; adaptive/pi; float32; primal; common | 9,086.46 µs (IQR 14.79; 10 accepted) | 1,999.72 µs (IQR 3.09; 0.22×; 10 accepted/0 rejected) | — |
| ode scalar n=1; tsit5; adaptive/pi; float32; vjp; common | 23,060.68 µs (IQR 503.35; 10 accepted) | 6,935.51 µs (IQR 23.81; 0.30×; 10 accepted/0 rejected) | — |
| ode scalar n=1; tsit5; adaptive/pi; float64; jvp; common | 11,016.10 µs (IQR 72.34; 10 accepted) | 1,842.20 µs (IQR 2.21; 0.17×; 10 accepted/0 rejected) | — |
| ode scalar n=1; tsit5; adaptive/pi; float64; jvp; precision | 11,011.00 µs (IQR 28.79; 20 accepted) | 3,263.97 µs (IQR 7.27; 0.30×; 19 accepted/0 rejected) | — |
| ode scalar n=1; tsit5; adaptive/pi; float64; primal; common | 10,877.24 µs (IQR 50.67; 10 accepted) | 2,041.38 µs (IQR 5.67; 0.19×; 10 accepted/0 rejected) | — |
| ode scalar n=1; tsit5; adaptive/pi; float64; primal; precision | 10,807.31 µs (IQR 66.23; 20 accepted) | 3,626.83 µs (IQR 3.80; 0.34×; 19 accepted/0 rejected) | — |
| ode scalar n=1; tsit5; adaptive/pi; float64; vjp; common | 24,445.77 µs (IQR 496.42; 10 accepted) | 7,101.97 µs (IQR 15.55; 0.29×; 10 accepted/0 rejected) | — |
| ode scalar n=1; tsit5; adaptive/pi; float64; vjp; precision | 24,512.58 µs (IQR 150.23; 20 accepted) | 13,228.26 µs (IQR 97.03; 0.54×; 19 accepted/0 rejected) | — |
| ode scalar n=1; tsit5; fixed; float32; jvp; common | 2,210.21 µs (IQR 53.43) | 10,161.66 µs (IQR 24.02; 4.60×) | — |
| ode scalar n=1; tsit5; fixed; float32; primal; common | 1,569.38 µs (IQR 25.73) | 11,754.51 µs (IQR 25.94; 7.49×) | — |
| ode scalar n=1; tsit5; fixed; float32; vjp; common | 3,013.41 µs (IQR 66.74) | 56,845.47 µs (IQR 119.32; 18.86×) | — |
| ode scalar n=1; tsit5; fixed; float64; jvp; common | 2,241.48 µs (IQR 25.19) | 10,197.77 µs (IQR 18.04; 4.55×) | — |
| ode scalar n=1; tsit5; fixed; float64; primal; common | 1,932.83 µs (IQR 4.93) | 11,762.64 µs (IQR 8.27; 6.09×) | — |
| ode scalar n=1; tsit5; fixed; float64; vjp; common | 3,059.18 µs (IQR 7.35) | 57,009.48 µs (IQR 101.32; 18.64×) | — |
| ode vector n=256; euler; fixed; float32; jvp; common | 3,134.21 µs (IQR 14.87) | 2,511.73 µs (IQR 7.72; 0.80×) | — |
| ode vector n=256; euler; fixed; float32; primal; common | 2,661.16 µs (IQR 25.79) | 2,503.63 µs (IQR 11.31; 0.94×) | — |
| ode vector n=256; euler; fixed; float32; vjp; common | 4,681.73 µs (IQR 43.58) | 26,092.49 µs (IQR 14.92; 5.57×) | — |
| ode vector n=256; euler; fixed; float64; jvp; common | 3,310.58 µs (IQR 36.52) | 2,529.53 µs (IQR 6.98; 0.76×) | — |
| ode vector n=256; euler; fixed; float64; primal; common | 2,816.84 µs (IQR 27.36) | 2,504.29 µs (IQR 6.12; 0.89×) | — |
| ode vector n=256; euler; fixed; float64; vjp; common | 4,859.20 µs (IQR 47.98) | 26,145.79 µs (IQR 28.12; 5.38×) | — |
| ode vector n=256; rk4; fixed; float32; jvp; common | 3,606.02 µs (IQR 88.55) | — | — |
| ode vector n=256; rk4; fixed; float32; primal; common | 3,004.68 µs (IQR 30.02) | — | — |
| ode vector n=256; rk4; fixed; float32; vjp; common | 5,984.20 µs (IQR 311.69) | — | — |
| ode vector n=256; rk4; fixed; float64; jvp; common | 5,155.72 µs (IQR 40.88) | — | — |
| ode vector n=256; rk4; fixed; float64; primal; common | 4,148.30 µs (IQR 63.73) | — | — |
| ode vector n=256; rk4; fixed; float64; vjp; common | 7,524.92 µs (IQR 308.60) | — | — |
| ode vector n=256; rodas5p; adaptive/i; float32; jvp; common | 17,328.06 µs (IQR 41.77; 5 accepted) | — | — |
| ode vector n=256; rodas5p; adaptive/i; float32; primal; common | 15,377.81 µs (IQR 49.74; 5 accepted) | — | — |
| ode vector n=256; rodas5p; adaptive/i; float32; vjp; common | 44,942.99 µs (IQR 138.53; 5 accepted) | — | — |
| ode vector n=256; rodas5p; adaptive/i; float64; jvp; common | 19,471.74 µs (IQR 61.18; 4 accepted) | — | — |
| ode vector n=256; rodas5p; adaptive/i; float64; jvp; precision | 19,566.55 µs (IQR 35.61; 4 accepted) | — | — |
| ode vector n=256; rodas5p; adaptive/i; float64; primal; common | 10,902.26 µs (IQR 29.48; 4 accepted) | — | — |
| ode vector n=256; rodas5p; adaptive/i; float64; primal; precision | 10,913.37 µs (IQR 147.78; 4 accepted) | — | — |
| ode vector n=256; rodas5p; adaptive/i; float64; vjp; common | 48,856.38 µs (IQR 88.77; 4 accepted) | — | — |
| ode vector n=256; rodas5p; adaptive/i; float64; vjp; precision | 49,281.97 µs (IQR 239.79; 4 accepted) | — | — |
| ode vector n=256; rodas5p; fixed; float32; jvp; common | 143,534.80 µs (IQR 2,498.95) | — | — |
| ode vector n=256; rodas5p; fixed; float32; primal; common | 100,731.79 µs (IQR 2,195.89) | — | — |
| ode vector n=256; rodas5p; fixed; float32; vjp; common | 158,422.30 µs (IQR 2,474.93) | — | — |
| ode vector n=256; rodas5p; fixed; float64; jvp; common | 236,461.99 µs (IQR 6,055.46) | — | — |
| ode vector n=256; rodas5p; fixed; float64; primal; common | 172,781.97 µs (IQR 5,649.58) | — | — |
| ode vector n=256; rodas5p; fixed; float64; vjp; common | 287,428.29 µs (IQR 2,638.85) | — | — |
| ode vector n=256; tsit5; adaptive/i; float32; jvp; common | 10,696.02 µs (IQR 33.44; 4 accepted) | 835.73 µs (IQR 1.24; 0.08×; 4 accepted/0 rejected) | — |
| ode vector n=256; tsit5; adaptive/i; float32; primal; common | 9,219.59 µs (IQR 29.29; 4 accepted) | 940.34 µs (IQR 0.61; 0.10×; 4 accepted/0 rejected) | — |
| ode vector n=256; tsit5; adaptive/i; float32; vjp; common | 23,116.65 µs (IQR 150.53; 4 accepted) | 2,915.55 µs (IQR 2.48; 0.13×; 4 accepted/0 rejected) | — |
| ode vector n=256; tsit5; adaptive/i; float64; jvp; common | 10,728.90 µs (IQR 139.07; 4 accepted) | 1,048.14 µs (IQR 7.63; 0.10×; 4 accepted/0 rejected) | — |
| ode vector n=256; tsit5; adaptive/i; float64; jvp; precision | 10,660.95 µs (IQR 34.33; 4 accepted) | 1,050.54 µs (IQR 1.38; 0.10×; 4 accepted/0 rejected) | — |
| ode vector n=256; tsit5; adaptive/i; float64; primal; common | 10,595.13 µs (IQR 113.66; 4 accepted) | 958.24 µs (IQR 4.47; 0.09×; 4 accepted/0 rejected) | — |
| ode vector n=256; tsit5; adaptive/i; float64; primal; precision | 10,654.43 µs (IQR 25.61; 4 accepted) | 965.46 µs (IQR 1.83; 0.09×; 4 accepted/0 rejected) | — |
| ode vector n=256; tsit5; adaptive/i; float64; vjp; common | 30,809.35 µs (IQR 195.53; 4 accepted) | 3,133.18 µs (IQR 5.36; 0.10×; 4 accepted/0 rejected) | — |
| ode vector n=256; tsit5; adaptive/i; float64; vjp; precision | 30,486.70 µs (IQR 239.22; 4 accepted) | 3,144.62 µs (IQR 4.17; 0.10×; 4 accepted/0 rejected) | — |
| ode vector n=256; tsit5; adaptive/pi; float32; jvp; common | 10,797.49 µs (IQR 37.08; 7 accepted) | 1,361.86 µs (IQR 2.80; 0.13×; 7 accepted/0 rejected) | — |
| ode vector n=256; tsit5; adaptive/pi; float32; primal; common | 10,665.16 µs (IQR 97.31; 7 accepted) | 1,588.26 µs (IQR 0.76; 0.15×; 7 accepted/0 rejected) | — |
| ode vector n=256; tsit5; adaptive/pi; float32; vjp; common | 23,729.22 µs (IQR 217.65; 7 accepted) | 4,782.11 µs (IQR 4.07; 0.20×; 7 accepted/0 rejected) | — |
| ode vector n=256; tsit5; adaptive/pi; float64; jvp; common | 12,545.35 µs (IQR 122.14; 6 accepted) | 1,519.72 µs (IQR 2.28; 0.12×; 6 accepted/0 rejected) | — |
| ode vector n=256; tsit5; adaptive/pi; float64; jvp; precision | 12,597.86 µs (IQR 27.81; 8 accepted) | 1,988.24 µs (IQR 3.08; 0.16×; 8 accepted/0 rejected) | — |
| ode vector n=256; tsit5; adaptive/pi; float64; primal; common | 12,422.36 µs (IQR 30.68; 6 accepted) | 1,382.98 µs (IQR 4.22; 0.11×; 6 accepted/0 rejected) | — |
| ode vector n=256; tsit5; adaptive/pi; float64; primal; precision | 12,491.21 µs (IQR 62.71; 8 accepted) | 1,755.83 µs (IQR 6.89; 0.14×; 8 accepted/0 rejected) | — |
| ode vector n=256; tsit5; adaptive/pi; float64; vjp; common | 32,643.82 µs (IQR 328.80; 6 accepted) | 4,577.72 µs (IQR 4.52; 0.14×; 6 accepted/0 rejected) | — |
| ode vector n=256; tsit5; adaptive/pi; float64; vjp; precision | 32,218.60 µs (IQR 262.72; 8 accepted) | 6,004.30 µs (IQR 8.53; 0.19×; 8 accepted/0 rejected) | — |
| ode vector n=256; tsit5; fixed; float32; jvp; common | 4,815.69 µs (IQR 3.01) | 20,247.24 µs (IQR 63.60; 4.20×) | — |
| ode vector n=256; tsit5; fixed; float32; primal; common | 3,772.58 µs (IQR 99.53) | 24,718.38 µs (IQR 158.30; 6.55×) | — |
| ode vector n=256; tsit5; fixed; float32; vjp; common | 6,561.14 µs (IQR 168.07) | 107,843.14 µs (IQR 242.38; 16.44×) | — |
| ode vector n=256; tsit5; fixed; float64; jvp; common | 6,791.86 µs (IQR 59.23) | 26,276.77 µs (IQR 21.41; 3.87×) | — |
| ode vector n=256; tsit5; fixed; float64; primal; common | 5,359.73 µs (IQR 19.30) | 24,358.60 µs (IQR 45.94; 4.54×) | — |
| ode vector n=256; tsit5; fixed; float64; vjp; common | 9,895.41 µs (IQR 367.82) | 112,896.10 µs (IQR 61.25; 11.41×) | — |
| sdae ensemble n=1 B=256; em; fixed; float32; jvp; common | 20,722.71 µs (IQR 58.92) | — | — |
| sdae ensemble n=1 B=256; em; fixed; float32; primal; common | 14,966.94 µs (IQR 215.58) | — | — |
| sdae ensemble n=1 B=256; em; fixed; float32; vjp; common | 22,540.22 µs (IQR 74.63) | — | — |
| sdae ensemble n=1 B=256; em; fixed; float64; jvp; common | 27,751.28 µs (IQR 131.37) | — | — |
| sdae ensemble n=1 B=256; em; fixed; float64; primal; common | 23,516.06 µs (IQR 133.59) | — | — |
| sdae ensemble n=1 B=256; em; fixed; float64; vjp; common | 29,547.98 µs (IQR 184.48) | — | — |
| sde ensemble n=1 B=1; em; fixed; float32; jvp; common | 564.27 µs (IQR 84.91) | 21,191.74 µs (IQR 6,540.49; 37.56×) | — |
| sde ensemble n=1 B=256; em; fixed; float32; jvp; common | 581.42 µs (IQR 7.49) | 22,011.82 µs (IQR 110.86; 37.86×) | — |
| sde ensemble n=1 B=16384; em; fixed; float32; jvp; common | 656.19 µs (IQR 20.07) | 32,034.58 µs (IQR 2,329.90; 48.82×) | — |
| sde ensemble n=1 B=1; em; fixed; float32; primal; common | 583.65 µs (IQR 34.60) | 20,971.17 µs (IQR 121.86; 35.93×) | — |
| sde ensemble n=1 B=256; em; fixed; float32; primal; common | 616.84 µs (IQR 22.62) | 21,178.90 µs (IQR 56.87; 34.33×) | 1,422.58 µs (IQR 16.90; 2.31×) |
| sde ensemble n=1 B=16384; em; fixed; float32; primal; common | 687.42 µs (IQR 32.84) | 23,449.17 µs (IQR 579.68; 34.11×) | — |
| sde ensemble n=1 B=1; em; fixed; float32; vjp; common | 1,466.50 µs (IQR 333.77) | 46,670.36 µs (IQR 82.06; 31.82×) | — |
| sde ensemble n=1 B=256; em; fixed; float32; vjp; common | 1,166.55 µs (IQR 3.97) | 47,401.24 µs (IQR 232.51; 40.63×) | — |
| sde ensemble n=1 B=16384; em; fixed; float32; vjp; common | 1,250.22 µs (IQR 3.25) | 47,147.89 µs (IQR 491.53; 37.71×) | — |
| sde ensemble n=1 B=1; em; fixed; float64; jvp; common | 562.79 µs (IQR 0.97) | 21,301.17 µs (IQR 180.35; 37.85×) | — |
| sde ensemble n=1 B=256; em; fixed; float64; jvp; common | 767.71 µs (IQR 1.33) | 21,752.46 µs (IQR 120.35; 28.33×) | — |
| sde ensemble n=1 B=16384; em; fixed; float64; jvp; common | 1,919.21 µs (IQR 9.10) | 23,695.59 µs (IQR 142.23; 12.35×) | — |
| sde ensemble n=1 B=1; em; fixed; float64; primal; common | 573.88 µs (IQR 15.04) | 20,832.58 µs (IQR 289.82; 36.30×) | — |
| sde ensemble n=1 B=256; em; fixed; float64; primal; common | 688.80 µs (IQR 5.87) | 21,819.94 µs (IQR 243.25; 31.68×) | — |
| sde ensemble n=1 B=16384; em; fixed; float64; primal; common | 1,798.18 µs (IQR 4.14) | 22,979.60 µs (IQR 67.37; 12.78×) | — |
| sde ensemble n=1 B=1; em; fixed; float64; vjp; common | 1,138.15 µs (IQR 9.63) | 46,553.08 µs (IQR 233.34; 40.90×) | — |
| sde ensemble n=1 B=256; em; fixed; float64; vjp; common | 1,236.17 µs (IQR 9.11) | 49,208.35 µs (IQR 178.54; 39.81×) | — |
| sde ensemble n=1 B=16384; em; fixed; float64; vjp; common | 2,380.32 µs (IQR 2.36) | 51,278.30 µs (IQR 828.63; 21.54×) | — |

## Empty-cell reasons

- `diffrax dae_scalar rk4 jvp`: no equivalent semi-explicit DAE interface
- `diffrax dae_scalar rk4 primal`: no equivalent semi-explicit DAE interface
- `diffrax dae_scalar rk4 vjp`: no equivalent semi-explicit DAE interface
- `diffrax dae_scalar rodas5p jvp`: no equivalent semi-explicit DAE interface
- `diffrax dae_scalar rodas5p primal`: no equivalent semi-explicit DAE interface
- `diffrax dae_scalar rodas5p vjp`: no equivalent semi-explicit DAE interface
- `diffrax dae_scalar tsit5 jvp`: no equivalent semi-explicit DAE interface
- `diffrax dae_scalar tsit5 primal`: no equivalent semi-explicit DAE interface
- `diffrax dae_scalar tsit5 vjp`: no equivalent semi-explicit DAE interface
- `diffrax dae_vector rk4 jvp`: no equivalent semi-explicit DAE interface
- `diffrax dae_vector rk4 primal`: no equivalent semi-explicit DAE interface
- `diffrax dae_vector rk4 vjp`: no equivalent semi-explicit DAE interface
- `diffrax dae_vector rodas5p jvp`: no equivalent semi-explicit DAE interface
- `diffrax dae_vector rodas5p primal`: no equivalent semi-explicit DAE interface
- `diffrax dae_vector rodas5p vjp`: no equivalent semi-explicit DAE interface
- `diffrax dae_vector tsit5 jvp`: no equivalent semi-explicit DAE interface
- `diffrax dae_vector tsit5 primal`: no equivalent semi-explicit DAE interface
- `diffrax dae_vector tsit5 vjp`: no equivalent semi-explicit DAE interface
- `diffrax ode_scalar rk4 jvp`: Diffrax has no public classic RK4 solver
- `diffrax ode_scalar rk4 primal`: Diffrax has no public classic RK4 solver
- `diffrax ode_scalar rk4 vjp`: Diffrax has no public classic RK4 solver
- `diffrax ode_scalar rodas5p jvp`: Diffrax has no Rodas5P implementation
- `diffrax ode_scalar rodas5p primal`: Diffrax has no Rodas5P implementation
- `diffrax ode_scalar rodas5p vjp`: Diffrax has no Rodas5P implementation
- `diffrax ode_vector rk4 jvp`: Diffrax has no public classic RK4 solver
- `diffrax ode_vector rk4 primal`: Diffrax has no public classic RK4 solver
- `diffrax ode_vector rk4 vjp`: Diffrax has no public classic RK4 solver
- `diffrax ode_vector rodas5p jvp`: Diffrax has no Rodas5P implementation
- `diffrax ode_vector rodas5p primal`: Diffrax has no Rodas5P implementation
- `diffrax ode_vector rodas5p vjp`: Diffrax has no Rodas5P implementation
- `diffrax sdae_ensemble em jvp`: no equivalent explicit projected SDAE interface
- `diffrax sdae_ensemble em primal`: no equivalent explicit projected SDAE interface
- `diffrax sdae_ensemble em vjp`: no equivalent explicit projected SDAE interface
- `sciml dae_scalar rodas5p vjp`: First call to automatic differentiation for time gradient
- `sciml dae_vector rodas5p vjp`: First call to automatic differentiation for time gradient
- `sciml ode_scalar euler vjp`: MethodError: no method matching similar(::Float32, ::Tuple{Int64}); MethodError: no method matching similar(::Float64, ::Tuple{Int64}); Sensitivity algorithm ReverseDiffAdjoint only supports vector u0
- `sciml ode_scalar rk4 vjp`: MethodError: no method matching similar(::Float32, ::Tuple{Int64}); MethodError: no method matching similar(::Float64, ::Tuple{Int64}); Sensitivity algorithm ReverseDiffAdjoint only supports vector u0
- `sciml ode_scalar rodas5p vjp`: Sensitivity algorithm ReverseDiffAdjoint only supports vector u0
- `sciml ode_scalar tsit5 vjp`: MethodError: no method matching similar(::Float32, ::Tuple{Int64}); MethodError: no method matching similar(::Float64, ::Tuple{Int64}); Sensitivity algorithm ReverseDiffAdjoint only supports vector u0
- `sciml ode_vector rodas5p vjp`: First call to automatic differentiation for time gradient
- `sciml sdae_ensemble implicit_em jvp`: SciML pathwise SDAE ensemble derivative not enabled in this suite
- `sciml sdae_ensemble implicit_em primal`: ArgumentError: range(0.8, stop=1.2, length=1): endpoints differ
- `sciml sdae_ensemble implicit_em vjp`: SciML pathwise SDAE ensemble derivative not enabled in this suite
- `sciml sde_ensemble em jvp`: SciML pathwise ensemble derivative not enabled in this suite
- `sciml sde_ensemble em primal`: ArgumentError: range(0.8, stop=1.2, length=1): endpoints differ
- `sciml sde_ensemble em vjp`: SciML pathwise ensemble derivative not enabled in this suite

## Reproduction

Run `./run.sh --quick` for the current representative matrix or `./run.sh --full` for
all configured system sizes and ensemble sizes. Raw measurements are under `results/`.
The Python and Julia projects are isolated below this directory and do not alter the
tinydiffeq dependency graph.
