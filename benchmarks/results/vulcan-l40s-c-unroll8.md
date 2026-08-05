# gpu_trajectories — NVIDIA L40S (gpu, jax 0.9.1)

matmul_precision=None, scan_unroll=8, remat=none

| solver | drift | B | d | n_steps | dtype | noise | save | mode | ms/call | traj/s | compile s |
|---|---|---:|---:|---:|---|---|---|---|---:|---:|---:|
| sra1 | stochastic-growth | 1 | 2 | 63 | float32 | explicit | steps | primal | 0.700 | 1.43e+03 | 2.10 |
| sra1 | stochastic-growth | 1 | 2 | 63 | float32 | explicit | steps | grad | 1.752 | 571 | 15.51 |
| sra1 | stochastic-growth | 1 | 2 | 255 | float32 | explicit | steps | primal | 2.500 | 400 | 2.13 |
| sra1 | stochastic-growth | 1 | 2 | 255 | float32 | explicit | steps | grad | 6.634 | 151 | 15.41 |
| sra1 | stochastic-growth | 4 | 2 | 63 | float32 | explicit | steps | primal | 1.065 | 3.76e+03 | 2.01 |
| sra1 | stochastic-growth | 4 | 2 | 63 | float32 | explicit | steps | grad | 3.708 | 1.08e+03 | 13.49 |
| sra1 | stochastic-growth | 4 | 2 | 255 | float32 | explicit | steps | primal | 3.919 | 1.02e+03 | 1.74 |
| sra1 | stochastic-growth | 4 | 2 | 255 | float32 | explicit | steps | grad | 14.400 | 278 | 12.48 |
| sra1 | stochastic-growth | 16 | 2 | 63 | float32 | explicit | steps | primal | 1.080 | 1.48e+04 | 2.11 |
| sra1 | stochastic-growth | 16 | 2 | 63 | float32 | explicit | steps | grad | 3.548 | 4.51e+03 | 12.48 |
| sra1 | stochastic-growth | 16 | 2 | 255 | float32 | explicit | steps | primal | 3.985 | 4.02e+03 | 1.94 |
| sra1 | stochastic-growth | 16 | 2 | 255 | float32 | explicit | steps | grad | 13.776 | 1.16e+03 | 11.73 |
| sra1 | stochastic-growth | 32 | 2 | 63 | float32 | explicit | steps | primal | 1.091 | 2.93e+04 | 2.71 |
| sra1 | stochastic-growth | 32 | 2 | 63 | float32 | explicit | steps | grad | 3.432 | 9.32e+03 | 13.43 |
| sra1 | stochastic-growth | 32 | 2 | 255 | float32 | explicit | steps | primal | 4.075 | 7.85e+03 | 2.35 |
| sra1 | stochastic-growth | 32 | 2 | 255 | float32 | explicit | steps | grad | 13.295 | 2.41e+03 | 13.82 |
| em | stochastic-growth | 1 | 2 | 63 | float32 | explicit | steps | primal | 0.434 | 2.3e+03 | 1.22 |
| em | stochastic-growth | 1 | 2 | 63 | float32 | explicit | steps | grad | 1.003 | 997 | 7.30 |
| em | stochastic-growth | 1 | 2 | 255 | float32 | explicit | steps | primal | 1.425 | 702 | 1.25 |
| em | stochastic-growth | 1 | 2 | 255 | float32 | explicit | steps | grad | 3.566 | 280 | 7.36 |
| em | stochastic-growth | 4 | 2 | 63 | float32 | explicit | steps | primal | 0.599 | 6.68e+03 | 0.95 |
| em | stochastic-growth | 4 | 2 | 63 | float32 | explicit | steps | grad | 2.018 | 1.98e+03 | 6.38 |
| em | stochastic-growth | 4 | 2 | 255 | float32 | explicit | steps | primal | 2.032 | 1.97e+03 | 0.97 |
| em | stochastic-growth | 4 | 2 | 255 | float32 | explicit | steps | grad | 7.539 | 531 | 6.39 |
| em | stochastic-growth | 16 | 2 | 63 | float32 | explicit | steps | primal | 0.615 | 2.6e+04 | 1.01 |
| em | stochastic-growth | 16 | 2 | 63 | float32 | explicit | steps | grad | 1.907 | 8.39e+03 | 6.51 |
| em | stochastic-growth | 16 | 2 | 255 | float32 | explicit | steps | primal | 2.089 | 7.66e+03 | 0.99 |
| em | stochastic-growth | 16 | 2 | 255 | float32 | explicit | steps | grad | 7.179 | 2.23e+03 | 6.39 |
| em | stochastic-growth | 32 | 2 | 63 | float32 | explicit | steps | primal | 0.623 | 5.13e+04 | 1.25 |
| em | stochastic-growth | 32 | 2 | 63 | float32 | explicit | steps | grad | 1.897 | 1.69e+04 | 6.80 |
| em | stochastic-growth | 32 | 2 | 255 | float32 | explicit | steps | primal | 2.191 | 1.46e+04 | 1.15 |
| em | stochastic-growth | 32 | 2 | 255 | float32 | explicit | steps | grad | 7.048 | 4.54e+03 | 7.12 |
| rk4 | neoclassical | 1 | 2 | 63 | float32 | none | steps | primal | 1.247 | 802 | 3.15 |
| rk4 | neoclassical | 1 | 2 | 63 | float32 | none | steps | grad | 3.362 | 297 | 32.87 |
| rk4 | neoclassical | 1 | 2 | 255 | float32 | none | steps | primal | 4.601 | 217 | 3.03 |
| rk4 | neoclassical | 1 | 2 | 255 | float32 | none | steps | grad | 13.169 | 75.9 | 32.81 |
| rk4 | neoclassical | 4 | 2 | 63 | float32 | none | steps | primal | 2.121 | 1.89e+03 | 5.32 |
| rk4 | neoclassical | 4 | 2 | 63 | float32 | none | steps | grad | 6.540 | 612 | 40.33 |
| rk4 | neoclassical | 4 | 2 | 255 | float32 | none | steps | primal | 8.167 | 490 | 5.44 |
| rk4 | neoclassical | 4 | 2 | 255 | float32 | none | steps | grad | 25.998 | 154 | 39.74 |
| rk4 | neoclassical | 16 | 2 | 63 | float32 | none | steps | primal | 1.730 | 9.25e+03 | 4.07 |
| rk4 | neoclassical | 16 | 2 | 63 | float32 | none | steps | grad | 6.083 | 2.63e+03 | 41.33 |
| rk4 | neoclassical | 16 | 2 | 255 | float32 | none | steps | primal | 6.529 | 2.45e+03 | 4.32 |
| rk4 | neoclassical | 16 | 2 | 255 | float32 | none | steps | grad | 24.196 | 661 | 48.49 |
| rk4 | neoclassical | 32 | 2 | 63 | float32 | none | steps | primal | 1.625 | 1.97e+04 | 4.04 |
| rk4 | neoclassical | 32 | 2 | 63 | float32 | none | steps | grad | 6.009 | 5.32e+03 | 35.89 |
| rk4 | neoclassical | 32 | 2 | 255 | float32 | none | steps | primal | 6.132 | 5.22e+03 | 4.40 |
| rk4 | neoclassical | 32 | 2 | 255 | float32 | none | steps | grad | 25.074 | 1.28e+03 | 45.26 |
