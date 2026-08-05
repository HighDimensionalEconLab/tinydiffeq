# gpu_trajectories — NVIDIA L40S (gpu, jax 0.9.1)

matmul_precision=None, scan_unroll=4, remat=none

| solver | drift | B | d | n_steps | dtype | noise | save | mode | ms/call | traj/s | compile s |
|---|---|---:|---:|---:|---|---|---|---|---:|---:|---:|
| sra1 | stochastic-growth | 1 | 2 | 63 | float32 | explicit | steps | primal | 0.697 | 1.43e+03 | 1.25 |
| sra1 | stochastic-growth | 1 | 2 | 63 | float32 | explicit | steps | grad | 2.124 | 471 | 7.26 |
| sra1 | stochastic-growth | 1 | 2 | 255 | float32 | explicit | steps | primal | 2.564 | 390 | 1.18 |
| sra1 | stochastic-growth | 1 | 2 | 255 | float32 | explicit | steps | grad | 8.216 | 122 | 7.18 |
| sra1 | stochastic-growth | 4 | 2 | 63 | float32 | explicit | steps | primal | 1.069 | 3.74e+03 | 1.25 |
| sra1 | stochastic-growth | 4 | 2 | 63 | float32 | explicit | steps | grad | 4.304 | 929 | 6.75 |
| sra1 | stochastic-growth | 4 | 2 | 255 | float32 | explicit | steps | primal | 3.934 | 1.02e+03 | 0.99 |
| sra1 | stochastic-growth | 4 | 2 | 255 | float32 | explicit | steps | grad | 16.605 | 241 | 5.85 |
| sra1 | stochastic-growth | 16 | 2 | 63 | float32 | explicit | steps | primal | 1.099 | 1.46e+04 | 1.25 |
| sra1 | stochastic-growth | 16 | 2 | 63 | float32 | explicit | steps | grad | 4.105 | 3.9e+03 | 6.14 |
| sra1 | stochastic-growth | 16 | 2 | 255 | float32 | explicit | steps | primal | 4.041 | 3.96e+03 | 1.07 |
| sra1 | stochastic-growth | 16 | 2 | 255 | float32 | explicit | steps | grad | 15.869 | 1.01e+03 | 5.67 |
| sra1 | stochastic-growth | 32 | 2 | 63 | float32 | explicit | steps | primal | 1.121 | 2.85e+04 | 1.77 |
| sra1 | stochastic-growth | 32 | 2 | 63 | float32 | explicit | steps | grad | 4.043 | 7.92e+03 | 6.63 |
| sra1 | stochastic-growth | 32 | 2 | 255 | float32 | explicit | steps | primal | 4.116 | 7.78e+03 | 1.36 |
| sra1 | stochastic-growth | 32 | 2 | 255 | float32 | explicit | steps | grad | 15.573 | 2.05e+03 | 6.73 |
| em | stochastic-growth | 1 | 2 | 63 | float32 | explicit | steps | primal | 0.463 | 2.16e+03 | 0.78 |
| em | stochastic-growth | 1 | 2 | 63 | float32 | explicit | steps | grad | 1.181 | 847 | 3.30 |
| em | stochastic-growth | 1 | 2 | 255 | float32 | explicit | steps | primal | 1.534 | 652 | 0.78 |
| em | stochastic-growth | 1 | 2 | 255 | float32 | explicit | steps | grad | 4.280 | 234 | 3.31 |
| em | stochastic-growth | 4 | 2 | 63 | float32 | explicit | steps | primal | 0.591 | 6.77e+03 | 0.67 |
| em | stochastic-growth | 4 | 2 | 63 | float32 | explicit | steps | grad | 2.295 | 1.74e+03 | 3.77 |
| em | stochastic-growth | 4 | 2 | 255 | float32 | explicit | steps | primal | 2.103 | 1.9e+03 | 0.66 |
| em | stochastic-growth | 4 | 2 | 255 | float32 | explicit | steps | grad | 8.796 | 455 | 3.75 |
| em | stochastic-growth | 16 | 2 | 63 | float32 | explicit | steps | primal | 0.612 | 2.61e+04 | 0.67 |
| em | stochastic-growth | 16 | 2 | 63 | float32 | explicit | steps | grad | 2.176 | 7.35e+03 | 3.59 |
| em | stochastic-growth | 16 | 2 | 255 | float32 | explicit | steps | primal | 2.140 | 7.48e+03 | 0.65 |
| em | stochastic-growth | 16 | 2 | 255 | float32 | explicit | steps | grad | 8.230 | 1.94e+03 | 3.51 |
| em | stochastic-growth | 32 | 2 | 63 | float32 | explicit | steps | primal | 0.608 | 5.26e+04 | 1.07 |
| em | stochastic-growth | 32 | 2 | 63 | float32 | explicit | steps | grad | 2.183 | 1.47e+04 | 3.92 |
| em | stochastic-growth | 32 | 2 | 255 | float32 | explicit | steps | primal | 2.194 | 1.46e+04 | 0.94 |
| em | stochastic-growth | 32 | 2 | 255 | float32 | explicit | steps | grad | 8.260 | 3.87e+03 | 4.01 |
| rk4 | neoclassical | 1 | 2 | 63 | float32 | none | steps | primal | 1.263 | 792 | 2.06 |
| rk4 | neoclassical | 1 | 2 | 63 | float32 | none | steps | grad | 4.307 | 232 | 14.47 |
| rk4 | neoclassical | 1 | 2 | 255 | float32 | none | steps | primal | 4.710 | 212 | 2.12 |
| rk4 | neoclassical | 1 | 2 | 255 | float32 | none | steps | grad | 17.146 | 58.3 | 14.55 |
| rk4 | neoclassical | 4 | 2 | 63 | float32 | none | steps | primal | 2.140 | 1.87e+03 | 3.15 |
| rk4 | neoclassical | 4 | 2 | 63 | float32 | none | steps | grad | 7.487 | 534 | 17.59 |
| rk4 | neoclassical | 4 | 2 | 255 | float32 | none | steps | primal | 8.162 | 490 | 3.15 |
| rk4 | neoclassical | 4 | 2 | 255 | float32 | none | steps | grad | 29.879 | 134 | 17.49 |
| rk4 | neoclassical | 16 | 2 | 63 | float32 | none | steps | primal | 1.746 | 9.17e+03 | 2.47 |
| rk4 | neoclassical | 16 | 2 | 63 | float32 | none | steps | grad | 7.121 | 2.25e+03 | 18.36 |
| rk4 | neoclassical | 16 | 2 | 255 | float32 | none | steps | primal | 6.706 | 2.39e+03 | 2.61 |
| rk4 | neoclassical | 16 | 2 | 255 | float32 | none | steps | grad | 28.370 | 564 | 24.23 |
| rk4 | neoclassical | 32 | 2 | 63 | float32 | none | steps | primal | 1.649 | 1.94e+04 | 2.39 |
| rk4 | neoclassical | 32 | 2 | 63 | float32 | none | steps | grad | 7.063 | 4.53e+03 | 15.94 |
| rk4 | neoclassical | 32 | 2 | 255 | float32 | none | steps | primal | 6.319 | 5.06e+03 | 2.55 |
| rk4 | neoclassical | 32 | 2 | 255 | float32 | none | steps | grad | 29.352 | 1.09e+03 | 23.50 |
