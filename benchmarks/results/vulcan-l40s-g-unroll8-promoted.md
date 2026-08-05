# gpu_trajectories — NVIDIA L40S (gpu, jax 0.9.1)

matmul_precision=None, scan_unroll=8, remat=none

| solver | drift | B | d | n_steps | dtype | noise | save | mode | ms/call | traj/s | compile s |
|---|---|---:|---:|---:|---|---|---|---|---:|---:|---:|
| sra1 | stochastic-growth | 1 | 2 | 63 | float32 | explicit | steps | primal | 0.715 | 1.4e+03 | 2.18 |
| sra1 | stochastic-growth | 1 | 2 | 63 | float32 | explicit | steps | grad | 1.763 | 567 | 15.09 |
| sra1 | stochastic-growth | 1 | 2 | 255 | float32 | explicit | steps | primal | 2.501 | 400 | 2.12 |
| sra1 | stochastic-growth | 1 | 2 | 255 | float32 | explicit | steps | grad | 6.679 | 150 | 15.13 |
| sra1 | stochastic-growth | 32 | 2 | 63 | float32 | explicit | steps | primal | 1.089 | 2.94e+04 | 2.61 |
| sra1 | stochastic-growth | 32 | 2 | 63 | float32 | explicit | steps | grad | 3.411 | 9.38e+03 | 13.02 |
| sra1 | stochastic-growth | 32 | 2 | 255 | float32 | explicit | steps | primal | 4.008 | 7.98e+03 | 2.12 |
| sra1 | stochastic-growth | 32 | 2 | 255 | float32 | explicit | steps | grad | 13.284 | 2.41e+03 | 13.37 |
| em | stochastic-growth | 1 | 2 | 63 | float32 | explicit | steps | primal | 0.445 | 2.25e+03 | 1.24 |
| em | stochastic-growth | 1 | 2 | 63 | float32 | explicit | steps | grad | 1.025 | 976 | 7.04 |
| em | stochastic-growth | 1 | 2 | 255 | float32 | explicit | steps | primal | 1.451 | 689 | 1.25 |
| em | stochastic-growth | 1 | 2 | 255 | float32 | explicit | steps | grad | 3.635 | 275 | 7.04 |
| em | stochastic-growth | 32 | 2 | 63 | float32 | explicit | steps | primal | 0.626 | 5.11e+04 | 1.22 |
| em | stochastic-growth | 32 | 2 | 63 | float32 | explicit | steps | grad | 1.892 | 1.69e+04 | 6.61 |
| em | stochastic-growth | 32 | 2 | 255 | float32 | explicit | steps | primal | 2.186 | 1.46e+04 | 1.20 |
| em | stochastic-growth | 32 | 2 | 255 | float32 | explicit | steps | grad | 7.066 | 4.53e+03 | 6.87 |
| rk4 | neoclassical | 1 | 2 | 63 | float32 | none | steps | primal | 1.251 | 799 | 2.71 |
| rk4 | neoclassical | 1 | 2 | 63 | float32 | none | steps | grad | 3.360 | 298 | 32.03 |
| rk4 | neoclassical | 1 | 2 | 255 | float32 | none | steps | primal | 4.642 | 215 | 3.11 |
| rk4 | neoclassical | 1 | 2 | 255 | float32 | none | steps | grad | 13.240 | 75.5 | 32.79 |
| rk4 | neoclassical | 32 | 2 | 63 | float32 | none | steps | primal | 1.612 | 1.98e+04 | 3.95 |
| rk4 | neoclassical | 32 | 2 | 63 | float32 | none | steps | grad | 6.025 | 5.31e+03 | 34.90 |
| rk4 | neoclassical | 32 | 2 | 255 | float32 | none | steps | primal | 6.146 | 5.21e+03 | 4.02 |
| rk4 | neoclassical | 32 | 2 | 255 | float32 | none | steps | grad | 25.152 | 1.27e+03 | 44.81 |
