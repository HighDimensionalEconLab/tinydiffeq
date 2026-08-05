# gpu_trajectories — NVIDIA L40S (gpu, jax 0.9.1)

matmul_precision=None, scan_unroll=1, remat=none

| solver | drift | B | d | n_steps | dtype | noise | save | mode | ms/call | traj/s | compile s |
|---|---|---:|---:|---:|---|---|---|---|---:|---:|---:|
| sra1 | stochastic-growth | 1 | 2 | 63 | float32 | explicit | steps | primal | 0.962 | 1.04e+03 | 0.47 |
| sra1 | stochastic-growth | 1 | 2 | 63 | float32 | explicit | steps | grad | 5.053 | 198 | 1.51 |
| sra1 | stochastic-growth | 1 | 2 | 255 | float32 | explicit | steps | primal | 3.588 | 279 | 0.37 |
| sra1 | stochastic-growth | 1 | 2 | 255 | float32 | explicit | steps | grad | 19.817 | 50.5 | 1.36 |
| sra1 | stochastic-growth | 32 | 2 | 63 | float32 | explicit | steps | primal | 1.346 | 2.38e+04 | 0.63 |
| sra1 | stochastic-growth | 32 | 2 | 63 | float32 | explicit | steps | grad | 8.021 | 3.99e+03 | 1.57 |
| sra1 | stochastic-growth | 32 | 2 | 255 | float32 | explicit | steps | primal | 5.147 | 6.22e+03 | 0.45 |
| sra1 | stochastic-growth | 32 | 2 | 255 | float32 | explicit | steps | grad | 31.858 | 1e+03 | 1.54 |
| em | stochastic-growth | 1 | 2 | 63 | float32 | explicit | steps | primal | 0.695 | 1.44e+03 | 0.20 |
| em | stochastic-growth | 1 | 2 | 63 | float32 | explicit | steps | grad | 3.072 | 326 | 0.77 |
| em | stochastic-growth | 1 | 2 | 255 | float32 | explicit | steps | primal | 2.501 | 400 | 0.22 |
| em | stochastic-growth | 1 | 2 | 255 | float32 | explicit | steps | grad | 11.790 | 84.8 | 0.74 |
| em | stochastic-growth | 32 | 2 | 63 | float32 | explicit | steps | primal | 0.844 | 3.79e+04 | 0.26 |
| em | stochastic-growth | 32 | 2 | 63 | float32 | explicit | steps | grad | 4.686 | 6.83e+03 | 0.92 |
| em | stochastic-growth | 32 | 2 | 255 | float32 | explicit | steps | primal | 3.073 | 1.04e+04 | 0.25 |
| em | stochastic-growth | 32 | 2 | 255 | float32 | explicit | steps | grad | 18.493 | 1.73e+03 | 0.94 |
| rk4 | neoclassical | 1 | 2 | 63 | float32 | none | steps | primal | 1.748 | 572 | 0.72 |
| rk4 | neoclassical | 1 | 2 | 63 | float32 | none | steps | grad | 9.451 | 106 | 2.52 |
| rk4 | neoclassical | 1 | 2 | 255 | float32 | none | steps | primal | 6.671 | 150 | 0.85 |
| rk4 | neoclassical | 1 | 2 | 255 | float32 | none | steps | grad | 37.462 | 26.7 | 2.66 |
| rk4 | neoclassical | 32 | 2 | 63 | float32 | none | steps | primal | 2.109 | 1.52e+04 | 0.69 |
| rk4 | neoclassical | 32 | 2 | 63 | float32 | none | steps | grad | 14.227 | 2.25e+03 | 3.12 |
| rk4 | neoclassical | 32 | 2 | 255 | float32 | none | steps | primal | 8.247 | 3.88e+03 | 0.83 |
| rk4 | neoclassical | 32 | 2 | 255 | float32 | none | steps | grad | 56.899 | 562 | 3.94 |
