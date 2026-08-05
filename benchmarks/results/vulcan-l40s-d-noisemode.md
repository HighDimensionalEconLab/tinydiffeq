# gpu_trajectories — NVIDIA L40S (gpu, jax 0.9.1)

matmul_precision=None, scan_unroll=1, remat=none

| solver | drift | B | d | n_steps | dtype | noise | save | mode | ms/call | traj/s | compile s |
|---|---|---:|---:|---:|---|---|---|---|---:|---:|---:|
| sra1 | stochastic-growth | 16 | 2 | 63 | float32 | explicit | steps | primal | 1.332 | 1.2e+04 | 0.61 |
| sra1 | stochastic-growth | 16 | 2 | 63 | float32 | key | steps | primal | 1.337 | 1.2e+04 | 0.72 |
| sra1 | stochastic-growth | 16 | 2 | 63 | float32 | explicit | steps | grad | 8.066 | 1.98e+03 | 1.58 |
| sra1 | stochastic-growth | 16 | 2 | 63 | float32 | key | steps | grad | 7.812 | 2.05e+03 | 1.56 |
| sra1 | stochastic-growth | 16 | 2 | 255 | float32 | explicit | steps | primal | 5.057 | 3.16e+03 | 0.41 |
| sra1 | stochastic-growth | 16 | 2 | 255 | float32 | key | steps | primal | 5.012 | 3.19e+03 | 0.74 |
| sra1 | stochastic-growth | 16 | 2 | 255 | float32 | explicit | steps | grad | 31.584 | 507 | 1.43 |
| sra1 | stochastic-growth | 16 | 2 | 255 | float32 | key | steps | grad | 30.595 | 523 | 1.62 |
| em | stochastic-growth | 16 | 2 | 63 | float32 | explicit | steps | primal | 0.851 | 1.88e+04 | 0.23 |
| em | stochastic-growth | 16 | 2 | 63 | float32 | key | steps | primal | 0.854 | 1.87e+04 | 0.35 |
| em | stochastic-growth | 16 | 2 | 63 | float32 | explicit | steps | grad | 4.469 | 3.58e+03 | 0.93 |
| em | stochastic-growth | 16 | 2 | 63 | float32 | key | steps | grad | 4.436 | 3.61e+03 | 1.02 |
| em | stochastic-growth | 16 | 2 | 255 | float32 | explicit | steps | primal | 3.106 | 5.15e+03 | 0.22 |
| em | stochastic-growth | 16 | 2 | 255 | float32 | key | steps | primal | 3.117 | 5.13e+03 | 0.36 |
| em | stochastic-growth | 16 | 2 | 255 | float32 | explicit | steps | grad | 17.610 | 909 | 0.93 |
| em | stochastic-growth | 16 | 2 | 255 | float32 | key | steps | grad | 17.427 | 918 | 1.05 |
