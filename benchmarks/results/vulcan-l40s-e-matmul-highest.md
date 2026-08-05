# gpu_trajectories — NVIDIA L40S (gpu, jax 0.9.1)

matmul_precision=highest, scan_unroll=1, remat=none

| solver | drift | B | d | n_steps | dtype | noise | save | mode | ms/call | traj/s | compile s |
|---|---|---:|---:|---:|---|---|---|---|---:|---:|---:|
| sra1 | stochastic-growth | 32 | 2 | 63 | float32 | explicit | steps | primal | 1.826 | 1.75e+04 | 0.43 |
| sra1 | stochastic-growth | 32 | 2 | 63 | float32 | explicit | steps | grad | 9.604 | 3.33e+03 | 1.27 |
| sra1 | stochastic-growth | 32 | 2 | 255 | float32 | explicit | steps | primal | 7.049 | 4.54e+03 | 0.34 |
| sra1 | stochastic-growth | 32 | 2 | 255 | float32 | explicit | steps | grad | 38.592 | 829 | 1.31 |
| rk4 | neoclassical | 32 | 2 | 63 | float32 | none | steps | primal | 2.811 | 1.14e+04 | 0.73 |
| rk4 | neoclassical | 32 | 2 | 63 | float32 | none | steps | grad | 16.519 | 1.94e+03 | 2.97 |
| rk4 | neoclassical | 32 | 2 | 255 | float32 | none | steps | primal | 11.022 | 2.9e+03 | 0.87 |
| rk4 | neoclassical | 32 | 2 | 255 | float32 | none | steps | grad | 66.519 | 481 | 3.86 |
