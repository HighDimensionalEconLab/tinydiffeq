import jax

jax.config.update("jax_enable_x64", True)
# XLA:GPU serves float32 dot_general from TF32 tensor cores by default (10-bit
# mantissa, ~1e-3). The float32 tests compare against closed forms, and the
# linear-exponential and Markov paths are matmul-heavy, so without this a GPU
# run disagrees with CPU at ~1e-3 against tolerances set from float32 eps.
# A no-op on CPU.
jax.config.update("jax_default_matmul_precision", "highest")
