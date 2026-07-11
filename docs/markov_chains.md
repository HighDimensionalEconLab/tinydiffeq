# Finite-State Markov Chains

tinydiffeq provides simulation and deterministic probability-mass forecasts for
homogeneous, finite-state discrete-time Markov chains (DTMCs) and continuous-time
Markov chains (CTMCs). The APIs are JIT-compatible and can be mapped over
independent inputs with `jax.vmap`.

## Prepare once, simulate many times

Preparation validates the matrix, normalizes probabilities where appropriate,
and constructs [Vose alias tables](https://doi.org/10.1109/32.92917). It is a
host operation and must happen outside `jit` and `vmap`; the resulting object is
a pytree of JAX arrays.

```python
import jax
import jax.numpy as jnp

from tinydiffeq import (
    AssociativeMarkov,
    DiscreteMarkovChain,
    SaveAt,
    simulate_markov_chain,
)

chain = DiscreteMarkovChain(
    jnp.asarray([[0.9, 0.1], [0.25, 0.75]], dtype=jnp.float32)
)
keys = jax.random.split(jax.random.key(0), 256)

def trajectory(key):
    return simulate_markov_chain(
        chain,
        jnp.int32(0),
        key=key,
        num_steps=1_024,
        method=AssociativeMarkov(),
        save_at=SaveAt(steps=True),
    ).xs

paths = jax.jit(jax.vmap(trajectory))(keys)  # (256, 1025)
```

An alias draw uses one uniform variate: its integer part chooses the bucket and
its fractional part chooses the bucket's primary or alias state. Sampling is
therefore constant work per realized transition after preparation.

## Sequential and associative methods

`SequentialMarkov()` is the default. It follows the chain chronologically with
`lax.scan` and is consistently fastest on CPU. `SequentialMarkov(unroll=16)`
can reduce GPU loop overhead for large ensembles, at substantial CPU cost.

`AssociativeMarkov()` exposes temporal parallelism. For uniform draw `u_t`, let
`F_t(i)` be the sampled successor from state `i`. Function composition is
associative:

\[
    (F_b \circ F_a)(i) = F_b(F_a(i)).
\]

`lax.associative_scan` computes every prefix map, after which each prefix is
applied to the initial state. Sequential and associative DTMC paths are exactly
identical for the same key. The parallel method does more work and stores one
map of length `K` per random step, so it is an explicit option rather than an
automatic backend choice. JAX documents `associative_scan` as a parallel scan
over an associative operator in its
[control-flow API](https://docs.jax.dev/en/latest/jax.lax.html).

## Continuous time

```python
from tinydiffeq import (
    ContinuousTimeMarkovChain,
    SequentialMarkov,
    simulate_continuous_time_markov_chain,
)

chain = ContinuousTimeMarkovChain(
    jnp.asarray([[-2.0, 2.0], [1.0, -1.0]], dtype=jnp.float64)
)
solution = simulate_continuous_time_markov_chain(
    chain,
    0.0,
    10.0,
    jnp.int32(0),
    key=jax.random.key(1),
    max_jumps=128,
    method=SequentialMarkov(),
    save_at=SaveAt(ts=jnp.linspace(0.0, 10.0, 101)),
)
```

The sequential method is Gillespie's Direct/Doob recurrence for a finite-state
generator: draw an exponential holding time from the current exit rate, then
draw from the embedded jump chain. This follows the same exact chronological
principle as SciML's
[`Direct` aggregator with `SSAStepper`](https://docs.sciml.ai/JumpProcesses/stable/jump_solve/);
see also SciML's
[Gillespie tutorial](https://docs.sciml.ai/JumpProcesses/stable/tutorials/discrete_stochastic_example/).

For a random jump map `F` and state-dependent holding-time map `H`, composition
is

\[
  (F_b,H_b)\circ(F_a,H_a)
  = \left(F_b\circ F_a,\;H_a + H_b\circ F_a\right).
\]

This operation is mathematically associative and gives a parallel CTMC method.
Floating-point addition is not exactly associative, however: post-jump states
match the sequential method, while cumulative event times differ by rounding.
On the local 1024-jump test the maximum discrepancy was about `1e-3` in
float32 and `3e-12` in float64. Consequently `AssociativeMarkov()` is explicit
opt-in and may classify an event extremely close to `t_1` differently.

## Forecast distributions

Use `forecast_markov_chain` when the distribution itself, rather than a sampled
path, is the object of interest. For a row probability vector \(\pi_n\) and
transition matrix \(P\), the endpoint is

\[
    \pi_N = \pi_0 P^N.
\]

```python
from tinydiffeq import MatrixPowerMarkov, forecast_markov_chain

forecast = forecast_markov_chain(
    chain,
    jnp.asarray([1.0, 0.0], dtype=jnp.float32),
    num_steps=1_024,
    method=MatrixPowerMarkov(),
)
```

Binary matrix powering is the endpoint default and avoids 1,024 chronological
matrix-vector products. `SaveAt(steps=True)` returns every distribution and
defaults to `SequentialMarkov()`. `AssociativeMarkov()` forms prefix matrix
products; it does more matrix-matrix work but can be much faster for small state
spaces on GPU. Integer `SaveAt(ts=...)` selects rows from the full forecast.

For a dense CTMC generator \(Q\), `forecast_continuous_time_markov_chain`
defaults to scaling-and-squaring evaluation of

\[
    \pi(t) = \pi(t_0)\exp((t-t_0)Q).
\]

The dense method is the right baseline for small state spaces. It is especially
effective for a batch of initial distributions sharing one fixed generator,
because JAX can reuse the matrix exponential.

### Matrix-free CTMC forecasts with probability pytrees

Large structured state spaces need not construct \(Q\). Supply the forward
generator action \(L(\pi)\) and use `KrylovExponential`. Every floating leaf in the
probability pytree contains a group of discrete-state masses; the sum across
all leaves must be one. The action must preserve the exact structure and dtype.

```python
from tinydiffeq import (
    KrylovExponential,
    MatrixFreeContinuousTimeMarkovChain,
    SaveAt,
    forecast_continuous_time_markov_chain,
)

def forward_generator(probabilities):
    employed = probabilities["employment"]
    inventory = probabilities["inventory"]
    employment_switch = 0.4 * employed[0] - 0.2 * employed[1]
    sector_switch = 0.1 * employed[0] - 0.3 * inventory[0]
    inventory_switch = 0.5 * inventory[0] - 0.25 * inventory[1]
    return {
        "employment": jnp.asarray(
            [-employment_switch - sector_switch, employment_switch]
        ),
        "inventory": jnp.asarray(
            [sector_switch - inventory_switch, inventory_switch]
        ),
    }

chain = MatrixFreeContinuousTimeMarkovChain(forward_generator)
distribution_0 = {
    "employment": jnp.asarray([0.2, 0.3]),
    "inventory": jnp.asarray([0.1, 0.4]),
}
times = jnp.linspace(0.0, 10.0, 101)
forecast = forecast_continuous_time_markov_chain(
    chain,
    0.0,
    10.0,
    distribution_0,
    method=KrylovExponential(krylov_dim=30, num_substeps=2),
    save_at=SaveAt(ts=times),
)
```

The implementation temporarily ravels the pytree for Arnoldi orthogonalization,
but generator evaluations and returned probabilities retain the user structure.
It computes exponential-vector products without forming either the generator or
its exponential. `krylov_dim` and `num_substeps` are static accuracy/work
controls. The defaults are 30 and 1; increase the dimension or split a long time
interval when `forecast.ok` is false. The check combines the leading Arnoldi
error estimate with finiteness, nonnegativity to a precision-scaled tolerance,
and conservation of total mass. Default Krylov tolerances are `rtol=1e-5`,
`atol=1e-7` in float32 and `rtol=1e-10`, `atol=1e-12` in float64.
Two-pass reorthogonalization is the stable default. For a repeatedly used,
validated generator, `reorthogonalization_passes=1` reduces basis traffic and
can materially improve large CPU/GPU forecasts; compare against two passes in
both supported precisions before selecting it.

For unknown or widely varying horizons, replace the method with
`AdaptiveKrylovExponential(krylov_dim=30, max_steps=128)`. It keeps the basis
size static and accepts or rejects internal time slices from the Arnoldi
residual. This remains matrix-free and `vmap`-compatible; `max_steps` is the
combined accepted/rejected attempt budget. See
[Linear Exponential Solves](exponential.md#adaptive-matrix-free-propagation)
for the tolerance, failure, and differentiation contracts.

This follows the same Arnoldi exponential-action family exposed by SciML's
[`ExponentialUtilities.expv`](https://docs.sciml.ai/ExponentialUtilities/stable/expv/).
Uniformization is another important CTMC technique, but its Poisson truncation
can require many generator applications when the maximum exit rate times the
horizon is large; it is therefore not used as the general matrix-free default.

Distribution forecasts support ordinary JVPs and VJPs with respect to the
initial probability array or pytree. Valid initial distributions are passed
through exactly—the implementation does not silently renormalize and thereby
alter their derivatives. Use zero-sum tangents when interpreting a JVP as a
direction within the probability simplex.

## Output and failure contract

- The default `SaveAt(t_1=True)` returns the endpoint.
- DTMC `SaveAt(steps=True)` returns `num_steps + 1` states with no padding.
- DTMC `SaveAt(ts=...)` accepts integer step indices.
- CTMC `SaveAt(steps=True)` returns `max_jumps + 1` rows. Events after `t_1`
  are replaced by endpoint padding and identified by `sol.accepted`.
- Markov states are integers, so steps mode requires `fill="last"`; `fill="inf"`
  is rejected rather than silently casting an infinite sentinel.
- CTMC `SaveAt(ts=...)` evaluates the exact right-continuous piecewise-constant
  path at the requested times.
- CTMC `sol.ok` is false if `max_jumps` does not cover `t_1`. Reaching an
  absorbing state covers every later time and succeeds immediately.

## Performance choice

Use the measured table in
[`performance_comparison/MARKOV_RESULTS.md`](https://github.com/HighDimensionalEconLab/tinydiffeq/blob/main/performance_comparison/MARKOV_RESULTS.md)
for guidance. In short: use sequential scan on CPU; benchmark associative scan
against an unrolled sequential scan on GPU. State count, path length, output
size, and ensemble size all affect the crossover.

## Differentiation scope for sampled paths

The sampled-path simulators are primal-only. Integer states and categorical branch choices
do not have ordinary pathwise JVPs or VJPs. Prepared transition/generator
matrices are therefore not differentiation targets. Future sensitivity support
must choose an estimator deliberately—such as likelihood-ratio/score-function
estimators, categorical relaxations, or expectation-semiring calculations—rather
than returning a misleading zero derivative through sampled indices.
