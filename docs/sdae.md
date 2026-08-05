# Semi-Explicit Index-1 SDAEs

`solve_semi_explicit_sdae` integrates the Itô system

$$
dy = f(y,z,t)\,dt + \sigma(y,z,t)\,dW,
\qquad 0=g(y,z,t),
$$

with fixed steps and diagonal noise in the differential state. The algebraic
Jacobian $g_z$ must be square and nonsingular along the chosen root branch.
Arrays and pytrees are supported for both `y` and `z`.

```python
import jax
import jax.numpy as jnp

from tinydiffeq import EulerMaruyama, SaveAt, solve_semi_explicit_sdae


def drift(y, z, t, args, p, algebraic_aux):
    value = p["mu"] * z
    saved_aux = {
        "variance_scale": algebraic_aux["variance_scale"],
        "flow": value,
    }
    return value, saved_aux


def diffusion(y, z, t, args, p, algebraic_aux):
    return p["sigma"] * z


def constraint(y, z, t, args, p):
    context = {"variance_scale": p["sigma"] ** 2 * z**2}
    return z - y, context


sol = solve_semi_explicit_sdae(
    drift,
    diffusion,
    constraint,
    EulerMaruyama(),
    0.0,
    1.0,
    jnp.asarray(1.0),
    jnp.asarray(0.8),
    key=jax.random.key(0),
    n_steps=256,
    p={"mu": jnp.asarray(0.4), "sigma": jnp.asarray(0.3)},
    save_at=SaveAt(steps=True),
)
```

## Algorithm and convergence

`EulerMaruyama` and `SRA1` are supported. At a consistent node,
Euler–Maruyama updates

$$
y_{n+1}=y_n+f(y_n,z_n,t_n)h+\sigma(y_n,z_n,t_n)\Delta W_n,
$$

followed by a root solve for $g(y_{n+1},z_{n+1},t_{n+1})=0$. Locally writing
the unique root as $z=Z(y,t)$ shows this is exactly Euler–Maruyama on the
reduced SDE — no Itô correction is missing, since `z` is reconstructed from
the constraint rather than advanced by a separate SDE. It has strong order
0.5 and weak order 1 under the usual regularity assumptions on the reduced
system (strong order 1 for additive noise). A per-node root error
$\epsilon$ should be $O(\sqrt h)$ or smaller to preserve strong order 0.5;
normal root tolerances are far tighter.

`SRA1` applies the Rößler additive-noise stochastic Runge–Kutta scheme to
the same reduced SDE: each step evaluates the drift at the node and at one
internal stage, with a root solve restoring consistency at the stage time
$t_n + \tfrac34 h$ and at the endpoint — two root solves per step. Its
strong order 1.5 requires the diffusion to be independent of the state; in
the SDAE setting, `z` depends on `y` through the constraint, so the
requirement is that `diffusion(y, z, t)` depends only on `t`. As in
`solve_sde`, the contract is documented, not runtime-checked, and the
endpoint diffusion evaluation reuses the current node's `(y, z, context)`
with the endpoint time. `Milstein` is deliberately not wired in: its
commutativity condition and derivative correction do not translate cleanly
through the implicit reduction $z = Z(y, t)$.

## Randomness, aux, and AD

Noise follows the `solve_sde` sampling contract via
`solver.sample_noise(y_0, key, n_steps, dt, dtype)`: Brownian increments
for `EulerMaruyama`, an independent $(\Delta W, \Delta Z)$ pair for `SRA1`.
Arrays use the draw shape `(n_steps,) + y_0.shape`; pytrees use one flat
draw partitioned in deterministic leaf order. A fixed key defines a fixed
path, so JVP/VJP with respect to `y_0` and `p` are pathwise derivatives
under common random numbers. The key is not differentiable.

`z_0` is a root guess, receives zero tangent, and selects a local root
branch. Algebraic solves use the same `LMRootSolver` configuration and
implicit-AD contract as deterministic DAEs — see
[Nonlinear-solve and AD contract](dae.md#nonlinear-solve-and-ad-contract):
residual-only stopping, `CONVERGED` with residual norm strictly below the
root `atol`, and nlls-gram's direct square `LU()` implicit derivative.

The algebraic function may return `(residual, algebraic_aux)`; that internal
context is passed to both drift and diffusion but is not stored. The drift
may return `(drift_value, saved_aux)`, and only that saved aux becomes
`sol.aux`: steps mode stores it at every consistent node, endpoint mode
evaluates it only at the final node. Its derivatives include both direct
parameter dependence and dependence through the implicit root. Invalid
algebraic context at initialization sets `ok=False` before any stochastic
step; invalid saved aux terminates at the previous consistent node (steps
mode) or zero-fills aux at the retained endpoint (endpoint mode).

`sol.num_root_solves` counts logical active root calls — for `SRA1`, a
stage and an endpoint root per step plus the initial consistency solve —
and `sol.num_root_steps` sums their LM updates; both have exact-zero
tangents.

Only `SaveAt(t_1=True)` and `SaveAt(steps=True)` are supported: stochastic
paths are rough, so deterministic dense interpolation between nodes would be
mathematically wrong. A failed root freezes the last consistent prefix, sets
`ok=False`, and pads the remaining static buffer with zero implicit
tangents, so masked lanes preserve successful JVPs/VJPs under `vmap`. For
that contract, pass `failure_ad_reference=(y_ref, z_ref, t_ref, p_ref)` at a
point where the residual, context, and saved-aux maps are finite and
differentiable — the same semantics as the
[DAE reference point](dae.md#nonlinear-solve-and-ad-contract).
