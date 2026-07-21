# Semi-Explicit Index-1 SDAEs

`solve_semi_explicit_sdae` integrates the Itô system

$$
dy = f(y,z,t)\,dt + \sigma(y,z,t)\,dW,
\qquad 0=g(y,z,t),
$$

with fixed-step Euler–Maruyama and diagonal noise in the differential state.
The algebraic Jacobian $g_z$ must be square and nonsingular along the chosen
root branch. Arrays and pytrees are supported for both `y` and `z`.

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

At a consistent node, the update is

$$
y_{n+1}=y_n+f(y_n,z_n,t_n)h+\sigma(y_n,z_n,t_n)\Delta W_n,
$$

followed by a root solve for
$g(y_{n+1},z_{n+1},t_{n+1})=0$. Locally writing the unique root as
$z=Z(y,t)$ shows that this is exactly Euler–Maruyama on the reduced SDE. No
Itô correction is missing: `z` is reconstructed from the constraint rather
than advanced by a separate SDE.

The method has strong order 0.5 and weak order 1 when $g_z$ is uniformly
nonsingular near the path, the reduced drift and diffusion are globally
Lipschitz with linear growth, and the usual additional smoothness assumptions
for weak convergence hold. Additive noise gives strong order 1. A per-node
root error $\epsilon$ should be $O(\sqrt h)$ or smaller to preserve strong
order 0.5, and $O(h)$ or smaller to preserve weak order 1; normal root
tolerances are typically much tighter.

## Randomness, aux, and AD

The Brownian increments use the same sampling contract as `solve_sde`.
Arrays use the draw shape `(n_steps,) + y_0.shape`; pytrees use one flat draw
partitioned in deterministic leaf order. A fixed key defines a fixed path,
so JVP/VJP with respect to `y_0` and `p` are pathwise derivatives under common
random numbers. The key is not differentiable.

`z_0` is a root guess, receives zero tangent, and selects a local root branch.
Algebraic solves use the same nlls-gram 2.4 `LMRootSolver` configuration and
implicit-AD contract as deterministic DAEs; see
[Nonlinear-solve and AD contract](dae.md#nonlinear-solve-and-ad-contract).
The algebraic function may return `(residual, algebraic_aux)`; that internal
context is passed to both drift and diffusion but is not stored. The drift may
return `(drift_value, saved_aux)`, and only that saved aux becomes `sol.aux`.
Steps mode stores it at every consistent node; endpoint mode evaluates it only
at the final node. Its derivatives include both direct parameter dependence
and dependence through the implicit root. See
[Auxiliary Outputs](aux.md) for all contract variations and explicit flags.
Invalid algebraic context at initialization sets `ok=False` before any
stochastic step. In steps mode, invalid saved aux terminates at the previous
consistent node. In endpoint mode, invalid final saved aux retains the
endpoint state, returns zero aux, and sets `ok=False`.

Only `SaveAt(t_1=True)` and `SaveAt(steps=True)` are supported. Stochastic
paths are rough, so deterministic dense interpolation between nodes would be
mathematically wrong. A root failure freezes the last consistent prefix,
sets `ok=False`, and pads the remaining static buffer. Failed roots have zero
implicit tangents, and aux at a failed initial root is zero-filled, so masked
lanes can preserve successful JVPs or VJPs under `vmap`. For that contract,
pass `failure_ad_reference=(y_ref, z_ref, t_ref, p_ref)` at a point where the
residual, context, and saved-aux maps are finite and differentiable. The
reference is used only to linearize inactive lanes before zeroing their
tangents. Without one,
an all-ones best-effort default is used and a batch containing failures has no
gradient guarantee. Outputs and gradients from the failed lane are not a
valid solution.
