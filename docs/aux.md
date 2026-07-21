# Auxiliary Outputs

Auxiliary outputs let a model expose quantities it already computes without
adding them to the dynamical state. They are ordinary JAX pytrees, are saved
with the solution, and participate in JVP and VJP. The no-aux code path remains
available explicitly with `has_aux=False`.

## ODE contract

An ODE field may return only its derivative,

```python
def f(x, t, args, p):
    return dx
```

or pair the derivative with saved output:

```python
def f(x, t, args, p):
    return dx, {"flow": flow, "moment": moment}
```

The first item must have the state pytree structure and dtype. Saved aux must
be a nonempty pytree of nonempty real floating arrays; its leaves may have
different floating dtypes. The result is available as `sol.aux` with the same
leading saved-time axis as `sol.xs`.

`has_aux=None` (the default) performs one `jax.eval_shape` trace to detect the
contract. A complete field output matching the state pytree takes precedence,
so a two-item tuple state is not misclassified as `(dx, aux)`. Set
`has_aux=True` to require aux, or `has_aux=False` to skip detection and select
the minimal value-only path.

## DAE contracts

For a semi-explicit DAE, saved output belongs to the differential field:

```python
def f(y, z, t, args, p):
    return dy, saved_aux


def g(y, z, t, args, p):
    return residual
```

The algebraic function may additionally expose internal context that avoids
recomputing quantities already formed while evaluating the residual:

```python
def g(y, z, t, args, p):
    return residual, algebraic_aux


def f(y, z, t, args, p, algebraic_aux):
    return dy, saved_aux
```

All four combinations are supported: neither aux, saved aux only, algebraic
aux only, or both. When algebraic aux is present, the differential function
must use the full six-argument signature. `has_algebraic_aux=None` detects the
form; explicit `False` avoids that trace.

Algebraic aux is internal context. It is passed to `f`, included in the
implicit derivative path, and ignored by the nonlinear solver's residual
interface. It is not independently stored or interpolated. It may be a
nonempty pytree of bool, integer, real, or complex arrays; every inexact leaf
must be finite. This broader dtype contract is useful for cached masks and
indices that should not become solution output.

Only the differential field's real-floating `saved_aux` becomes `sol.aux`.
This separation keeps one unambiguous output to differentiate and interpolate:
the algebraic function shares work with the dynamics, while the differential
function decides what users retain.

## SDE and SDAE contracts

For an SDE, only the drift owns saved aux:

```python
def drift(x, t, args, p):
    return drift_value, saved_aux


def diffusion(x, t, args, p):
    return diffusion_value
```

For an SDAE, algebraic aux is passed to both stochastic fields, and the drift
still owns saved aux:

```python
def g(y, z, t, args, p):
    return residual, algebraic_aux


def drift(y, z, t, args, p, algebraic_aux):
    return drift_value, saved_aux


def diffusion(y, z, t, args, p, algebraic_aux):
    return diffusion_value
```

The fixed random key defines the path for both primal evaluation and AD.
Aux tangents and cotangents are therefore pathwise derivatives under common
random numbers. SDE/SDAE aux is saved only at actual grid nodes;
`SaveAt(ts=...)` remains unsupported because deterministic interpolation is
not valid for rough paths.

## Deterministic interpolation and AD

`SaveAt(t_1=True)` evaluates saved aux only at the final state.
`SaveAt(steps=True)` stores it at the initial and accepted nodes and applies
the same prefix/padding mask as the state.

For `SaveAt(ts=grid)`, ODE and root-restored DAE aux uses normalized cubic
Hermite interpolation. Endpoint aux slopes are JVPs along the full solution
velocity, so they include direct parameter dependence and indirect dependence
through the state and any implicit algebraic root. Rodas5P obtains endpoint
state velocities from its published stiff-aware continuous extension, then
uses the same JVP construction for aux. No requested-time algebraic root or
aux recomputation is performed. See [Rodas5P](rodas5p.md) and
[Semi-Explicit DAEs](dae.md) for the dense-output details and links to SciML's
implementation.

Ordinary JAX transformations compose through every saved or interpolated aux
leaf. This includes `jax.jvp`, reverse-mode VJP/`jax.grad`, `vmap`, and
reverse-over-forward. Adaptive accept/reject decisions and mesh selection
retain the package's frozen-controller derivative convention.

## Failure behavior

Every saved aux leaf and every inexact algebraic-aux leaf must remain finite.
Invalid algebraic context at initialization fails before time stepping.
Invalid algebraic context at a later required stage/node, or invalid saved aux
in a prefix-saving mode, terminates at the previous accepted node.
Endpoint-only saved aux is evaluated after integration; if it is invalid, the
endpoint state is retained, aux is zero-filled, and `sol.ok` is false.

For batched differentiation where inactive lanes may leave the model domain,
`failure_ad_reference` supplies a finite point used only for safe
linearization. ODE/SDE references have the form `(x, t, p)`; DAE/SDAE
references have `(y, z, t, p)`. Without one, tinydiffeq uses an all-ones
best-effort reference. For DAE/SDAE roots, it is substituted before nlls only
after a lane is already inactive. A newly attempted root and its model context
must be JVP-safe at the actual initial point; the reference is not a post-hoc
replacement for an active failure.
