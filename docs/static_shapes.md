# Static Shapes

JAX jits fixed-shape programs. An adaptive integrator is naturally
dynamic — the number of steps depends on the data — so something must give.
tinydiffeq's answer is a **bounded scan**: `solve_ode` always runs one
`lax.scan` of exactly `max_steps` iterations, whatever the controller does.

Each iteration attempts one step:

- an **accepted** attempt advances `(t, x)` and (for FSAL solvers) reuses the
  last stage as the next first stage;
- a **rejected** attempt leaves the state in place and retries with a smaller
  step;
- once `t_1` is reached, the remaining iterations **freeze**.

The raw internal scan buffer contains repeated rows for rejected and frozen
iterations. That buffer is an implementation detail used by interpolation;
step output compacts it into accepted rows plus tail padding.

The scan loop itself always has `max_steps` iterations, preserving static
shapes and reverse-mode AD. A scalar `lax.cond` nevertheless keeps the
expensive vector-field, solver-stage, and controller computations out of the
frozen tail, so a scalar solve's compute cost follows its attempted steps plus
cheap loop overhead. Under `vmap`, JAX may turn each lane's conditional into
selection; batched work can therefore continue until the slowest lane
finishes.

Fixed-step and adaptive integration share this single code path:
`ConstantStepSize` accepts every attempt, so `dt_0 = (t_1 - t_0)/n` with
`max_steps = n` reproduces a fixed grid exactly. The conditional has modest
overhead for very cheap solves whose budget is already exact; it pays off
when the budget has a padded tail or the vector field is expensive.

If the budget runs out before `t_1`, `sol.ok` is `False` and the outputs hold
the reached prefix. The package never poisons values; the caller decides:

```python
xs = jnp.where(sol.ok, sol.xs, jnp.inf)  # kernels-style rejection
```

## SaveAt is the shape contract

Exactly one of three modes:

### `SaveAt(t_1=True)` — endpoint only (default)

`sol.ts` is the reached time (equals `t_1` when `ok`), `sol.xs` the final
state.

### `SaveAt(ts=grid)` — interpolation onto a fixed grid

This is the answer to "adaptive steps vs static shapes". Internal steps
adapt freely; the output is cubic-Hermite interpolation onto your fixed
query grid, so each output leaf has shape
`(len(grid),) + corresponding_input_leaf.shape` **regardless of how many
steps the controller took**. Changing tolerances, initial conditions,
or curvature changes the internal knots but never the output shape — no
recompilation. A one-dimensional JAX/NumPy array or Python sequence is
accepted; times must be nondecreasing and within `[t_0, t_1]`, while repeated
times and omitted endpoints are allowed. Changing values without changing
the grid length does not recompile.

These are observation times, not mandatory internal stops. The adaptive
controller chooses exactly the same mesh regardless of the requested grid,
then the solver evaluates every requested point through dense interpolation.
Forcing exact internal landing times is a distinct feature and is not part of
this API.

The interpolation runs directly over the raw padded rows: duplicate knots
from rejections or the frozen tail form zero-width brackets, and the
bracketing `searchsorted` lands on the last duplicate at-or-before each
query, so no compaction pass is needed. Queries outside the knot span clamp
to the boundary values — in particular, when `ok` is `False`, queries beyond
the reached time return the last state (flat extrapolation) rather than
evaluating a cubic outside its bracket.

The interpolant is 4th-order accurate between 5th-order-accurate knots:
expect grid values slightly less accurate than the knots themselves, which
is the standard dense-output trade-off.

### `SaveAt(steps=True)` — accepted steps with padding

`max_steps + 1` rows including the initial state. Accepted internal steps are
gathered chronologically into a contiguous prefix; rejected attempts are not
returned. `sol.accepted` is the validity mask (`accepted[0]` is always
`True`, so `accepted.sum() == num_accepted + 1`). On a successful solve, the
endpoint is at index `num_accepted`.

- `fill="last"` (default) repeats the last valid time and state through the
  padded tail.
- `fill="inf"` fills only the invalid tail with `inf`.

If the attempt budget is exhausted, the same contract holds for the reached
accepted prefix, the tail repeats or masks its last state, and `sol.ok` is
`False`. No fake endpoint is inserted. Compaction is a stable fixed-size
gather; it performs no sorting.

## Pytree states

ODE, SDE, and DAE states may be arbitrary JAX pytrees, including registered
dataclasses. Each state contains at least one nonempty real floating array,
and all leaves use one dtype. Solver arithmetic is mapped directly
over leaves: states are not flattened or concatenated inside a time step.
Consequently, a single-array state retains the array execution path; the
pytree structure is resolved while tracing. A change to leaf values with the
same shapes and structure reuses a compilation, while changing the treedef or
a leaf shape requires a new compilation.

For every multi-row `SaveAt` mode the leading row dimension is added to each
leaf independently. `sol.accepted` is one shared mask for all leaves.

## Why one compilation, precisely

- The scan length `max_steps` is static; nothing else about the loop depends
  on data shapes.
- Tolerances and PI coefficients (`IController(...)` / `PIController(...)`),
  `dt_0`, `t_0`, `t_1`, `x_0`,
  `args`, `p`, and `SaveAt.ts` are pytree **data leaves**. Only genuine
  structure — the solver type, `SaveAt` mode, `fill`, `max_steps`, the
  functions themselves — is static.

An omitted tolerance or `dt_min` is represented by `None`, so switching a
jitted call between automatic and explicit values changes the controller
pytree structure and compiles once for each policy. Changing the numeric
values of already-explicit controller fields does not recompile.

So this compiles once:

```python
@jax.jit
def run(x_0, dt_0, controller, args):
    return solve_ode(f, Tsit5(), 0.0, 1.0, x_0, args=args, dt_0=dt_0,
                     controller=controller, max_steps=128,
                     save_at=SaveAt(steps=True))
```

across different curvatures (different accepted counts), tolerances, initial
steps, and initial conditions — pinned by `tests/test_recompile.py` with
`_cache_size() == 1` assertions.
