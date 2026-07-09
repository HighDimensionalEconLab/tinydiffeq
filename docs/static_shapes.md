# Static Shapes

JAX jits fixed-shape programs. An adaptive integrator is naturally
dynamic — the number of steps depends on the data — so something must give.
tinydiffeq's answer is a **bounded scan**: `solve_ode` always runs one
`lax.scan` of exactly `max_steps` iterations, whatever the controller does.

Each iteration attempts one step:

- an **accepted** attempt advances `(t, x)` and (for FSAL solvers) reuses the
  last stage as the next first stage;
- a **rejected** attempt leaves the state in place and retries with a smaller
  step — the emitted row duplicates the previous state;
- once `t1` is reached, the remaining iterations **freeze** and keep emitting
  duplicates of the final state.

Fixed-step and adaptive integration share this single code path:
`ConstantStepSize` accepts every attempt, so `dt0 = (t1 - t0)/n` with
`max_steps = n` reproduces a fixed grid exactly. The masking overhead is
noise next to the vector-field evaluations.

If the budget runs out before `t1`, `sol.ok` is `False` and the outputs hold
the reached prefix. The package never poisons values; the caller decides:

```python
xs = jnp.where(sol.ok, sol.xs, jnp.inf)  # kernels-style rejection
```

## SaveAt is the shape contract

Exactly one of three modes:

### `SaveAt(t1=True)` — endpoint only (default)

`sol.ts` is the reached time (equals `t1` when `ok`), `sol.xs` the final
state.

### `SaveAt(ts=grid)` — interpolation onto a fixed grid

This is the answer to "adaptive steps vs static shapes". Internal steps
adapt freely; the output is cubic-Hermite interpolation onto your fixed
query grid, so `sol.xs.shape == (len(grid),) + x0.shape` **regardless of how
many steps the controller took**. Changing tolerances, initial conditions,
or curvature changes the internal knots but never the output shape — no
recompilation. This is the same approach diffrax takes.

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

### `SaveAt(steps=True)` — raw padded attempt rows

`max_steps + 1` rows including the initial state, with the per-row
`sol.accepted` mask (`accepted[0]` is always `True`, so
`accepted.sum() == num_accepted + 1`).

- `fill="last"` (default) keeps the duplicate rows from rejections and the
  frozen tail. Downstream least-squares residuals that vanish at every state
  tolerate duplicates as harmless repeated rows — this is byte-for-byte what
  collocation-style consumers need.
- `fill="inf"` overwrites every non-accepted row of `ts` and `xs` with
  `inf`, diffrax-style masking. Note that unlike diffrax's compacted
  buffers, tinydiffeq's rows are positional: a **mid-trajectory** rejection
  row gets inf'd too, so `ts` is not monotone in this mode — use `accepted`
  to recover the trajectory.

## Why one compilation, precisely

- The scan length `max_steps` is static; nothing else about the loop depends
  on data shapes.
- Tolerances (`IController(rtol=..., atol=...)`), `dt0`, `t0`, `t1`, `x0`,
  `args`, `p`, and `SaveAt.ts` are pytree **data leaves**. Only genuine
  structure — the solver type, `SaveAt` mode, `fill`, `max_steps`, the
  functions themselves — is static.

So this compiles once:

```python
@jax.jit
def run(x0, dt0, controller, args):
    return solve_ode(f, Tsit5(), 0.0, 1.0, x0, args=args, dt0=dt0,
                     controller=controller, max_steps=128,
                     saveat=SaveAt(steps=True))
```

across different curvatures (different accepted counts), tolerances, initial
steps, and initial conditions — pinned by `tests/test_recompile.py` with
`_cache_size() == 1` assertions.
