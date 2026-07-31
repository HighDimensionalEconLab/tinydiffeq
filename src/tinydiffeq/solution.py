from dataclasses import dataclass
from typing import Any

import jax


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Solution:
    """Result of ``solve_ode``/``solve_sde``/``solve_linear_ode``.

    - ``ts``/``xs``: times and states in the shape dictated by ``SaveAt`` —
      scalar/endpoint for ``t_1``; for multi-row modes, every state pytree leaf
      receives a leading ``len(ts)`` or ``max_steps + 1`` axis.
    - ``ok``: scalar bool, True iff the integration reached ``t_1`` within the
      attempt budget and every required saved output was valid. The package
      never poisons outputs; callers that want diverging residuals map
      ``jnp.where(sol.ok, x, jnp.inf)`` over leaves.
    - ``num_accepted``: number of accepted steps (excluding the initial
      state).
    - ``num_steps``: scalar integer array counting logical attempted steps,
      including rejected attempts. Public solver outputs always populate it;
      the optional default only preserves direct-construction compatibility.
    - ``accepted``: ``steps`` mode only (otherwise None): validity mask for
      the contiguous accepted-step prefix. Row 0 (the initial state) is
      always True, so ``accepted.sum() == num_accepted + 1``.
    - ``aux``: optional floating pytree returned by the differential field
      (or stochastic drift), with the same leading saved-time axis as
      ``xs``. It is ``None`` when the field has no auxiliary output.
    """

    ts: jax.Array
    xs: Any
    ok: jax.Array
    num_accepted: jax.Array
    accepted: jax.Array | None = None
    aux: Any = None
    num_steps: jax.Array | None = None


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class DAESolution:
    """Result of the deterministic or stochastic semi-explicit DAE solver.

    ``ts`` and ``ys`` follow the same :class:`tinydiffeq.SaveAt` shape
    contract as :class:`Solution`. ``zs`` holds algebraic states and ``aux``
    holds the optional differential-field auxiliary-output pytree. An
    algebraic auxiliary output is internal context passed into the
    differential field; it is not stored directly. RK4/Tsit5
    internal-step and endpoint values are evaluated at converged roots.
    Rodas5P instead satisfies the algebraic equation to integration accuracy
    after its initial consistency root. Requested values are dense
    interpolants and need not satisfy the constraint exactly. ``ok`` is true
    only when initialization, all required stage and saved-output operations
    succeeded, and the integration reached ``t_1``. ``num_root_solves`` counts
    logical active algebraic root calls, including failed calls and the initial
    consistency solve. ``num_root_steps`` sums their nonlinear LM update steps.
    ``num_steps`` counts logical attempted integration steps, including rejected
    attempts, with the same semantics as :class:`Solution`.
    All three are path diagnostics with exact-zero tangents. Under ``vmap``, masked
    lanes may still execute physically even though they do not increment these
    logical per-lane counters.
    """

    ts: jax.Array
    ys: Any
    zs: Any
    ok: jax.Array
    num_accepted: jax.Array
    accepted: jax.Array | None = None
    aux: Any = None
    num_steps: jax.Array | None = None
    num_root_solves: jax.Array | None = None
    num_root_steps: jax.Array | None = None
