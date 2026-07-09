from dataclasses import dataclass

import jax


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Solution:
    """Result of ``solve_ode``/``solve_sde``.

    - ``ts``/``xs``: times and states in the shape dictated by ``SaveAt`` —
      scalar/endpoint for ``t1``, ``(len(ts), ...)`` for ``ts``,
      ``(max_steps + 1, ...)`` for ``steps``.
    - ``ok``: scalar bool, True iff the integration reached ``t1`` within the
      attempt budget. The package never poisons outputs; callers that want
      diverging residuals do ``jnp.where(sol.ok, sol.xs, jnp.inf)``.
    - ``num_accepted``: number of accepted steps (excluding the initial
      state).
    - ``accepted``: ``steps`` mode only (otherwise None): per-row bool mask,
      True for rows holding a newly computed state. Row 0 (the initial state)
      is always True, so ``accepted.sum() == num_accepted + 1``.
    """

    ts: jax.Array
    xs: jax.Array
    ok: jax.Array
    num_accepted: jax.Array
    accepted: jax.Array | None = None
