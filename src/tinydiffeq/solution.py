from dataclasses import dataclass

import jax


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Solution:
    """Result of ``solve_ode``/``solve_sde``.

    - ``ts``/``xs``: times and states in the shape dictated by ``SaveAt`` —
      scalar/endpoint for ``t_1``, ``(len(ts), ...)`` for ``ts``,
      ``(max_steps + 1, ...)`` for ``steps``.
    - ``ok``: scalar bool, True iff the integration reached ``t_1`` within the
      attempt budget. The package never poisons outputs; callers that want
      diverging residuals do ``jnp.where(sol.ok, sol.xs, jnp.inf)``.
    - ``num_accepted``: number of accepted steps (excluding the initial
      state).
    - ``accepted``: ``steps`` mode only (otherwise None): validity mask for
      the contiguous accepted-step prefix. Row 0 (the initial state) is
      always True, so ``accepted.sum() == num_accepted + 1``.
    """

    ts: jax.Array
    xs: jax.Array
    ok: jax.Array
    num_accepted: jax.Array
    accepted: jax.Array | None = None


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class DAESolution:
    """Result of :func:`tinydiffeq.solve_semi_explicit_dae`.

    ``ts`` and ``ys`` follow the same :class:`tinydiffeq.SaveAt` shape
    contract as :class:`Solution`. ``zs`` contains algebraic states satisfying
    the constraint at those times: requested observation times interpolate
    ``y`` and then solve the algebraic equation rather than interpolating
    ``z``. ``ok`` is true only when the integration reached ``t_1`` and every
    algebraic solve needed for the returned output converged.
    """

    ts: jax.Array
    ys: jax.Array
    zs: jax.Array
    ok: jax.Array
    num_accepted: jax.Array
    accepted: jax.Array | None = None
