from dataclasses import dataclass
from typing import Any

import jax


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Solution:
    """Result of ``solve_ode``/``solve_sde``/``solve_linear_ode``.

    ``ts``/``xs`` hold times and states in the shape dictated by ``SaveAt``.
    ``ok`` is a scalar bool: the integration reached ``t_1`` and every
    required saved output was valid. Outputs are never poisoned; callers that
    want diverging values map ``jnp.where(sol.ok, x, jnp.inf)`` over leaves.
    ``num_accepted`` counts accepted steps, ``num_steps`` counts logical
    attempts including rejections, ``accepted`` masks the valid prefix in
    ``steps`` mode (row 0 is always True), and ``aux`` holds the field's
    saved auxiliary pytree with the same leading saved-time axis as ``xs``.
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
    """Result of the deterministic or stochastic semi-explicit DAE solvers.

    ``ts``/``ys``/``zs`` follow the :class:`Solution` shape contract, with
    ``zs`` the algebraic states. Explicit-method saved values sit at
    converged roots; Rodas5P satisfies the constraint to integration accuracy
    after its initial consistency root, and requested-grid interpolants need
    not satisfy it exactly. ``num_root_solves`` counts logical active root
    calls (including failures and the initial consistency solve) and
    ``num_root_steps`` sums their LM update steps; like ``num_steps``, both
    are path diagnostics with exact-zero tangents.
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
