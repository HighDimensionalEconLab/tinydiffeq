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
class BVPSolution:
    """Result of ``solve_bvp``.

    Arrays are padded to the static ``max_nodes``: the ``t`` tail repeats the
    right endpoint and the ``y``/``yp`` tails repeat the last active row, so
    ``hermite_interpolate(ts, sol.t, sol.y, sol.yp)`` evaluates exactly the C1
    cubic spline scipy's ``solve_bvp`` returns and ``hermite_derivative`` its
    derivative. ``z`` holds the solved unknown parameters (``None`` when the
    problem has none), ``rms_residuals`` is zero on inactive intervals,
    ``num_nodes`` counts active mesh nodes, and ``num_iterations`` is scipy's
    ``niter``. ``status`` uses scipy's codes (0 converged, 1 ``max_nodes``
    exceeded, 2 singular Jacobian, 3 boundary-condition tolerance unsatisfied)
    and ``ok`` is ``status == 0``; a failed status returns the last iterate,
    which may be non-finite. Under AD only ``y``, ``yp``, ``z``, and ``aux``
    carry tangents with respect to ``p``; every other field is
    differentiation-inert with exact-zero tangents.
    """

    t: jax.Array
    y: Any
    yp: Any
    z: Any
    rms_residuals: jax.Array
    num_nodes: jax.Array
    num_iterations: jax.Array
    status: jax.Array
    ok: jax.Array
    aux: Any = None


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
