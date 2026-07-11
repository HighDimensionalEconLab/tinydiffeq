from dataclasses import dataclass, field

import jax
from jax.typing import ArrayLike


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SaveAt:
    """What ``solve_ode``/``solve_sde`` return. Exactly one mode must be set.

    - ``t_1=True``: the endpoint only (the default in the solve functions).
    - ``ts=grid``: dense interpolation of the internal steps onto a fixed,
      sorted query grid in ``[t_0, t_1]``. Explicit methods use cubic Hermite;
      Rodas5P uses its stiff-aware fourth-order extension. Output shape is
      ``(len(ts), ...)`` regardless of how many internal steps the controller
      takes, so changing curvature never changes shapes or recompiles.
      ``ts`` is a data leaf; a different grid of the same length retraces
      nothing. ODE, deterministic DAE, and linear exponential solves only.
    - ``steps=True``: the initial state and accepted internal steps as a
      chronological prefix of a ``max_steps + 1`` buffer. Rejected attempts
      are omitted. ``fill="last"`` (default) pads the tail with the final
      valid row; ``fill="inf"`` pads the tail with ``inf``. The returned
      ``Solution.accepted`` mask distinguishes the valid prefix from padding.

    ``fill`` only applies to ``steps=True``.
    """

    t_1: bool = field(default=False, metadata=dict(static=True))
    ts: ArrayLike | None = None
    steps: bool = field(default=False, metadata=dict(static=True))
    fill: str = field(default="last", metadata=dict(static=True))

    def __post_init__(self):
        modes = int(bool(self.t_1)) + int(self.ts is not None) + int(bool(self.steps))
        if modes != 1:
            raise ValueError(
                "SaveAt requires exactly one of t_1=True, ts=..., steps=True"
            )
        if self.fill not in ("last", "inf"):
            raise ValueError('SaveAt fill must be "last" or "inf"')
