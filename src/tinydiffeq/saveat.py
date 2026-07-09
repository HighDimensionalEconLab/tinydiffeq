from dataclasses import dataclass, field

import jax


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SaveAt:
    """What ``solve_ode``/``solve_sde`` return. Exactly one mode must be set.

    - ``t1=True``: the endpoint only (the default in the solve functions).
    - ``ts=grid``: cubic-Hermite interpolation of the internal steps onto a
      fixed, sorted query grid in ``[t0, t1]``. Output shape is
      ``(len(ts), ...)`` regardless of how many internal steps the controller
      takes, so changing curvature never changes shapes or recompiles.
      ``ts`` is a data leaf; a different grid of the same length retraces
      nothing. ODE only.
    - ``steps=True``: the raw padded attempt rows, ``max_steps + 1`` of them
      including the initial state. Rejected attempts and post-horizon frozen
      iterations duplicate the previous row. ``fill="last"`` (default) keeps
      those duplicates; ``fill="inf"`` overwrites every non-accepted row
      (including mid-trajectory rejection rows) with ``inf``, diffrax-style.

    ``fill`` only applies to ``steps=True``.
    """

    t1: bool = field(default=False, metadata=dict(static=True))
    ts: jax.Array | None = None
    steps: bool = field(default=False, metadata=dict(static=True))
    fill: str = field(default="last", metadata=dict(static=True))

    def __post_init__(self):
        modes = int(bool(self.t1)) + int(self.ts is not None) + int(bool(self.steps))
        if modes != 1:
            raise ValueError(
                "SaveAt requires exactly one of t1=True, ts=..., steps=True"
            )
        if self.fill not in ("last", "inf"):
            raise ValueError('SaveAt fill must be "last" or "inf"')
