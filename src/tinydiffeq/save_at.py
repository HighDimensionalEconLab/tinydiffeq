from dataclasses import dataclass, field

import jax
from jax.typing import ArrayLike


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SaveAt:
    """What the solve functions return. Exactly one mode must be set.

    ``t_1=True`` (the solver default) returns the endpoint only. ``ts=grid``
    interpolates the internal steps onto a fixed query grid — output shape is
    ``(len(ts), ...)`` however many steps the controller takes, and ``ts`` is
    a data leaf; with ``exact=True`` an explicit constant-step ODE instead
    gathers states at queries that must coincide with realized knots.
    ``steps=True`` returns the initial state and accepted steps as the valid
    prefix of a ``max_steps + 1`` buffer, padded with the last valid row
    (``fill="last"``) or ``inf`` (``fill="inf"``) and masked by
    ``Solution.accepted``.
    """

    t_1: bool = field(default=False, metadata=dict(static=True))
    ts: ArrayLike | None = None
    steps: bool = field(default=False, metadata=dict(static=True))
    fill: str = field(default="last", metadata=dict(static=True))
    exact: bool = field(default=False, metadata=dict(static=True))

    def __post_init__(self):
        modes = int(bool(self.t_1)) + int(self.ts is not None) + int(bool(self.steps))
        if modes != 1:
            raise ValueError(
                "SaveAt requires exactly one of t_1=True, ts=..., steps=True"
            )
        if self.fill not in ("last", "inf"):
            raise ValueError('SaveAt fill must be "last" or "inf"')
        if self.exact and self.ts is None:
            raise ValueError("SaveAt exact=True requires ts=...")
