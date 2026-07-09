import jax
import jax.numpy as jnp


def cumulative_trapezoid(g, ts, *, substeps=1):
    """Cumulative composite-trapezoid integral of a time-only ``g(t)`` on the
    (possibly nonuniform) sorted grid ``ts``.

    Each grid interval is subdivided into ``substeps`` uniform panels, so the
    quadrature error shrinks with ``substeps`` without changing the output
    grid. Returns ``(integral, values)`` where ``integral[k]`` approximates
    the integral of ``g`` from ``ts[0]`` to ``ts[k]`` (``integral[0] = 0``)
    and ``values = g(ts)``; ``g`` may return any array shape, which is
    appended to the leading grid axis.
    """
    if substeps < 1:
        raise ValueError("substeps must be at least 1")
    values = jax.vmap(g)(ts)

    def interval_increment(left, right):
        nodes = jnp.linspace(left, right, substeps + 1)
        node_values = jax.vmap(g)(nodes)
        widths = nodes[1:] - nodes[:-1]
        widths = widths.reshape(widths.shape + (1,) * (node_values.ndim - 1))
        return jnp.sum(0.5 * widths * (node_values[:-1] + node_values[1:]), axis=0)

    increments = jax.vmap(interval_increment)(ts[:-1], ts[1:])
    integral = jnp.concatenate(
        [jnp.zeros_like(increments[:1]), jnp.cumsum(increments, axis=0)]
    )
    return integral, values
