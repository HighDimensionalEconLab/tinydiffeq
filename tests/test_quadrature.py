import jax
import jax.numpy as jnp

from tinydiffeq import cumulative_trapezoid


def nonuniform_grid():
    key = jax.random.PRNGKey(0)
    increments = jax.random.uniform(key, (40,), minval=0.02, maxval=0.3)
    return jnp.concatenate([jnp.zeros(1), jnp.cumsum(increments)])


def test_integral_of_cos_is_sin():
    ts = nonuniform_grid()
    integral, values = cumulative_trapezoid(jnp.cos, ts, substeps=8)
    assert integral[0] == 0.0
    assert jnp.array_equal(values, jnp.cos(ts))
    assert jnp.max(jnp.abs(integral - (jnp.sin(ts) - jnp.sin(ts[0])))) < 1e-3


def test_substeps_reduce_error():
    ts = nonuniform_grid()
    exact = jnp.sin(ts) - jnp.sin(ts[0])
    errors = []
    for substeps in (1, 4, 16):
        integral, _ = cumulative_trapezoid(jnp.cos, ts, substeps=substeps)
        errors.append(float(jnp.max(jnp.abs(integral - exact))))
    assert errors[0] > errors[1] > errors[2]
    assert errors[2] < errors[0] / 100.0  # trapezoid is second order in width


def test_parity_with_kernels_integrate_time_derivative():
    # verbatim copy of the kernels helper this generalizes
    def integrate_time_derivative(derivative_fn, t, substeps):
        derivative_at_t = jax.vmap(derivative_fn)(t)

        def interval_increment(left, right):
            nodes = jnp.linspace(left, right, substeps + 1)
            values = jax.vmap(derivative_fn)(nodes)
            widths = nodes[1:] - nodes[:-1]
            return jnp.sum(0.5 * widths[:, None] * (values[:-1] + values[1:]), axis=0)

        increments = jax.vmap(interval_increment)(t[:-1], t[1:])
        integral = jnp.concatenate(
            [
                jnp.zeros((1, increments.shape[1]), dtype=t.dtype),
                jnp.cumsum(increments, axis=0),
            ]
        )
        return integral, derivative_at_t

    def g(t):
        return jnp.asarray([jnp.cos(t), t**2])

    ts = nonuniform_grid()
    for substeps in (1, 3, 8):
        ours_int, ours_val = cumulative_trapezoid(g, ts, substeps=substeps)
        ref_int, ref_val = integrate_time_derivative(g, ts, substeps)
        assert jnp.array_equal(ours_int, ref_int)
        assert jnp.array_equal(ours_val, ref_val)


def test_scalar_and_matrix_output_shapes():
    ts = jnp.linspace(0.0, 1.0, 9)
    integral, values = cumulative_trapezoid(jnp.sin, ts)
    assert integral.shape == ts.shape and values.shape == ts.shape

    def g(t):
        return jnp.asarray([[t, 1.0], [0.0, t**2]])

    integral, values = cumulative_trapezoid(g, ts, substeps=2)
    assert integral.shape == (9, 2, 2)
    assert values.shape == (9, 2, 2)


def test_differentiable():
    ts = jnp.linspace(0.0, 2.0, 15)

    def total(a):
        integral, _ = cumulative_trapezoid(lambda t: jnp.cos(a * t), ts, substeps=8)
        return integral[-1]

    grad = jax.grad(total)(jnp.asarray(1.3))
    exact = jax.grad(lambda a: jnp.sin(a * 2.0) / a)(jnp.asarray(1.3))
    assert jnp.abs(grad - exact) < 1e-4
