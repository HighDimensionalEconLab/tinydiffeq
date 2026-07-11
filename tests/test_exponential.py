import jax
import jax.numpy as jnp
import pytest

from tinydiffeq import (
    AdaptiveKrylovExponential,
    DenseExponential,
    KrylovExponential,
    SaveAt,
    jvp_linear_ode,
    solve_linear_ode,
    vjp_linear_ode,
)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize(
    "method", [DenseExponential(), KrylovExponential(krylov_dim=3)]
)
def test_linear_exponential_array_endpoint_and_grid(dtype, method):
    operator = jnp.asarray(
        [[-0.4, 0.2, 0.0], [0.1, -0.7, 0.3], [0.0, 0.2, -0.5]], dtype
    )
    x_0 = jnp.asarray([1.0, -0.2, 0.4], dtype)
    times = jnp.linspace(jnp.asarray(0.0, dtype), jnp.asarray(1.3, dtype), 9)
    endpoint = solve_linear_ode(operator, method, 0.0, 1.3, x_0)
    trajectory = solve_linear_ode(
        operator, method, 0.0, 1.3, x_0, save_at=SaveAt(ts=times)
    )
    expected = jax.scipy.linalg.expm(jnp.asarray(1.3, dtype) * operator) @ x_0
    expected_trajectory = jax.vmap(
        lambda time: jax.scipy.linalg.expm(time * operator) @ x_0
    )(times)
    tolerance = 3e-5 if dtype == jnp.float32 else 2e-11
    assert bool(endpoint.ok) and bool(trajectory.ok)
    assert jnp.allclose(endpoint.xs, expected, atol=tolerance)
    assert jnp.allclose(trajectory.xs, expected_trajectory, atol=tolerance)


@pytest.mark.parametrize(
    "method", [DenseExponential(), KrylovExponential(krylov_dim=4)]
)
def test_callable_operator_preserves_probability_pytree(method):
    matrix = jnp.asarray(
        [
            [-0.7, 0.2, 0.1, 0.0],
            [0.3, -0.6, 0.0, 0.2],
            [0.1, 0.0, -0.4, 0.3],
            [0.0, 0.2, 0.1, -0.5],
        ]
    )

    def flatten(state):
        return jnp.concatenate([state["slow"], state["fast"]["value"]])

    def unravel(flat):
        return {"slow": flat[:2], "fast": {"value": flat[2:]}}

    def operator(state):
        return unravel(matrix @ flatten(state))

    x_0 = unravel(jnp.asarray([0.2, 0.3, -0.1, 0.5]))
    solution = solve_linear_ode(operator, method, 0.0, 0.8, x_0)
    expected = jax.scipy.linalg.expm(0.8 * matrix) @ flatten(x_0)
    assert bool(solution.ok)
    assert jax.tree.structure(solution.xs) == jax.tree.structure(x_0)
    assert jnp.allclose(flatten(solution.xs), expected, atol=2e-10)


def test_dense_and_krylov_jvp_vjp_through_initial_state_and_operator_parameter():
    base = jnp.asarray([[-0.8, 0.1, 0.0], [0.4, -0.5, 0.2], [0.0, 0.3, -0.6]])
    direction = jnp.asarray([[0.1, -0.2, 0.0], [0.0, 0.05, 0.1], [-0.1, 0.0, 0.2]])
    x_0 = jnp.asarray([0.3, -0.2, 0.7])
    tangent_0 = jnp.asarray([0.2, 0.1, -0.3])
    weights = jnp.asarray([0.4, -0.7, 0.2])

    def endpoint(method, parameter, initial):
        matrix = base + parameter * direction

        def operator(state):
            return matrix @ state

        return solve_linear_ode(operator, method, 0.0, 0.9, initial).xs

    def dense(parameter, initial):
        return endpoint(DenseExponential(), parameter, initial)

    def krylov(parameter, initial):
        return endpoint(KrylovExponential(krylov_dim=3), parameter, initial)

    primals = (jnp.asarray(0.4), x_0)
    tangents = (jnp.asarray(0.6), tangent_0)
    dense_value, dense_tangent = jax.jvp(dense, primals, tangents)
    krylov_value, krylov_tangent = jax.jvp(krylov, primals, tangents)
    dense_gradient = jax.grad(lambda p: weights @ dense(p, x_0))(primals[0])
    krylov_gradient = jax.grad(lambda p: weights @ krylov(p, x_0))(primals[0])
    dense_initial_gradient = jax.grad(
        lambda initial: weights @ dense(primals[0], initial)
    )(x_0)
    krylov_initial_gradient = jax.grad(
        lambda initial: weights @ krylov(primals[0], initial)
    )(x_0)
    assert jnp.allclose(krylov_value, dense_value, atol=2e-10)
    assert jnp.allclose(krylov_tangent, dense_tangent, atol=2e-9)
    assert jnp.allclose(krylov_gradient, dense_gradient, atol=2e-9)
    assert jnp.allclose(krylov_initial_gradient, dense_initial_gradient, atol=2e-9)


def test_krylov_query_grid_jvp_vjp_and_vmap():
    operator = jnp.asarray([[-0.5, 0.2], [0.3, -0.4]])
    times = jnp.linspace(0.0, 1.0, 7)
    weights = jnp.linspace(-0.3, 0.7, 14).reshape(7, 2)

    def trajectory(method, initial):
        return solve_linear_ode(
            operator, method, 0.0, 1.0, initial, save_at=SaveAt(ts=times)
        ).xs

    initial = jnp.asarray([0.6, -0.1])
    tangent = jnp.asarray([0.2, 0.4])

    def dense(value):
        return trajectory(DenseExponential(), value)

    def krylov(value):
        return trajectory(KrylovExponential(krylov_dim=2), value)

    dense_value, dense_tangent = jax.jvp(dense, (initial,), (tangent,))
    krylov_value, krylov_tangent = jax.jvp(krylov, (initial,), (tangent,))
    dense_gradient = jax.grad(lambda value: jnp.vdot(weights, dense(value)))(initial)
    krylov_gradient = jax.grad(lambda value: jnp.vdot(weights, krylov(value)))(initial)
    batch = jnp.asarray([[1.0, 0.0], [0.0, 1.0], initial])
    batched = jax.jit(jax.vmap(krylov))(batch)
    assert jnp.allclose(krylov_value, dense_value, atol=2e-10)
    assert jnp.allclose(krylov_tangent, dense_tangent, atol=2e-9)
    assert jnp.allclose(krylov_gradient, dense_gradient, atol=2e-9)
    assert batched.shape == (3, 7, 2)


def test_one_pass_krylov_traced_jvp_vjp_match_dense():
    matrix = jnp.asarray([[-0.8, 0.4, 0.0], [0.1, -0.6, 0.3], [0.2, 0.0, -0.5]])
    initial = jnp.asarray([0.3, -0.2, 0.7])
    tangent = jnp.asarray([0.1, 0.4, -0.3])
    cotangent = jnp.asarray([-0.2, 0.5, 0.6])
    method = KrylovExponential(krylov_dim=3, reorthogonalization_passes=1)

    def endpoint(state):
        return solve_linear_ode(matrix, method, 0.0, 0.9, state).xs

    value, output_tangent = jax.jvp(endpoint, (initial,), (tangent,))
    input_cotangent = jax.grad(lambda state: jnp.vdot(cotangent, endpoint(state)))(
        initial
    )
    exponential = jax.scipy.linalg.expm(0.9 * matrix)
    assert jnp.allclose(value, exponential @ initial, atol=2e-10)
    assert jnp.allclose(output_tangent, exponential @ tangent, atol=2e-9)
    assert jnp.allclose(input_cotangent, exponential.T @ cotangent, atol=2e-9)


def test_linear_exponential_input_validation():
    x_0 = jnp.asarray([1.0, 0.0])
    with pytest.raises(TypeError, match="method"):
        solve_linear_ode(jnp.eye(2), object(), 0.0, 1.0, x_0)
    with pytest.raises(ValueError, match="square"):
        solve_linear_ode(jnp.ones((2, 3)), DenseExponential(), 0.0, 1.0, x_0)
    with pytest.raises(ValueError, match="endpoint"):
        solve_linear_ode(
            jnp.eye(2), DenseExponential(), 0.0, 1.0, x_0, save_at=SaveAt(steps=True)
        )
    outside = solve_linear_ode(
        jnp.eye(2),
        DenseExponential(),
        0.0,
        1.0,
        x_0,
        save_at=SaveAt(ts=jnp.asarray([-0.1, 0.2])),
    )
    assert not bool(outside.ok)
    with pytest.raises(ValueError, match="reorthogonalization_passes"):
        KrylovExponential(reorthogonalization_passes=0)
    with pytest.raises(ValueError, match="reorthogonalization_passes"):
        KrylovExponential(reorthogonalization_passes=True)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("passes", [1, 2])
def test_krylov_reorthogonalization_options_on_nonnormal_generator(dtype, passes):
    off_diagonal = jnp.asarray(
        [
            [0.0, 4.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.0, 3.0, 0.0, 0.0, 0.0],
            [0.0, 0.2, 0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 0.3, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.4, 0.0, 0.5],
            [0.2, 0.0, 0.0, 0.0, 0.1, 0.0],
        ],
        dtype,
    )
    generator = off_diagonal - jnp.diag(jnp.sum(off_diagonal, axis=1))
    operator = generator.T
    initial = jnp.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype)
    method = KrylovExponential(
        krylov_dim=6,
        num_substeps=2,
        reorthogonalization_passes=passes,
    )
    solution = solve_linear_ode(operator, method, 0.0, 1.5, initial)
    expected = jax.scipy.linalg.expm(jnp.asarray(1.5, dtype) * operator) @ initial
    tolerance = 8e-5 if dtype == jnp.float32 else 3e-11
    assert bool(solution.ok)
    assert jnp.allclose(solution.xs, expected, rtol=tolerance, atol=tolerance)
    assert jnp.allclose(jnp.sum(solution.xs), 1.0, atol=tolerance)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize(
    "method",
    [
        DenseExponential(),
        KrylovExponential(krylov_dim=3),
        KrylovExponential(krylov_dim=3, reorthogonalization_passes=1),
    ],
)
def test_handcoded_initial_state_jvp_vjp_include_zero_state(dtype, method):
    matrix = jnp.asarray([[-0.7, 0.2, 0.1], [0.4, -0.6, 0.0], [0.1, 0.3, -0.5]], dtype)
    tangent = jnp.asarray([0.2, -0.1, 0.4], dtype)
    cotangent = jnp.asarray([-0.3, 0.7, 0.2], dtype)
    expected_operator = jax.scipy.linalg.expm(jnp.asarray(0.8, dtype) * matrix)
    tolerance = 4e-5 if dtype == jnp.float32 else 3e-11
    for initial in (
        jnp.asarray([0.3, -0.2, 0.5], dtype),
        jnp.zeros(3, dtype),
    ):
        jvp_solution, output_tangent = jvp_linear_ode(
            matrix, method, 0.0, 0.8, initial, tangent
        )
        vjp_solution, input_cotangent = vjp_linear_ode(
            matrix, method, 0.0, 0.8, initial, cotangent
        )
        expected_value = expected_operator @ initial
        assert bool(jvp_solution.ok & vjp_solution.ok)
        assert jnp.allclose(jvp_solution.xs, expected_value, atol=tolerance)
        assert jnp.allclose(vjp_solution.xs, expected_value, atol=tolerance)
        assert jnp.allclose(output_tangent, expected_operator @ tangent, atol=tolerance)
        assert jnp.allclose(
            input_cotangent, expected_operator.T @ cotangent, atol=tolerance
        )


@pytest.mark.parametrize(
    "method", [DenseExponential(), KrylovExponential(krylov_dim=4)]
)
def test_handcoded_batched_jvp_vjp_reuse_and_pytree(method):
    matrix = jnp.asarray(
        [
            [-0.8, 0.1, 0.2, 0.0],
            [0.3, -0.5, 0.0, 0.1],
            [0.0, 0.2, -0.6, 0.2],
            [0.1, 0.0, 0.3, -0.7],
        ]
    )

    def flatten(state):
        return jnp.concatenate([state["left"], state["right"]], axis=-1)

    def unravel(flat):
        return {"left": flat[..., :2], "right": flat[..., 2:]}

    def operator(state):
        return unravel(matrix @ flatten(state))

    initial = unravel(jnp.asarray([0.2, -0.1, 0.5, 0.3]))
    tangents = unravel(
        jnp.asarray(
            [[0.1, 0.2, -0.3, 0.4], [-0.2, 0.5, 0.1, -0.4], [0.3, 0.0, 0.2, 0.1]]
        )
    )
    cotangents = unravel(
        jnp.asarray(
            [[0.4, -0.1, 0.2, 0.3], [0.0, 0.2, -0.5, 0.7], [0.1, 0.3, 0.2, -0.2]]
        )
    )
    jvp_solution, output_tangents = jvp_linear_ode(
        operator, method, 0.0, 0.6, initial, tangents, batched=True
    )
    vjp_solution, input_cotangents = vjp_linear_ode(
        operator, method, 0.0, 0.6, initial, cotangents, batched=True
    )
    exponential = jax.scipy.linalg.expm(0.6 * matrix)
    expected_value = exponential @ flatten(initial)
    assert bool(jvp_solution.ok & vjp_solution.ok)
    assert jnp.allclose(flatten(jvp_solution.xs), expected_value, atol=2e-10)
    assert jnp.allclose(
        flatten(output_tangents), flatten(tangents) @ exponential.T, atol=2e-9
    )
    assert jnp.allclose(
        flatten(input_cotangents), flatten(cotangents) @ exponential, atol=2e-9
    )


def test_dense_custom_rule_operator_and_time_derivatives_match_finite_difference():
    matrix = jnp.asarray([[-0.8, 0.3], [0.2, -0.4]])
    matrix_tangent = jnp.asarray([[0.1, -0.2], [0.05, 0.3]])
    initial = jnp.asarray([0.4, -0.7])
    initial_tangent = jnp.asarray([0.2, 0.1])

    def endpoint(current_matrix, current_initial, horizon):
        return solve_linear_ode(
            current_matrix, DenseExponential(), 0.0, horizon, current_initial
        ).xs

    _, tangent = jax.jvp(
        endpoint,
        (matrix, initial, jnp.asarray(0.9)),
        (matrix_tangent, initial_tangent, jnp.asarray(0.2)),
    )
    epsilon = 1e-5
    plus = endpoint(
        matrix + epsilon * matrix_tangent,
        initial + epsilon * initial_tangent,
        0.9 + epsilon * 0.2,
    )
    minus = endpoint(
        matrix - epsilon * matrix_tangent,
        initial - epsilon * initial_tangent,
        0.9 - epsilon * 0.2,
    )
    finite_difference = (plus - minus) / (2 * epsilon)
    assert jnp.allclose(tangent, finite_difference, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_adaptive_krylov_rejects_then_meets_endpoint_tolerance(dtype):
    eigenvalues = jnp.linspace(jnp.asarray(-0.1, dtype), jnp.asarray(-10.0, dtype), 40)
    initial = jnp.ones(40, dtype)
    tolerance = 2e-5 if dtype == jnp.float32 else 2e-9
    method = AdaptiveKrylovExponential(
        krylov_dim=10,
        max_steps=64,
        rtol=tolerance,
        atol=tolerance * 1e-2,
    )
    solution = jax.jit(solve_linear_ode, static_argnums=1)(
        jnp.diag(eigenvalues), method, 0.0, 2.0, initial
    )
    expected = jnp.exp(2 * eigenvalues)
    assert bool(solution.ok)
    assert 1 < int(solution.num_accepted) < method.max_steps
    assert jnp.linalg.norm(solution.xs - expected) <= 2 * tolerance


def test_adaptive_krylov_attempt_budget_failure_is_fast_and_finite():
    eigenvalues = jnp.linspace(-0.1, -10.0, 40)
    initial = jnp.ones(40)
    solution = solve_linear_ode(
        jnp.diag(eigenvalues),
        AdaptiveKrylovExponential(
            krylov_dim=6,
            max_steps=1,
            rtol=1e-10,
            atol=1e-12,
        ),
        0.0,
        2.0,
        initial,
    )
    assert not bool(solution.ok)
    assert int(solution.num_accepted) == 0
    assert jnp.all(jnp.isfinite(solution.xs))
    assert jnp.array_equal(solution.xs, initial)


def test_adaptive_krylov_pytree_vmap_and_handcoded_derivatives():
    matrix = jnp.asarray(
        [
            [-0.8, 0.2, 0.0, 0.1],
            [0.3, -0.7, 0.2, 0.0],
            [0.0, 0.1, -0.6, 0.3],
            [0.2, 0.0, 0.1, -0.5],
        ]
    )

    def unravel(vector):
        return {"left": vector[..., :2], "right": (vector[..., 2:],)}

    def flatten(state):
        return jnp.concatenate([state["left"], state["right"][0]], axis=-1)

    def operator(state):
        return unravel(matrix @ flatten(state))

    method = AdaptiveKrylovExponential(
        krylov_dim=3, max_steps=128, rtol=1e-6, atol=1e-8
    )
    initial = unravel(jnp.asarray([0.4, -0.2, 0.7, 0.1]))
    tangent = unravel(jnp.asarray([0.1, 0.3, -0.2, 0.4]))
    cotangent = unravel(jnp.asarray([-0.2, 0.5, 0.1, 0.3]))
    exponential = jax.scipy.linalg.expm(0.8 * matrix)

    solve_batch = jax.jit(
        jax.vmap(
            lambda vector: (
                solve_linear_ode(operator, method, 0.0, 0.8, unravel(vector)).xs
            )
        )
    )
    batch = jnp.stack([flatten(initial), 2 * flatten(initial)])
    batched = solve_batch(batch)
    jvp_solution, output_tangent = jvp_linear_ode(
        operator, method, 0.0, 0.8, initial, tangent
    )
    vjp_solution, input_cotangent = vjp_linear_ode(
        operator, method, 0.0, 0.8, initial, cotangent
    )

    def endpoint(vector):
        return flatten(solve_linear_ode(operator, method, 0.0, 0.8, unravel(vector)).xs)

    _, traced_tangent = jax.jvp(endpoint, (flatten(initial),), (flatten(tangent),))
    traced_cotangent = jax.grad(
        lambda vector: jnp.vdot(flatten(cotangent), endpoint(vector))
    )(flatten(initial))
    assert jnp.allclose(flatten(batched), batch @ exponential.T, atol=3e-5)
    assert bool(jvp_solution.ok & vjp_solution.ok)
    assert jnp.allclose(
        flatten(output_tangent), exponential @ flatten(tangent), atol=3e-5
    )
    assert jnp.allclose(
        flatten(input_cotangent), exponential.T @ flatten(cotangent), atol=3e-5
    )
    assert jnp.allclose(traced_tangent, exponential @ flatten(tangent), atol=3e-5)
    assert jnp.allclose(traced_cotangent, exponential.T @ flatten(cotangent), atol=3e-5)


def test_adaptive_krylov_constructor_validation():
    with pytest.raises(ValueError, match="max_steps"):
        AdaptiveKrylovExponential(max_steps=0)
    with pytest.raises(ValueError, match="initial_step"):
        AdaptiveKrylovExponential(initial_step=0.0)
    with pytest.raises(ValueError, match="safety"):
        AdaptiveKrylovExponential(safety=1.1)
    with pytest.raises(ValueError, match="min_factor"):
        AdaptiveKrylovExponential(min_factor=1.0)
    with pytest.raises(ValueError, match="max_factor"):
        AdaptiveKrylovExponential(max_factor=1.0)
