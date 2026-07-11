import jax
import jax.numpy as jnp
import pytest

from tinydiffeq import (
    RK4,
    ConstantStepSize,
    IController,
    LMRootSolver,
    Rodas5P,
    SaveAt,
    Tsit5,
    solve_semi_explicit_dae,
)


def fields():
    def f(y, z, t, args, p):
        return p["a"] * y

    def g(y, z, t, args, p):
        residual = z - y**2 - p["b"] * t
        aux = {
            "mixed": (p["b"] * z + y).astype(jnp.float32),
            "state": jnp.stack([y, z, p["b"] * z + y]),
        }
        return residual, aux

    return f, g


def solve_aux(save_at, p=None, n=32):
    f, g = fields()
    if p is None:
        p = {"a": jnp.asarray(0.3), "b": jnp.asarray(0.4)}
    return solve_semi_explicit_dae(
        f,
        g,
        RK4(),
        0.0,
        1.0,
        jnp.asarray(0.8),
        jnp.asarray(0.5),
        p=p,
        dt_0=1.0 / n,
        controller=ConstantStepSize(),
        root_solver=LMRootSolver(atol=1e-11),
        max_steps=n,
        save_at=save_at,
        has_aux=True,
    )


@pytest.mark.parametrize(
    "save_at",
    [SaveAt(t_1=True), SaveAt(steps=True), SaveAt(ts=jnp.linspace(0.0, 1.0, 17))],
)
def test_aux_all_save_modes_and_mixed_leaf_dtypes(save_at):
    sol = solve_aux(save_at)
    assert bool(sol.ok)
    assert sol.aux["mixed"].dtype == jnp.float32
    assert sol.aux["state"].dtype == jnp.float64
    y = sol.ys
    z = sol.zs
    expected = 0.4 * z + y
    assert jnp.max(jnp.abs(sol.aux["mixed"] - expected)) < 2e-5
    assert jnp.max(jnp.abs(sol.aux["state"][..., 2] - expected)) < 2e-8
    if save_at.steps:
        assert sol.aux["mixed"].shape == sol.ts.shape
        assert sol.aux["state"].shape == sol.ts.shape + (3,)


def test_interpolated_z_and_aux_jvp_vjp_include_direct_and_root_paths():
    grid = jnp.linspace(0.0, 1.0, 13)
    a = jnp.asarray(0.3)
    b = jnp.asarray(0.4)

    def sampled(q):
        sol = solve_aux(SaveAt(ts=grid), {"a": a, "b": q}, n=48)
        return jnp.sum(sol.ys + sol.zs + sol.aux["state"][..., 2])

    y = 0.8 * jnp.exp(a * grid)
    z = y**2 + b * grid
    # d/db [z + (b*z + y)] = t + z + b*t
    exact = jnp.sum(grid + z + b * grid)
    tangent = jax.jvp(sampled, (b,), (jnp.ones_like(b),))[1]
    cotangent = jax.grad(sampled)(b)
    assert jnp.abs(tangent - exact) < 2e-4
    assert jnp.abs(cotangent - exact) < 2e-4
    assert jnp.abs(tangent - cotangent) < 2e-8


def test_dense_algebraic_outputs_have_fourth_order_error_and_knot_exactness():
    def error(n):
        # Keep the query at the midpoint of the first interval at every
        # refinement, so phase changes cannot hide the asymptotic order.
        query = jnp.asarray([0.5 / n])
        sol = solve_aux(SaveAt(ts=query), n=n)
        y = 0.8 * jnp.exp(0.3 * query)
        z = y**2 + 0.4 * query
        aux = 0.4 * z + y
        return jnp.maximum(
            jnp.abs(sol.zs - z),
            jnp.abs(sol.aux["state"][..., 2] - aux),
        )[0]

    errors = [error(n) for n in (8, 16, 32)]
    assert errors[0] / errors[1] > 12.0
    assert errors[1] / errors[2] > 12.0

    steps = solve_aux(SaveAt(steps=True), n=16)
    knots = steps.ts[steps.accepted]
    queried = solve_aux(SaveAt(ts=knots), n=16)
    assert jnp.array_equal(queried.zs, steps.zs[steps.accepted])
    assert jnp.array_equal(queried.aux["state"], steps.aux["state"][steps.accepted])


@pytest.mark.parametrize(
    ("dtype", "epsilon", "root_atol", "rtol", "atol"),
    [
        (jnp.float32, 1e-3, 1e-7, 3e-3, 3e-4),
        (jnp.float64, 1e-6, 1e-13, 3e-7, 3e-9),
    ],
)
def test_ill_conditioned_algebraic_jacobian_has_finite_dense_ad(
    dtype, epsilon, root_atol, rtol, atol
):
    epsilon = jnp.asarray(epsilon, dtype)
    query = jnp.asarray([0.37], dtype)

    def output(rate):
        def constraint(y, z):
            difference = z - jnp.stack([y, y])
            residual = jnp.asarray([difference[0], epsilon * difference[1]])
            return residual, {"sum": jnp.sum(z)}

        sol = solve_semi_explicit_dae(
            lambda y, z, t, args, p: p * y,
            constraint,
            RK4(),
            jnp.asarray(0.0, dtype),
            jnp.asarray(1.0, dtype),
            jnp.asarray(1.0, dtype),
            jnp.ones(2, dtype),
            p=rate,
            dt_0=jnp.asarray(0.05, dtype),
            root_solver=LMRootSolver(atol=root_atol, max_steps=64),
            max_steps=20,
            save_at=SaveAt(ts=query),
            has_aux=True,
        )
        return jnp.sum(sol.zs) + sol.aux["sum"][0]

    rate = jnp.asarray(0.2, dtype)
    value = output(rate)
    tangent = jax.jvp(output, (rate,), (jnp.ones_like(rate),))[1]
    cotangent = jax.grad(output)(rate)
    expected_value = 4.0 * jnp.exp(rate * query[0])
    expected_derivative = query[0] * expected_value
    assert jnp.all(jnp.isfinite(jnp.asarray([value, tangent, cotangent])))
    assert jnp.allclose(value, expected_value, rtol=rtol, atol=atol)
    assert jnp.allclose(tangent, expected_derivative, rtol=rtol, atol=atol)
    assert jnp.allclose(cotangent, expected_derivative, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    ("g", "has_aux", "match"),
    [
        (lambda y, z: (z - y, {"count": jnp.asarray(1)}), True, "real floating"),
        (lambda y, z: (z - y, {}), True, "at least one"),
        (lambda y, z: (z - y, {"x": jnp.asarray([])}), True, "not be empty"),
        (lambda y, z: z - y, True, "must return"),
        (lambda y, z: (z - y, {"x": y}), False, "has_aux=True"),
    ],
)
def test_aux_contract_validation(g, has_aux, match):
    with pytest.raises((TypeError, ValueError), match=match):
        solve_semi_explicit_dae(
            lambda y, z: z,
            g,
            RK4(),
            0.0,
            0.1,
            jnp.asarray(1.0),
            jnp.asarray(1.0),
            dt_0=0.1,
            max_steps=1,
            has_aux=has_aux,
        )


def test_vmap_adaptive_interpolated_aux_has_finite_jvp_and_vjp():
    f, g = fields()
    grid = jnp.linspace(0.0, 1.0, 9)

    def solution(a):
        return solve_semi_explicit_dae(
            f,
            g,
            Tsit5(),
            0.0,
            1.0,
            jnp.asarray(0.8),
            jnp.asarray(0.5),
            p={"a": a, "b": jnp.asarray(0.4)},
            dt_0=0.2,
            controller=IController(rtol=1e-5, atol=1e-7),
            root_solver=LMRootSolver(atol=1e-9),
            max_steps=96,
            save_at=SaveAt(ts=grid),
            has_aux=True,
        )

    def output(a):
        sol = solution(a)
        return jnp.sum(sol.zs + sol.aux["state"][..., 2])

    rates = jnp.asarray([0.1, 1.0, 3.0])
    solutions = jax.jit(jax.vmap(solution))(rates)
    values = jax.jit(jax.vmap(output))(rates)
    tangents = jax.jit(
        jax.vmap(lambda a: jax.jvp(output, (a,), (jnp.ones_like(a),))[1])
    )(rates)
    cotangents = jax.jit(jax.vmap(jax.grad(output)))(rates)
    assert jnp.all(solutions.ok)
    assert jnp.unique(solutions.num_accepted).size > 1
    assert jnp.all(jnp.isfinite(values))
    assert jnp.all(jnp.isfinite(tangents))
    assert jnp.all(jnp.isfinite(cotangents))
    assert jnp.allclose(tangents, cotangents, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize(
    ("save_at", "multiplicity"),
    [
        (SaveAt(t_1=True), 1.0),
        (SaveAt(ts=jnp.asarray([0.0, 0.1])), 2.0),
    ],
)
def test_masked_failed_lane_has_safe_aux_jvp_and_vjp(save_at, multiplicity):
    y_0 = jnp.asarray([1.0, -1.0])
    z_0 = jnp.asarray([1.0, 0.0])

    def one_lane(y, z, p):
        return solve_semi_explicit_dae(
            lambda y, z: jnp.zeros_like(y),
            lambda y, z, t, args, p: (
                z**2 - y - p,
                {"off_domain": p * jnp.sqrt(y)},
            ),
            RK4(),
            0.0,
            0.1,
            y,
            z,
            p=p,
            dt_0=0.1,
            max_steps=1,
            save_at=save_at,
            has_aux=True,
            failure_ad_reference=(1.0, 1.0, 0.0, 0.0),
        )

    def batch(p):
        return jax.vmap(lambda y, z: one_lane(y, z, p))(y_0, z_0)

    def loss(p):
        sol = batch(p)
        lane_values = sol.zs.reshape(2, -1).sum(axis=1)
        lane_values += sol.aux["off_domain"].reshape(2, -1).sum(axis=1)
        return jnp.sum(jnp.where(sol.ok, lane_values, 0.0))

    p = jnp.asarray(0.0)
    sol = batch(p)
    expected = 1.5 * multiplicity
    assert jnp.array_equal(sol.ok, jnp.asarray([True, False]))
    assert jnp.all(jnp.isfinite(sol.aux["off_domain"]))
    assert jnp.allclose(jax.jvp(loss, (p,), (jnp.ones_like(p),))[1], expected)
    assert jnp.allclose(jax.grad(loss)(p), expected)


@pytest.mark.parametrize("solver", [RK4(), Rodas5P()])
def test_nonfinite_initial_aux_fails_without_attempting_time_steps(solver):
    sol = solve_semi_explicit_dae(
        lambda y, z: jnp.full_like(y, jnp.nan),
        lambda y, z: (z - y, {"bad": jnp.sqrt(-y)}),
        solver,
        0.0,
        1.0,
        jnp.asarray(1.0),
        jnp.asarray(1.0),
        dt_0=0.1,
        max_steps=16,
        save_at=SaveAt(steps=True),
        has_aux=True,
    )
    assert not bool(sol.ok)
    assert int(sol.num_accepted) == 0
    assert jnp.all(sol.accepted == jnp.asarray([True] + [False] * 16))
    assert jnp.all(sol.ys == 1.0)
    assert jnp.all(sol.aux["bad"] == 0.0)


@pytest.mark.parametrize("solver", [RK4(), Rodas5P()])
def test_nonfinite_endpoint_aux_terminates_at_previous_node(solver):
    sol = solve_semi_explicit_dae(
        lambda y, z: jnp.ones_like(y),
        lambda y, z, t: (z - y, {"limited": jnp.sqrt(0.05 - t)}),
        solver,
        0.0,
        0.2,
        jnp.asarray(0.0),
        jnp.asarray(0.0),
        dt_0=0.1,
        max_steps=2,
        save_at=SaveAt(steps=True),
        has_aux=True,
    )
    assert not bool(sol.ok)
    assert int(sol.num_accepted) == 0
    assert jnp.all(sol.ys == 0.0)
    assert jnp.all(sol.aux["limited"] == sol.aux["limited"][0])
