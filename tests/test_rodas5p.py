import jax
import jax.numpy as jnp

from tinydiffeq import (
    IController,
    LMRootSolver,
    Rodas5P,
    SaveAt,
    solve_ode,
    solve_semi_explicit_dae,
)
from tinydiffeq._rodas5p import GAMMA


def test_fixed_ode_and_dae_match_sciml_rodas5p_reference_steps():
    # Generated with SciML's OrdinaryDiffEqRosenbrock.Rodas5P at adaptive=false.
    # Implementation: https://github.com/SciML/OrdinaryDiffEq.jl/tree/master/lib/OrdinaryDiffEqRosenbrock
    ode = solve_ode(
        lambda x: -x,
        Rodas5P(),
        0.0,
        1.0,
        jnp.asarray(1.0),
        dt_0=1.0,
        max_steps=1,
    )
    assert bool(ode.ok)
    assert jnp.allclose(ode.xs, 0.36788030893705337, rtol=0.0, atol=5e-14)

    nonautonomous = solve_ode(
        lambda x, t: -2.0 * x + t**2,
        Rodas5P(),
        0.0,
        0.3,
        jnp.asarray(1.0),
        dt_0=0.3,
        max_steps=1,
    )
    assert jnp.allclose(
        nonautonomous.xs,
        0.5566087803098965,
        rtol=0.0,
        atol=5e-14,
    )
    nonautonomous_dense = solve_ode(
        lambda x, t: -2.0 * x + t**2,
        Rodas5P(),
        0.0,
        0.3,
        jnp.asarray(1.0),
        dt_0=0.3,
        max_steps=1,
        save_at=SaveAt(ts=jnp.asarray([0.12])),
    )
    assert jnp.allclose(
        nonautonomous_dense.xs[0],
        0.7871691747753758,
        rtol=0.0,
        atol=5e-14,
    )

    dae = solve_semi_explicit_dae(
        lambda y, z: -0.2 * z,
        lambda y, z: z**2 - y - 2.0,
        Rodas5P(),
        0.0,
        0.1,
        jnp.asarray(1.0),
        jnp.sqrt(jnp.asarray(3.0)),
        dt_0=0.1,
        max_steps=1,
    )
    assert bool(dae.ok)
    assert jnp.allclose(dae.ys, 0.9654589838486518, rtol=0.0, atol=5e-14)
    assert jnp.allclose(dae.zs, 1.7220508075679213, rtol=0.0, atol=5e-14)

    dae_dense = solve_semi_explicit_dae(
        lambda y, z: -0.2 * z,
        lambda y, z: z**2 - y - 2.0,
        Rodas5P(),
        0.0,
        0.1,
        jnp.asarray(1.0),
        jnp.sqrt(jnp.asarray(3.0)),
        dt_0=0.1,
        max_steps=1,
        save_at=SaveAt(ts=jnp.asarray([0.04])),
    )
    assert jnp.allclose(dae_dense.ys[0], 0.9861595935382219, rtol=0.0, atol=5e-14)
    assert jnp.allclose(dae_dense.zs[0], 1.7280508077866055, rtol=0.0, atol=5e-14)


def test_fifth_order_convergence_and_fourth_order_dense_output():
    endpoint_errors = []
    dense_errors = []
    for n_steps in (1, 2, 4, 8):
        endpoint = solve_ode(
            lambda x: -x,
            Rodas5P(),
            0.0,
            1.0,
            jnp.asarray(1.0),
            dt_0=1.0 / n_steps,
            max_steps=n_steps,
        )
        grid = jnp.arange(2 * n_steps + 1) / (2 * n_steps)
        sampled = solve_ode(
            lambda x: -x,
            Rodas5P(),
            0.0,
            1.0,
            jnp.asarray(1.0),
            dt_0=1.0 / n_steps,
            max_steps=n_steps,
            save_at=SaveAt(ts=grid),
        )
        endpoint_errors.append(jnp.abs(endpoint.xs - jnp.exp(-1.0)))
        dense_errors.append(jnp.max(jnp.abs(sampled.xs - jnp.exp(-grid))))

    assert endpoint_errors[1] / endpoint_errors[2] > 20.0
    assert endpoint_errors[2] / endpoint_errors[3] > 20.0
    assert dense_errors[1] / dense_errors[2] > 15.0
    assert dense_errors[2] / dense_errors[3] > 15.0


def test_stiff_non_autonomous_ode():
    # Exact solution x(t)=cos(t); the transient eigenvalue is -1000.
    sol = solve_ode(
        lambda x, t: -1000.0 * (x - jnp.cos(t)) - jnp.sin(t),
        Rodas5P(),
        0.0,
        0.1,
        jnp.asarray(1.0),
        dt_0=0.05,
        max_steps=2,
    )
    assert bool(sol.ok)
    assert jnp.abs(sol.xs - jnp.cos(0.1)) < 3e-10


def test_nonlinear_dae_adaptive_constraint_accuracy_and_initial_root_only():
    p = jnp.asarray(-0.2)
    sol = solve_semi_explicit_dae(
        lambda y, z, t, args, rate: rate * z,
        lambda y, z: z**2 - y - 2.0,
        Rodas5P(),
        0.0,
        1.0,
        jnp.asarray(1.0),
        jnp.sqrt(jnp.asarray(3.0)),
        p=p,
        dt_0=0.2,
        controller=IController(rtol=1e-8, atol=1e-10),
        root_solver=LMRootSolver(max_steps=1, atol=1e-12),
        max_steps=64,
        save_at=SaveAt(steps=True),
    )
    assert bool(sol.ok)
    valid = sol.accepted
    assert jnp.max(jnp.abs(sol.zs[valid] ** 2 - sol.ys[valid] - 2.0)) < 2e-8

    def from_guess(guess):
        return solve_semi_explicit_dae(
            lambda y, z: -0.2 * z,
            lambda y, z: z**2 - y - 2.0,
            Rodas5P(),
            0.0,
            0.2,
            jnp.asarray(1.0),
            guess,
            dt_0=0.1,
            max_steps=2,
        ).ys

    assert jnp.abs(jax.grad(from_guess)(jnp.sqrt(jnp.asarray(3.0)))) < 1e-14


def test_dae_dense_state_aux_jvp_vjp_and_reverse_over_forward():
    grid = jnp.linspace(0.0, 1.0, 17)
    y_0 = jnp.asarray(1.0)
    z_0 = jnp.sqrt(jnp.asarray(3.0))

    def run(rate):
        return solve_semi_explicit_dae(
            lambda y, z, t, args, p: p * z,
            lambda y, z: (z**2 - y - 2.0, {"value": y + 2.0 * z}),
            Rodas5P(),
            0.0,
            1.0,
            y_0,
            z_0,
            p=rate,
            dt_0=0.4,
            controller=IController(rtol=1e-8, atol=1e-10),
            max_steps=64,
            save_at=SaveAt(ts=grid),
            has_aux=True,
        )

    def output(rate):
        sol = run(rate)
        return jnp.sum(sol.ys + sol.zs + sol.aux["value"])

    rate = jnp.asarray(-0.2)
    tangent = jax.jvp(output, (rate,), (jnp.ones_like(rate),))[1]
    gradient = jax.grad(output)(rate)
    curvature = jax.grad(
        lambda value: jax.jvp(
            output,
            (value,),
            (jnp.ones_like(value),),
        )[1]
    )(rate)
    assert jnp.allclose(tangent, gradient, rtol=2e-8, atol=2e-8)
    exact_z = z_0 + 0.5 * rate * grid
    exact_first = jnp.sum(2.0 * exact_z * grid + 1.5 * grid)
    exact_second = jnp.sum(grid**2)
    assert jnp.abs(tangent - exact_first) < 5e-5
    assert jnp.abs(curvature - exact_second) < 1e-3

    sol = run(rate)
    exact_y = exact_z**2 - 2.0
    assert bool(sol.ok)
    assert jnp.max(jnp.abs(sol.ys - exact_y)) < 5e-8
    assert jnp.max(jnp.abs(sol.zs - exact_z)) < 5e-8
    assert jnp.max(jnp.abs(sol.aux["value"] - (exact_y + 2.0 * exact_z))) < 3e-7


def test_aux_dense_interpolation_converges_without_query_time_roots():
    errors = []
    for n_steps in (2, 4, 8):
        grid = jnp.arange(2 * n_steps + 1) / (2 * n_steps)
        sol = solve_semi_explicit_dae(
            lambda y, z: -z,
            lambda y, z: (z - y, {"square": z**2}),
            Rodas5P(),
            0.0,
            1.0,
            jnp.asarray(1.0),
            jnp.asarray(1.0),
            dt_0=1.0 / n_steps,
            max_steps=n_steps,
            save_at=SaveAt(ts=grid),
            has_aux=True,
        )
        errors.append(jnp.max(jnp.abs(sol.aux["square"] - jnp.exp(-2.0 * grid))))
    assert errors[0] / errors[1] > 10.0
    assert errors[1] / errors[2] > 12.0


def test_jit_vmap_and_pytree_states():
    initial = {"a": jnp.asarray(1.0), "b": jnp.asarray([2.0, 3.0])}

    def endpoint(rate, state=initial):
        return solve_ode(
            lambda value, t, args, p: jax.tree.map(lambda leaf: p * leaf, value),
            Rodas5P(),
            0.0,
            1.0,
            state,
            p=rate,
            dt_0=0.25,
            max_steps=4,
        ).xs

    rates = jnp.asarray([-0.1, -0.2, -0.3])
    result = jax.jit(jax.vmap(endpoint))(rates)
    assert result["a"].shape == (3,)
    assert result["b"].shape == (3, 2)
    assert jnp.allclose(result["a"], jnp.exp(rates), rtol=2e-7, atol=2e-7)


def test_float32_defaults_and_fixed_linear_failure_are_safe():
    dtype = jnp.float32
    sol = solve_ode(
        lambda x: -x,
        Rodas5P(),
        dtype(0.0),
        dtype(1.0),
        jnp.asarray(1.0, dtype),
        dt_0=dtype(0.1),
        controller=IController(),
        max_steps=64,
    )
    assert bool(sol.ok)
    assert sol.xs.dtype == dtype
    assert jnp.abs(sol.xs - jnp.exp(dtype(-1.0))) < 2e-5

    failed = solve_ode(
        lambda x: jnp.full_like(x, jnp.nan),
        Rodas5P(),
        0.0,
        1.0,
        jnp.asarray(1.0),
        dt_0=0.1,
        max_steps=8,
    )
    assert not bool(failed.ok)
    assert int(failed.num_accepted) == 0
    assert failed.ts == 0.0
    assert jnp.isfinite(failed.xs)


def test_dae_preserves_distinct_y_and_z_dtypes():
    y_0 = jnp.asarray(1.0, jnp.float32)
    z_0 = jnp.asarray(0.8, jnp.float64)

    def run(rate):
        return solve_semi_explicit_dae(
            lambda y, z, t, args, p: (p * z).astype(y.dtype),
            lambda y, z: z - y.astype(z.dtype),
            Rodas5P(),
            0.0,
            0.2,
            y_0,
            z_0,
            p=rate,
            dt_0=0.05,
            controller=IController(),
            max_steps=16,
        )

    rate = jnp.asarray(0.2, jnp.float32)
    sol = run(rate)
    tangent = jax.jvp(run, (rate,), (jnp.ones_like(rate),))[1]
    gradient = jax.grad(lambda value: run(value).ys)(rate)
    assert sol.ys.dtype == jnp.float32
    assert sol.zs.dtype == jnp.float64
    assert tangent.ys.dtype == jnp.float32
    assert tangent.zs.dtype == jnp.float64
    assert gradient.dtype == jnp.float32
    assert jnp.allclose(tangent.ys, gradient, rtol=2e-5, atol=2e-6)


def test_failed_vmap_lane_does_not_contaminate_successful_vjp():
    def single(rate):
        return solve_ode(
            lambda x, t, args, p: p * x,
            Rodas5P(),
            0.0,
            0.1,
            jnp.asarray(1.0),
            p=rate,
            dt_0=0.1,
            max_steps=1,
        )

    # This makes W = 1/(gamma*h) - J exactly singular in the second lane.
    rates = jnp.asarray([-1.0, 1.0 / (GAMMA * 0.1)])
    solve_batch = jax.vmap(single)
    sol = solve_batch(rates)
    jacobian = jax.jacrev(lambda value: solve_batch(value).xs)(rates)
    assert jnp.array_equal(sol.ok, jnp.asarray([True, False]))
    assert jnp.all(jnp.isfinite(jacobian))
    assert jnp.abs(jacobian[0, 0] - 0.1 * jnp.exp(-0.1)) < 1e-7
    assert jacobian[0, 1] == 0.0
