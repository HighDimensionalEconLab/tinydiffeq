import jax
import jax.numpy as jnp
import pytest

from tinydiffeq import (
    RK4,
    Euler,
    EulerMaruyama,
    IController,
    Rodas5P,
    SaveAt,
    Tsit5,
    solve_ode,
    solve_sde,
    solve_semi_explicit_dae,
    solve_semi_explicit_sdae,
)


@pytest.mark.parametrize("solver", [Euler(), RK4(), Tsit5(), Rodas5P()])
@pytest.mark.parametrize(
    "save_at",
    [SaveAt(t_1=True), SaveAt(steps=True), SaveAt(ts=jnp.linspace(0.0, 1.0, 9))],
)
def test_ode_aux_all_methods_and_save_modes(solver, save_at):
    if isinstance(solver, (Tsit5, Rodas5P)):
        controller = IController(rtol=1e-6, atol=1e-8)
        max_steps = 128
    else:
        controller = None
        max_steps = 16

    def field(x, t, args, p):
        return p * x, {
            "square": x**2,
            "mixed": (x + p).astype(jnp.float32),
        }

    solution = solve_ode(
        field,
        solver,
        0.0,
        1.0,
        jnp.asarray(1.0),
        p=jnp.asarray(0.2),
        dt_0=1.0 / 16.0,
        controller=controller,
        max_steps=max_steps,
        save_at=save_at,
    )
    assert bool(solution.ok)
    assert solution.aux["mixed"].dtype == jnp.float32
    assert jnp.allclose(solution.aux["square"], solution.xs**2, rtol=3e-3, atol=3e-4)


def test_ode_interpolated_aux_jvp_vjp_and_tuple_state_detection():
    grid = jnp.linspace(0.0, 1.0, 11)

    def field(state, t, args, p):
        x, y = state
        derivative = (p * x, -p * y)
        return derivative, {"product": x * y + p}

    def output(rate):
        solution = solve_ode(
            field,
            RK4(),
            0.0,
            1.0,
            (jnp.asarray(1.0), jnp.asarray(2.0)),
            p=rate,
            dt_0=0.05,
            max_steps=20,
            save_at=SaveAt(ts=grid),
        )
        return jnp.sum(solution.aux["product"])

    rate = jnp.asarray(0.3)
    tangent = jax.jvp(output, (rate,), (jnp.ones_like(rate),))[1]
    cotangent = jax.grad(output)(rate)
    assert jnp.allclose(tangent, cotangent, rtol=1e-5, atol=1e-6)

    no_aux = solve_ode(
        lambda state: (-state[0], -state[1]),
        Euler(),
        0.0,
        0.1,
        (jnp.asarray(1.0), jnp.asarray(2.0)),
        dt_0=0.1,
        max_steps=1,
    )
    assert no_aux.aux is None


def test_ode_interpolated_aux_accepts_discrete_parameter_leaves():
    grid = jnp.linspace(0.0, 0.2, 5)

    def output(x_0):
        solution = solve_ode(
            lambda x, t, args, p: (
                p["rate"] * x,
                {"scaled": p["count"].astype(x.dtype) * x},
            ),
            RK4(),
            0.0,
            0.2,
            x_0,
            p={"rate": jnp.asarray(0.1), "count": jnp.asarray(2, jnp.int32)},
            dt_0=0.05,
            max_steps=4,
            save_at=SaveAt(ts=grid),
        )
        return jnp.sum(solution.aux["scaled"])

    x_0 = jnp.asarray(1.0)
    tangent = jax.jvp(output, (x_0,), (jnp.ones_like(x_0),))[1]
    assert jnp.isfinite(tangent)
    assert jnp.allclose(tangent, jax.grad(output)(x_0))


def test_ode_endpoint_nonfinite_aux_keeps_state_and_zeroes_aux():
    solution = solve_ode(
        lambda x, t: (jnp.ones_like(x), {"bad": jnp.sqrt(0.05 - t)}),
        RK4(),
        0.0,
        0.1,
        jnp.asarray(0.0),
        dt_0=0.1,
        max_steps=1,
    )
    assert not bool(solution.ok)
    assert jnp.allclose(solution.xs, 0.1)
    assert solution.aux["bad"] == 0.0


@pytest.mark.parametrize("stochastic", [False, True], ids=["ode", "sde"])
def test_ode_and_sde_masked_failed_lane_aux_ad_is_safe(stochastic):
    initial = jnp.asarray([1.0, -1.0])

    def one_lane(x_0, p):
        def field(x, t, args, p_value):
            return jnp.zeros_like(x), {"root": p_value * jnp.sqrt(x)}

        options = {
            "p": p,
            "failure_ad_reference": (1.0, 0.0, 0.0),
        }
        if stochastic:
            return solve_sde(
                field,
                lambda x: jnp.zeros_like(x),
                EulerMaruyama(),
                0.0,
                0.1,
                x_0,
                key=jax.random.key(0),
                n_steps=1,
                **options,
            )
        return solve_ode(
            field,
            Euler(),
            0.0,
            0.1,
            x_0,
            dt_0=0.1,
            max_steps=1,
            **options,
        )

    def batch(p):
        return jax.vmap(lambda x_0: one_lane(x_0, p))(initial)

    def loss(p):
        solution = batch(p)
        return jnp.sum(jnp.where(solution.ok, solution.aux["root"], 0.0))

    p = jnp.asarray(0.0)
    solution = batch(p)
    assert jnp.array_equal(solution.ok, jnp.asarray([True, False]))
    assert jax.jvp(loss, (p,), (jnp.ones_like(p),))[1] == 1.0
    assert jax.grad(loss)(p) == 1.0


def test_dae_context_reaches_field_and_supports_discrete_leaves():
    def constraint(y, z, t, args, p):
        return z - y, {
            "level": z,
            "enabled": jnp.asarray(True),
            "count": jnp.asarray(2, jnp.int32),
        }

    def field(y, z, t, args, p, algebraic_aux):
        scale = algebraic_aux["count"].astype(y.dtype) / 2.0
        derivative = p * algebraic_aux["level"] * scale
        return derivative, {"level": algebraic_aux["level"]}

    def output(rate):
        return solve_semi_explicit_dae(
            field,
            constraint,
            RK4(),
            0.0,
            0.5,
            jnp.asarray(1.0),
            jnp.asarray(0.8),
            p=rate,
            dt_0=0.05,
            max_steps=10,
            save_at=SaveAt(t_1=True),
        ).aux["level"]

    rate = jnp.asarray(0.2)
    tangent = jax.jvp(output, (rate,), (jnp.ones_like(rate),))[1]
    cotangent = jax.grad(output)(rate)
    assert jnp.allclose(tangent, cotangent, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("solver", [RK4(), Tsit5(), Rodas5P()])
@pytest.mark.parametrize(
    "save_at",
    [SaveAt(t_1=True), SaveAt(steps=True), SaveAt(ts=jnp.linspace(0.0, 0.5, 7))],
)
def test_dae_aux_all_methods_and_save_modes(solver, save_at):
    solution = solve_semi_explicit_dae(
        lambda y, z, t, args, p, context: (
            p * context["z"],
            {"square": context["z"] ** 2},
        ),
        lambda y, z: (z - y, {"z": z}),
        solver,
        0.0,
        0.5,
        jnp.asarray(1.0),
        jnp.asarray(0.8),
        p=jnp.asarray(0.2),
        dt_0=0.05,
        max_steps=64,
        save_at=save_at,
    )
    assert bool(solution.ok)
    assert jnp.allclose(solution.aux["square"], solution.zs**2, rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize(
    ("saved", "context"),
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_dae_supports_all_saved_and_algebraic_aux_combinations(saved, context):
    if context:

        def g(y, z):
            return z - y, {"z": z}

        if saved:

            def f(y, z, t, args, p, cache):
                return p * cache["z"], {"z": z}

        else:

            def f(y, z, t, args, p, cache):
                return p * cache["z"]

    else:

        def g(y, z):
            return z - y

        if saved:

            def f(y, z, t, args, p):
                return p * z, {"z": z}

        else:

            def f(y, z, t, args, p):
                return p * z

    solution = solve_semi_explicit_dae(
        f,
        g,
        RK4(),
        0.0,
        0.1,
        jnp.asarray(1.0),
        jnp.asarray(0.8),
        p=jnp.asarray(0.2),
        dt_0=0.1,
        max_steps=1,
    )
    assert bool(solution.ok)
    assert (solution.aux is not None) == saved


def test_sde_drift_aux_node_alignment_and_pathwise_ad():
    key = jax.random.key(7)

    def drift(x, t, args, p):
        return p * x, {"square": x**2}

    def diffusion(x):
        return 0.1 * jnp.ones_like(x)

    def output(rate):
        solution = solve_sde(
            drift,
            diffusion,
            EulerMaruyama(),
            0.0,
            0.5,
            jnp.asarray(1.0),
            key=key,
            n_steps=8,
            p=rate,
            save_at=SaveAt(steps=True),
        )
        assert jnp.allclose(solution.aux["square"], solution.xs**2)
        return jnp.sum(solution.aux["square"])

    rate = jnp.asarray(0.2)
    tangent = jax.jvp(output, (rate,), (jnp.ones_like(rate),))[1]
    cotangent = jax.grad(output)(rate)
    assert jnp.allclose(tangent, cotangent, rtol=1e-5, atol=1e-6)


def test_sdae_same_algebraic_aux_reaches_drift_and_diffusion():
    def constraint(y, z, t, args, p):
        return z - y, {"coefficient": p + z}

    def drift(y, z, t, args, p, algebraic_aux):
        coefficient = algebraic_aux["coefficient"]
        return 0.1 * coefficient, {"coefficient": coefficient}

    def diffusion(y, z, t, args, p, algebraic_aux):
        return 0.01 * algebraic_aux["coefficient"]

    solution = solve_semi_explicit_sdae(
        drift,
        diffusion,
        constraint,
        EulerMaruyama(),
        0.0,
        0.2,
        jnp.asarray(1.0),
        jnp.asarray(0.8),
        p=jnp.asarray(0.2),
        key=jax.random.key(4),
        n_steps=4,
        save_at=SaveAt(steps=True),
    )
    assert bool(solution.ok)
    assert jnp.allclose(
        solution.aux["coefficient"], 0.2 + solution.zs, rtol=1e-6, atol=1e-6
    )
