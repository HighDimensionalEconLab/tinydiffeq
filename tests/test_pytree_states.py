from dataclasses import dataclass

import jax
import jax.numpy as jnp
import pytest

from tinydiffeq import (
    RK4,
    EulerMaruyama,
    IController,
    SaveAt,
    Tsit5,
    solve_ode,
    solve_sde,
    solve_semi_explicit_dae,
)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class State:
    position: jax.Array
    velocity: jax.Array


def tree_initial():
    return {"a": jnp.asarray(1.0), "b": (jnp.asarray([2.0, 3.0]),)}


def tree_field(x, t, args, p):
    return jax.tree.map(lambda leaf: p * leaf, x)


@pytest.mark.parametrize(
    "save_at", [SaveAt(t_1=True), SaveAt(steps=True), SaveAt(ts=jnp.linspace(0, 1, 9))]
)
def test_ode_dict_tuple_state_all_output_modes(save_at):
    sol = solve_ode(
        tree_field,
        Tsit5(),
        0.0,
        1.0,
        tree_initial(),
        p=jnp.asarray(-0.5),
        dt_0=0.1,
        controller=IController(),
        max_steps=64,
        save_at=save_at,
    )
    assert bool(sol.ok)
    assert jax.tree.structure(sol.xs) == jax.tree.structure(tree_initial())
    expected = jnp.exp(-0.5) if save_at.t_1 else None
    if expected is not None:
        assert jnp.allclose(sol.xs["a"], expected, rtol=2e-4)
        assert jnp.allclose(
            sol.xs["b"][0], expected * jnp.asarray([2.0, 3.0]), rtol=2e-4
        )
    else:
        assert sol.xs["a"].shape == sol.ts.shape
        assert sol.xs["b"][0].shape == sol.ts.shape + (2,)


def test_fixed_ode_pytree_matches_equivalent_flat_array_exactly():
    flat_0 = jnp.asarray([1.0, 2.0, 3.0])
    kwargs = dict(dt_0=1 / 16, max_steps=16, save_at=SaveAt(steps=True))
    flat = solve_ode(lambda x: -0.2 * x, RK4(), 0.0, 1.0, flat_0, **kwargs)
    tree = solve_ode(
        lambda x: jax.tree.map(lambda leaf: -0.2 * leaf, x),
        RK4(),
        0.0,
        1.0,
        tree_initial(),
        **kwargs,
    )
    tree_flat = jnp.concatenate([tree.xs["a"][:, None], tree.xs["b"][0]], axis=1)
    assert jnp.array_equal(tree_flat, flat.xs)


@pytest.mark.parametrize(
    "save_at", [SaveAt(t_1=True), SaveAt(steps=True), SaveAt(ts=jnp.linspace(0, 1, 5))]
)
def test_adaptive_pytree_jvp_and_vjp_all_output_modes(save_at):
    def endpoint(rate):
        sol = solve_ode(
            tree_field,
            Tsit5(),
            0.0,
            1.0,
            tree_initial(),
            p=rate,
            dt_0=0.1,
            controller=IController(),
            max_steps=64,
            save_at=save_at,
        )
        return sol.xs["a"] if save_at.t_1 else sol.xs["a"][-1]

    rate = jnp.asarray(-0.5)
    _, tangent = jax.jvp(endpoint, (rate,), (jnp.ones_like(rate),))
    gradient = jax.grad(endpoint)(rate)
    assert jnp.allclose(tangent, jnp.exp(rate), rtol=5e-4)
    assert jnp.allclose(gradient, jnp.exp(rate), rtol=5e-4)


def test_vmap_over_batched_pytree_state():
    batched = {"a": jnp.asarray([1.0, 2.0, 3.0]), "b": jnp.ones((3, 2))}

    def endpoint(state):
        return solve_ode(
            lambda x: jax.tree.map(lambda leaf: -leaf, x),
            RK4(),
            0.0,
            1.0,
            state,
            dt_0=0.1,
            max_steps=10,
        ).xs

    result = jax.jit(jax.vmap(endpoint))(batched)
    assert result["a"].shape == (3,)
    assert result["b"].shape == (3, 2)
    assert jnp.allclose(result["a"], batched["a"] * jnp.exp(-1.0), rtol=2e-5)


def test_registered_dataclass_project_and_transforms():
    x_0 = State(jnp.asarray(1.0), jnp.asarray(-0.25))

    def endpoint(rate, initial=x_0):
        return solve_ode(
            lambda x: State(rate * x.position, rate * x.velocity),
            RK4(),
            0.0,
            1.0,
            initial,
            dt_0=0.05,
            max_steps=20,
            project=lambda x: State(jnp.maximum(x.position, 0.0), x.velocity),
        ).xs.position

    value, tangent = jax.jit(
        lambda rate: jax.jvp(endpoint, (rate,), (jnp.ones_like(rate),))
    )(jnp.asarray(0.3))
    gradient = jax.jit(jax.grad(endpoint))(jnp.asarray(0.3))
    curvature = jax.jit(
        jax.grad(lambda rate: jax.jvp(endpoint, (rate,), (jnp.ones_like(rate),))[1])
    )(jnp.asarray(0.3))
    batched = jax.jit(jax.vmap(endpoint))(jnp.asarray([0.1, 0.2, 0.3]))
    expected = jnp.exp(0.3)
    assert jnp.allclose(value, expected, rtol=2e-5)
    assert jnp.allclose(tangent, expected, rtol=5e-5)
    assert jnp.allclose(gradient, expected, rtol=5e-5)
    assert jnp.allclose(curvature, expected, rtol=1e-4)
    assert batched.shape == (3,)


def test_sde_pytree_matches_equivalent_flat_array_and_preserves_array_draw():
    key = jax.random.key(10)
    flat_0 = jnp.asarray([1.0, 2.0, 3.0])
    tree_0 = tree_initial()
    kwargs = dict(key=key, n_steps=16, save_at=SaveAt(steps=True))
    flat = solve_sde(
        lambda x: -0.2 * x,
        lambda x: 0.3 * jnp.ones_like(x),
        EulerMaruyama(),
        0.0,
        1.0,
        flat_0,
        **kwargs,
    )
    tree = solve_sde(
        lambda x: jax.tree.map(lambda leaf: -0.2 * leaf, x),
        lambda x: jax.tree.map(lambda leaf: 0.3 * jnp.ones_like(leaf), x),
        EulerMaruyama(),
        0.0,
        1.0,
        tree_0,
        **kwargs,
    )
    tree_flat = jnp.concatenate([tree.xs["a"][:, None], tree.xs["b"][0]], axis=1)
    assert jnp.array_equal(tree_flat, flat.xs)


def test_dae_supports_pytree_differential_and_algebraic_states():
    y_0 = {"a": jnp.asarray(1.0), "b": jnp.asarray(2.0)}
    z_0 = (jnp.asarray(0.8), jnp.asarray(1.8))

    def f(y, z, t, args, p):
        return {"a": p * z[0], "b": p * z[1]}

    def g(y, z):
        return jnp.stack([z[0] - y["a"], z[1] - y["b"]])

    sol = solve_semi_explicit_dae(
        f,
        g,
        Tsit5(),
        0.0,
        0.5,
        y_0,
        z_0,
        p=jnp.asarray(0.2),
        dt_0=0.05,
        controller=IController(),
        max_steps=32,
        save_at=SaveAt(ts=jnp.linspace(0.0, 0.5, 6)),
    )
    assert bool(sol.ok)
    assert jnp.allclose(sol.ys["a"], jnp.exp(0.2 * sol.ts), rtol=2e-4)
    assert jnp.allclose(sol.zs[0], sol.ys["a"], atol=2e-5)
    assert jnp.allclose(sol.zs[1], sol.ys["b"], atol=2e-5)


@pytest.mark.parametrize(
    ("state", "match"),
    [
        ({"x": jnp.asarray(1)}, "real floating"),
        (
            {"x": jnp.asarray(1.0, jnp.float16), "y": jnp.asarray(2.0, jnp.float32)},
            "same dtype",
        ),
        ({}, "at least one"),
        ({"x": jnp.asarray([])}, "must not be empty"),
    ],
)
def test_invalid_state_leaves_are_rejected(state, match):
    with pytest.raises((TypeError, ValueError), match=match):
        solve_ode(lambda x: x, RK4(), 0.0, 1.0, state, dt_0=0.1)


def test_field_and_project_must_preserve_structure():
    with pytest.raises(ValueError, match="same pytree structure"):
        solve_ode(
            lambda x: {"different": x["x"]},
            RK4(),
            0.0,
            1.0,
            {"x": 1.0},
            dt_0=0.1,
        )
    with pytest.raises(ValueError, match="same pytree structure"):
        solve_ode(
            lambda x: x,
            RK4(),
            0.0,
            1.0,
            {"x": 1.0},
            dt_0=0.1,
            project=lambda x: (x["x"],),
        )


def test_same_pytree_structure_and_shapes_reuse_compilation():
    @jax.jit
    def run(x):
        return solve_ode(
            lambda state: jax.tree.map(lambda leaf: -leaf, state),
            RK4(),
            0.0,
            1.0,
            x,
            dt_0=0.1,
            max_steps=10,
        ).xs

    run({"a": jnp.ones(2), "b": jnp.ones(3)})
    run({"a": 2 * jnp.ones(2), "b": 3 * jnp.ones(3)})
    assert run._cache_size() == 1
    run({"a": jnp.ones(2), "b": jnp.ones(4)})
    assert run._cache_size() == 2
