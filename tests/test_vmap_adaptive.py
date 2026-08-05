import jax
import jax.numpy as jnp
import pytest

from tinydiffeq import (
    IController,
    Rodas5P,
    SaveAt,
    Tsit5,
    solve_ode,
    solve_semi_explicit_dae,
)

# Under vmap, a batched-predicate lax.cond lowers to a select that executes
# both branches, so without the scalar unvmap_all gates the adaptive loops
# would run every max_steps attempt slot for every lane. These tests pin the
# gates' semantics (per-lane adaptivity and results unchanged) and the work
# skipping itself.

X_0S = jnp.linspace(0.5, 2.0, 8)


def adaptive_solve(x_0, max_steps=128, save_at=None):
    return solve_ode(
        lambda x, t, args, p: p * x * (1.0 - x),
        Tsit5(),
        0.0,
        4.0,
        x_0,
        p=jnp.asarray(1.0),
        dt_0=0.2,
        controller=IController(),
        max_steps=max_steps,
        save_at=save_at,
    )


def test_vmapped_adaptive_matches_scalar_lanes():
    batched = jax.jit(jax.vmap(lambda x: adaptive_solve(x).xs))(X_0S)
    stacked = jnp.stack([adaptive_solve(X_0S[i]).xs for i in range(len(X_0S))])
    assert jnp.allclose(batched, stacked, rtol=1e-12, atol=1e-12)
    counts = jax.jit(jax.vmap(lambda x: adaptive_solve(x).num_steps))(X_0S)
    scalar_counts = jnp.stack(
        [adaptive_solve(X_0S[i]).num_steps for i in range(len(X_0S))]
    )
    # Per-lane adaptivity: heterogeneous attempt counts survive batching.
    assert jnp.array_equal(counts, scalar_counts)
    assert int(counts.min()) != int(counts.max())


def test_vmapped_adaptive_is_budget_invariant():
    small = jax.jit(jax.vmap(lambda x: adaptive_solve(x, max_steps=64).xs))(X_0S)
    large = jax.jit(jax.vmap(lambda x: adaptive_solve(x, max_steps=512).xs))(X_0S)
    assert jnp.array_equal(small, large)


def test_vmapped_adaptive_skips_frozen_tail():
    calls = []

    def field(x, t, args, p):
        jax.debug.callback(lambda: calls.append(1))
        return p * x * (1.0 - x)

    def solve(x_0):
        return solve_ode(
            field,
            Tsit5(),
            0.0,
            4.0,
            x_0,
            p=jnp.asarray(1.0),
            dt_0=0.2,
            controller=IController(),
            max_steps=256,
        ).xs

    result = jax.block_until_ready(jax.vmap(solve)(X_0S))
    assert jnp.all(jnp.isfinite(result))
    # ~20 attempts x 7 stages of actual work; without the scalar gates every
    # one of the 256 slots would evaluate the field (>1700 calls).
    assert len(calls) < 600, len(calls)


def test_vmapped_adaptive_keeps_scalar_conds_in_jaxpr():
    jaxpr = str(jax.make_jaxpr(jax.vmap(lambda x: adaptive_solve(x).xs))(X_0S))
    assert "cond[" in jaxpr
    assert "unvmap_all" in jaxpr


def test_grad_through_vmapped_adaptive_matches_scalar():
    def batched_loss(x_0s):
        return jnp.sum(jax.vmap(lambda x: adaptive_solve(x).xs)(x_0s) ** 2)

    def scalar_loss(x_0s):
        return sum(adaptive_solve(x_0s[i]).xs ** 2 for i in range(len(X_0S)))

    grad_batched = jax.jit(jax.grad(batched_loss))(X_0S)
    grad_scalar = jax.grad(scalar_loss)(X_0S)
    assert jnp.allclose(grad_batched, grad_scalar, rtol=1e-10, atol=1e-12)
    tangent = jax.jvp(batched_loss, (X_0S,), (jnp.ones_like(X_0S),))[1]
    assert jnp.isfinite(tangent)


def test_masked_residuals_on_vmapped_adaptive_output():
    # The collocation pattern from the docs: evaluate the pointwise residual
    # on every padded row, zero the tail with `accepted`, normalize by the
    # inert accepted count. fill="last" keeps padded rows finite, so the
    # single where is safe for values and gradients.
    def rollout(p):
        return jax.vmap(
            lambda x: solve_ode(
                lambda x, t, args, p: p * x * (1.0 - x),
                Tsit5(),
                0.0,
                4.0,
                x,
                p=p,
                dt_0=0.2,
                controller=IController(),
                max_steps=64,
                save_at=SaveAt(steps=True),
            )
        )(X_0S)

    def residual(p):
        sol = rollout(p)
        rows = p * sol.xs * (1.0 - sol.xs) - (sol.xs - 1.0)
        masked = jnp.where(sol.accepted, rows, 0.0)
        count = jax.lax.stop_gradient(sol.accepted.sum())
        scaled = masked / jnp.sqrt(count.astype(masked.dtype))
        return jnp.where(sol.ok[:, None], scaled, jnp.inf).reshape(-1)

    p = jnp.asarray(1.2)
    value = jax.jit(residual)(p)
    assert value.shape == (len(X_0S) * 65,)
    assert bool(jnp.all(jnp.isfinite(value)))

    # Padded rows contribute exactly zero: the flattened sum of squares
    # equals the accepted-rows-only sum built lane by lane outside jit.
    sol = rollout(p)
    per_lane = 0.0
    for i in range(len(X_0S)):
        lane_rows = p * sol.xs[i] * (1.0 - sol.xs[i]) - (sol.xs[i] - 1.0)
        per_lane += float(jnp.sum(lane_rows[sol.accepted[i]] ** 2))
    count = float(sol.accepted.sum())
    assert jnp.allclose(jnp.sum(value**2), per_lane / count, rtol=1e-12)

    grad = jax.jit(jax.grad(lambda p: jnp.sum(residual(p) ** 2)))(p)
    assert bool(jnp.isfinite(grad))


@pytest.mark.parametrize("solver", [Tsit5(), Rodas5P()])
def test_vmapped_adaptive_dae_matches_scalar_lanes(solver):
    y_0s = jnp.linspace(0.8, 1.6, 4)

    def solve(y_0):
        return solve_semi_explicit_dae(
            lambda y, z, t, args, p: p * z,
            lambda y, z: z - y,
            solver,
            0.0,
            1.0,
            y_0,
            jnp.asarray(0.5),
            p=jnp.asarray(-0.4),
            dt_0=0.1,
            controller=IController(),
            max_steps=64,
            save_at=SaveAt(t_1=True),
        )

    batched = jax.jit(jax.vmap(lambda y: solve(y).ys))(y_0s)
    stacked = jnp.stack([solve(y_0s[i]).ys for i in range(len(y_0s))])
    assert jnp.allclose(batched, stacked, rtol=1e-12, atol=1e-12)
    ok = jax.jit(jax.vmap(lambda y: solve(y).ok))(y_0s)
    assert bool(jnp.all(ok))
