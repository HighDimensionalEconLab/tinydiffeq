import jax
import jax.numpy as jnp

from tinydiffeq import IController, SaveAt, Tsit5, solve_ode

# Hyperparameters that only change values (tolerances, dt0, x0, args) are
# pytree data leaves, and the bounded scan runs exactly max_steps iterations
# regardless of how many are accepted -- so none of these may retrace.


def field(x, t, args, p):
    return -args * x


@jax.jit
def run_steps(x0, dt0, controller, args):
    return solve_ode(
        field,
        Tsit5(),
        0.0,
        1.0,
        x0,
        args=args,
        dt0=dt0,
        controller=controller,
        max_steps=128,
        saveat=SaveAt(steps=True),
    )


def test_one_compilation_across_leaf_changes():
    base = run_steps(
        jnp.asarray(1.0), jnp.asarray(0.1), IController(rtol=1e-6, atol=1e-8), 1.0
    )
    # different curvature -> different accepted count, same compilation
    stiff = run_steps(
        jnp.asarray(1.0), jnp.asarray(0.1), IController(rtol=1e-6, atol=1e-8), 40.0
    )
    assert int(stiff.num_accepted) != int(base.num_accepted)
    # different tolerances, dt0, and x0
    run_steps(
        jnp.asarray(2.0), jnp.asarray(0.02), IController(rtol=1e-10, atol=1e-12), 3.0
    )
    run_steps(
        jnp.asarray(0.3),
        jnp.asarray(0.5),
        IController(rtol=1e-4, atol=1e-6, dtmin=1e-8, safety=0.8),
        1.0,
    )
    assert run_steps._cache_size() == 1


def test_one_compilation_ts_mode_same_grid_length():
    @jax.jit
    def run_ts(x0, saveat, args):
        return solve_ode(
            field,
            Tsit5(),
            0.0,
            1.0,
            x0,
            args=args,
            dt0=0.1,
            controller=IController(rtol=1e-8, atol=1e-10),
            max_steps=128,
            saveat=saveat,
        ).xs

    grid_a = jnp.linspace(0.0, 1.0, 11)
    grid_b = jnp.sqrt(jnp.linspace(0.0, 1.0, 11))  # same length, different knots
    run_ts(jnp.asarray(1.0), SaveAt(ts=grid_a), 1.0)
    run_ts(jnp.asarray(2.0), SaveAt(ts=grid_b), 25.0)
    assert run_ts._cache_size() == 1
