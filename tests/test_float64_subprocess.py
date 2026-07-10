import subprocess
import sys
import textwrap

# tinydiffeq never sets jax_enable_x64 itself; these subprocess scripts pin
# that the library both propagates float64 cleanly when the application
# enables it and keeps float32 problems float32 under x64.


def run_script(script):
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr + result.stdout


def test_float64_jaxprs_contain_no_float32():
    run_script(r"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from tinydiffeq import (
    ConstantStepSize,
    EulerMaruyama,
    IController,
    PIController,
    RK4,
    SaveAt,
    Tsit5,
    solve_ode,
    solve_sde,
)


def f(x, t, args, p):
    return -p * x


x_0 = jnp.asarray([1.0, 2.0], dtype=jnp.float64)
p = jnp.asarray(1.3, dtype=jnp.float64)

fixed = jax.make_jaxpr(
    lambda x_0, p: solve_ode(
        f, RK4(), 0.0, 1.0, x_0, p=p, dt_0=0.01, max_steps=100
    ).xs
)(x_0, p)
assert "f32" not in str(fixed), fixed

adaptive = jax.make_jaxpr(
    lambda x_0, p: solve_ode(
        f,
        Tsit5(),
        0.0,
        1.0,
        x_0,
        p=p,
        dt_0=0.1,
        controller=IController(rtol=1e-8, atol=1e-10),
        max_steps=128,
        save_at=SaveAt(steps=True),
    ).xs
)(x_0, p)
assert "f32" not in str(adaptive), adaptive

pi_adaptive = jax.make_jaxpr(
    lambda x_0, p: solve_ode(
        f,
        Tsit5(),
        0.0,
        1.0,
        x_0,
        p=p,
        dt_0=0.1,
        controller=PIController(rtol=1e-8, atol=1e-10),
        max_steps=128,
    ).xs
)(x_0, p)
assert "f32" not in str(pi_adaptive), pi_adaptive

default_adaptive = jax.make_jaxpr(
    lambda x_0, p: solve_ode(
        f,
        Tsit5(),
        0.0,
        1.0,
        x_0,
        p=p,
        dt_0=0.1,
        controller=PIController(),
        max_steps=128,
    ).xs
)(x_0, p)
assert "f32" not in str(default_adaptive), default_adaptive

grid = jnp.linspace(0.0, 1.0, 9, dtype=jnp.float64)
interp = jax.make_jaxpr(
    lambda x_0, p: solve_ode(
        f,
        Tsit5(),
        0.0,
        1.0,
        x_0,
        p=p,
        dt_0=0.1,
        controller=IController(rtol=1e-8, atol=1e-10),
        max_steps=128,
        save_at=SaveAt(ts=grid),
    ).xs
)(x_0, p)
assert "f32" not in str(interp), interp

sde = jax.make_jaxpr(
    lambda x_0, p: solve_sde(
        lambda x, t, args, q: q * x,
        lambda x, t, args, q: 0.3 * x,
        EulerMaruyama(),
        0.0,
        1.0,
        x_0,
        key=jax.random.PRNGKey(0),
        n_steps=32,
        p=p,
    ).xs
)(x_0, p)
assert "f32" not in str(sde), sde
""")


def test_float32_x_0_under_x64_stays_float32():
    run_script(r"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from tinydiffeq import (
    EulerMaruyama,
    IController,
    PIController,
    RK4,
    SaveAt,
    Tsit5,
    solve_ode,
    solve_sde,
)


def f(x):
    return -x


x_0 = jnp.asarray([1.0, 2.0], dtype=jnp.float32)

grid = jnp.linspace(0.0, 1.0, 5)
for controller in (
    IController(),
    PIController(),
    IController(rtol=1e-4, atol=1e-6),
    PIController(rtol=1e-4, atol=1e-6),
):
    for save_at in (SaveAt(t_1=True), SaveAt(steps=True), SaveAt(ts=grid)):
        sol = solve_ode(
            f,
            Tsit5(),
            0.0,
            1.0,
            x_0,
            dt_0=0.1,
            controller=controller,
            max_steps=64,
            save_at=save_at,
        )
        assert sol.xs.dtype == jnp.float32, (controller, save_at, sol.xs.dtype)
        assert sol.ts.dtype == jnp.float32, (controller, save_at, sol.ts.dtype)

default_jaxpr = jax.make_jaxpr(
    lambda x_0: solve_ode(
        f,
        Tsit5(),
        0.0,
        1.0,
        x_0,
        dt_0=0.1,
        controller=PIController(),
        max_steps=64,
    ).xs
)(x_0)
assert "f64" not in str(default_jaxpr), default_jaxpr

fixed = solve_ode(f, RK4(), 0.0, 1.0, x_0, dt_0=0.125, max_steps=8)
assert fixed.xs.dtype == jnp.float32
assert fixed.ts.dtype == jnp.float32

sde = solve_sde(
    lambda x: 0.5 * x,
    lambda x: 0.2 * x,
    EulerMaruyama(),
    0.0,
    1.0,
    x_0,
    key=jax.random.PRNGKey(0),
    n_steps=16,
    save_at=SaveAt(steps=True),
)
assert sde.xs.dtype == jnp.float32
assert sde.ts.dtype == jnp.float32
""")


def test_defaults_with_x64_disabled_are_float32_and_differentiable():
    run_script(r"""
import jax
import jax.numpy as jnp

from tinydiffeq import IController, PIController, Tsit5, solve_ode


assert not jax.config.x64_enabled
x_0 = jnp.asarray(1.0)
exact = jnp.exp(-1.0)

for controller in (IController(), PIController()):
    def endpoint(x_0):
        return solve_ode(
            lambda x: -x,
            Tsit5(),
            0.0,
            1.0,
            x_0,
            dt_0=0.1,
            controller=controller,
            max_steps=64,
        ).xs

    value = jax.jit(endpoint)(x_0)
    grad = jax.grad(endpoint)(x_0)
    assert value.dtype == jnp.float32
    assert grad.dtype == jnp.float32
    assert jnp.abs(value - exact) < 1e-4, (controller, value, exact)
    assert jnp.abs(grad - exact) < 1e-4, (controller, grad, exact)
""")
