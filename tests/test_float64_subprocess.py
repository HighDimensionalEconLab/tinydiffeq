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
    RK4,
    SaveAt,
    Tsit5,
    solve_ode,
    solve_sde,
)


def f(x, t, args, p):
    return -p * x


x0 = jnp.asarray([1.0, 2.0], dtype=jnp.float64)
p = jnp.asarray(1.3, dtype=jnp.float64)

fixed = jax.make_jaxpr(
    lambda x0, p: solve_ode(
        f, RK4(), 0.0, 1.0, x0, p=p, dt0=0.01, max_steps=100
    ).xs
)(x0, p)
assert "f32" not in str(fixed), fixed

adaptive = jax.make_jaxpr(
    lambda x0, p: solve_ode(
        f,
        Tsit5(),
        0.0,
        1.0,
        x0,
        p=p,
        dt0=0.1,
        controller=IController(rtol=1e-8, atol=1e-10),
        max_steps=128,
        saveat=SaveAt(steps=True),
    ).xs
)(x0, p)
assert "f32" not in str(adaptive), adaptive

grid = jnp.linspace(0.0, 1.0, 9, dtype=jnp.float64)
interp = jax.make_jaxpr(
    lambda x0, p: solve_ode(
        f,
        Tsit5(),
        0.0,
        1.0,
        x0,
        p=p,
        dt0=0.1,
        controller=IController(rtol=1e-8, atol=1e-10),
        max_steps=128,
        saveat=SaveAt(ts=grid),
    ).xs
)(x0, p)
assert "f32" not in str(interp), interp

sde = jax.make_jaxpr(
    lambda x0, p: solve_sde(
        lambda x, t, args, q: q * x,
        lambda x, t, args, q: 0.3 * x,
        EulerMaruyama(),
        0.0,
        1.0,
        x0,
        key=jax.random.PRNGKey(0),
        n_steps=32,
        p=p,
    ).xs
)(x0, p)
assert "f32" not in str(sde), sde
""")


def test_float32_x0_under_x64_stays_float32():
    run_script(r"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from tinydiffeq import (
    EulerMaruyama,
    IController,
    RK4,
    SaveAt,
    Tsit5,
    solve_ode,
    solve_sde,
)


def f(x):
    return -x


x0 = jnp.asarray([1.0, 2.0], dtype=jnp.float32)

grid = jnp.linspace(0.0, 1.0, 5)
for saveat in (SaveAt(t1=True), SaveAt(steps=True), SaveAt(ts=grid)):
    sol = solve_ode(
        f,
        Tsit5(),
        0.0,
        1.0,
        x0,
        dt0=0.1,
        controller=IController(rtol=1e-4, atol=1e-6),
        max_steps=64,
        saveat=saveat,
    )
    assert sol.xs.dtype == jnp.float32, (saveat, sol.xs.dtype)
    assert sol.ts.dtype == jnp.float32, (saveat, sol.ts.dtype)

fixed = solve_ode(f, RK4(), 0.0, 1.0, x0, dt0=0.125, max_steps=8)
assert fixed.xs.dtype == jnp.float32
assert fixed.ts.dtype == jnp.float32

sde = solve_sde(
    lambda x: 0.5 * x,
    lambda x: 0.2 * x,
    EulerMaruyama(),
    0.0,
    1.0,
    x0,
    key=jax.random.PRNGKey(0),
    n_steps=16,
    saveat=SaveAt(steps=True),
)
assert sde.xs.dtype == jnp.float32
assert sde.ts.dtype == jnp.float32
""")
