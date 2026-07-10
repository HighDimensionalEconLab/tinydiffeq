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


def test_pytree_states_preserve_float32_and_float64():
    run_script(r"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from tinydiffeq import IController, Tsit5, solve_ode


def endpoint(state):
    return solve_ode(
        lambda x: jax.tree.map(lambda leaf: -leaf, x),
        Tsit5(),
        0.0,
        1.0,
        state,
        dt_0=0.1,
        controller=IController(),
        max_steps=64,
    ).xs


for dtype, forbidden in ((jnp.float32, "f64"), (jnp.float64, "f32")):
    state = {"a": jnp.asarray(1.0, dtype), "b": (jnp.ones(2, dtype),)}
    result = endpoint(state)
    assert all(leaf.dtype == dtype for leaf in jax.tree.leaves(result))
    jaxpr = jax.make_jaxpr(endpoint)(state)
    assert forbidden not in str(jaxpr), jaxpr
""")


def test_dae_float64_and_float32_dtype_contracts():
    run_script(r"""
import os
os.environ["JAX_PLATFORMS"] = "cpu"
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from tinydiffeq import (
    IController,
    LMRootSolver,
    SaveAt,
    Tsit5,
    solve_semi_explicit_dae,
)


def f(y, z, t, args, p):
    return p * z


def g(y, z):
    return z - y


def endpoint(y_0, z_0, p):
    return solve_semi_explicit_dae(
        f,
        g,
        Tsit5(),
        0.0,
        1.0,
        y_0,
        z_0,
        p=p,
        dt_0=0.1,
        controller=IController(rtol=1e-9, atol=1e-11),
        root_solver=LMRootSolver(atol=1e-11),
        max_steps=128,
        save_at=SaveAt(t_1=True),
    ).ys


y64 = jnp.asarray(1.0, dtype=jnp.float64)
z64 = jnp.asarray(0.5, dtype=jnp.float64)
p64 = jnp.asarray(1.3, dtype=jnp.float64)
jaxpr64 = jax.make_jaxpr(endpoint)(y64, z64, p64)
assert "f32" not in str(jaxpr64), jaxpr64
value64 = endpoint(y64, z64, p64)
grad64 = jax.grad(lambda p: endpoint(y64, z64, p))(p64)
assert value64.dtype == jnp.float64
assert grad64.dtype == jnp.float64
assert jnp.abs(value64 - jnp.exp(p64)) < 1e-8
assert jnp.abs(grad64 - jnp.exp(p64)) < 1e-8

y32 = jnp.asarray(1.0, dtype=jnp.float32)
z32 = jnp.asarray(0.5, dtype=jnp.float32)
p32 = jnp.asarray(1.3, dtype=jnp.float32)


def endpoint32(y_0, z_0, p):
    return solve_semi_explicit_dae(
        f,
        g,
        Tsit5(),
        0.0,
        1.0,
        y_0,
        z_0,
        p=p,
        dt_0=0.1,
        controller=IController(),
        root_solver=LMRootSolver(),
        max_steps=128,
    ).ys

value32 = endpoint32(y32, z32, p32)
grad32 = jax.grad(lambda p: endpoint32(y32, z32, p))(p32)
assert value32.dtype == jnp.float32
assert grad32.dtype == jnp.float32
assert jnp.abs(value32 - jnp.exp(p32)) < 2e-4
assert jnp.abs(grad32 - jnp.exp(p32)) < 2e-4

# y and z each require an internally uniform dtype, but may differ.
def mixed_solve(q):
    return solve_semi_explicit_dae(
        lambda y, z, t, args, p: (p * z).astype(y.dtype),
        lambda y, z: z - y.astype(z.dtype),
        Tsit5(),
        0.0,
        0.2,
        y32,
        z64,
        p=q,
        dt_0=0.05,
        controller=IController(),
        root_solver=LMRootSolver(atol=1e-10),
        max_steps=16,
    )


p_mixed = jnp.asarray(0.2, jnp.float32)
mixed = mixed_solve(p_mixed)
mixed_jvp = jax.jvp(
    lambda q: mixed_solve(q).ys,
    (p_mixed,),
    (jnp.ones_like(p_mixed),),
)[1]
mixed_vjp = jax.grad(lambda q: mixed_solve(q).ys)(p_mixed)
assert mixed.ys.dtype == jnp.float32
assert mixed.zs.dtype == jnp.float64
assert mixed_jvp.dtype == jnp.float32
assert mixed_vjp.dtype == jnp.float32
assert jnp.allclose(mixed_jvp, mixed_vjp, rtol=1e-5, atol=1e-6)
assert mixed.ok
""")


def test_aux_dense_output_and_sdae_dtype_contracts():
    run_script(r"""
import os
os.environ["JAX_PLATFORMS"] = "cpu"
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from tinydiffeq import (
    EulerMaruyama,
    RK4,
    SaveAt,
    solve_semi_explicit_dae,
    solve_semi_explicit_sdae,
)


def dae_endpoint(dtype):
    y_0 = jnp.asarray(1.0, dtype)
    z_0 = jnp.asarray(0.8, dtype)
    p = jnp.asarray(0.2, dtype)
    grid = jnp.linspace(0.0, 1.0, 5, dtype=dtype)
    return solve_semi_explicit_dae(
        lambda y, z, t, args, q: q * z,
        lambda y, z, t, args, q: (z - y, {"value": q * z + y}),
        RK4(),
        0.0,
        1.0,
        y_0,
        z_0,
        p=p,
        dt_0=0.125,
        max_steps=8,
        save_at=SaveAt(ts=grid),
        has_aux=True,
    )


for dtype in (jnp.float32, jnp.float64):
    sol = dae_endpoint(dtype)
    assert sol.ys.dtype == dtype
    assert sol.zs.dtype == dtype
    assert sol.aux["value"].dtype == dtype
    if dtype == jnp.float64:
        jaxpr = jax.make_jaxpr(lambda: dae_endpoint(dtype))()
        assert "f32" not in str(jaxpr), jaxpr


mixed = solve_semi_explicit_dae(
    lambda y, z: z,
    lambda y, z: (z - y, {"f32": z.astype(jnp.float32), "f64": z}),
    RK4(),
    0.0,
    0.25,
    jnp.asarray(1.0, jnp.float64),
    jnp.asarray(1.0, jnp.float64),
    dt_0=0.125,
    max_steps=2,
    save_at=SaveAt(ts=jnp.linspace(0.0, 0.25, 3, dtype=jnp.float64)),
    has_aux=True,
)
assert mixed.aux["f32"].dtype == jnp.float32
assert mixed.aux["f64"].dtype == jnp.float64


def sdae_endpoint(y_0, p):
    return solve_semi_explicit_sdae(
        lambda y, z, t, args, q: q * z,
        lambda y, z: jnp.asarray(0.1, y.dtype) * z,
        lambda y, z, t, args, q: (z - y, {"value": q * z}),
        EulerMaruyama(),
        0.0,
        1.0,
        y_0,
        y_0,
        p=p,
        key=jax.random.key(0),
        n_steps=8,
        has_aux=True,
    )


for dtype in (jnp.float32, jnp.float64):
    y_0 = jnp.asarray(1.0, dtype)
    p = jnp.asarray(0.2, dtype)
    sol = sdae_endpoint(y_0, p)
    assert sol.ys.dtype == dtype
    assert sol.zs.dtype == dtype
    assert sol.aux["value"].dtype == dtype
    if dtype == jnp.float64:
        jaxpr = jax.make_jaxpr(sdae_endpoint)(y_0, p)
        assert "f32" not in str(jaxpr), jaxpr
""")
