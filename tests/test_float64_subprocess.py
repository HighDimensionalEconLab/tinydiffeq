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


def test_default_x64_disabled_representative_paths_stay_float32():
    run_script(r"""
import jax
import jax.numpy as jnp

from tinydiffeq import (
    ContinuousTimeMarkovChain,
    DenseExponential,
    DiscreteMarkovChain,
    EulerMaruyama,
    IController,
    SaveAt,
    Tsit5,
    forecast_continuous_time_markov_chain,
    forecast_markov_chain,
    simulate_continuous_time_markov_chain,
    simulate_markov_chain,
    solve_linear_ode,
    solve_ode,
    solve_sde,
    solve_semi_explicit_dae,
    solve_semi_explicit_sdae,
)


assert not jax.config.x64_enabled
x_0 = jnp.asarray(1.0)
parameter = jnp.asarray(0.2)


def check_value_jvp_vjp(function, value):
    jaxpr = jax.make_jaxpr(function)(value)
    assert "f64" not in str(jaxpr), jaxpr
    primal, tangent = jax.jvp(
        function,
        (value,),
        (jax.tree.map(jnp.ones_like, value),),
    )
    _, pullback = jax.vjp(function, value)
    cotangent = pullback(jax.tree.map(jnp.ones_like, primal))[0]
    for leaf in jax.tree.leaves((primal, tangent, cotangent)):
        if jnp.issubdtype(leaf.dtype, jnp.inexact):
            assert leaf.dtype == jnp.float32, leaf.dtype
            assert jnp.all(jnp.isfinite(leaf))


def ode_endpoint(rate):
    return solve_ode(
        lambda x, t, args, p: p * x,
        Tsit5(),
        0.0,
        0.5,
        x_0,
        p=rate,
        dt_0=0.1,
        controller=IController(),
        max_steps=32,
    ).xs


def dae_endpoint(rate):
    return solve_semi_explicit_dae(
        lambda y, z, t, args, p: p * z,
        lambda y, z: z - y,
        Tsit5(),
        0.0,
        0.5,
        x_0,
        x_0,
        p=rate,
        dt_0=0.1,
        controller=IController(),
        max_steps=32,
    ).ys


def sde_endpoint(rate):
    return solve_sde(
        lambda x, t, args, p: p * x,
        lambda x, t, args, p: jnp.asarray(0.1, x.dtype) * x,
        EulerMaruyama(),
        0.0,
        0.5,
        x_0,
        p=rate,
        key=jax.random.key(1),
        n_steps=8,
    ).xs


def sdae_endpoint(rate):
    return solve_semi_explicit_sdae(
        lambda y, z, t, args, p: p * z,
        lambda y, z, t, args, p: jnp.asarray(0.1, y.dtype) * z,
        lambda y, z: z - y,
        EulerMaruyama(),
        0.0,
        0.5,
        x_0,
        x_0,
        p=rate,
        key=jax.random.key(2),
        n_steps=8,
    ).ys


def exponential_endpoint(rate):
    return solve_linear_ode(
        lambda x: rate * x,
        DenseExponential(),
        0.0,
        0.5,
        x_0,
    ).xs


for endpoint in (
    ode_endpoint,
    dae_endpoint,
    sde_endpoint,
    sdae_endpoint,
    exponential_endpoint,
):
    check_value_jvp_vjp(endpoint, parameter)


discrete = DiscreteMarkovChain(jnp.asarray([[0.8, 0.2], [0.3, 0.7]]))
continuous = ContinuousTimeMarkovChain(jnp.asarray([[-1.0, 1.0], [0.5, -0.5]]))
keys = jax.random.split(jax.random.key(3), 3)


def discrete_path(key):
    return simulate_markov_chain(
        discrete,
        jnp.int32(0),
        key=key,
        num_steps=8,
        save_at=SaveAt(steps=True),
    )


def continuous_path(key):
    return simulate_continuous_time_markov_chain(
        continuous,
        0.0,
        2.0,
        jnp.int32(0),
        key=key,
        max_jumps=32,
        save_at=SaveAt(steps=True),
    )


for path in (discrete_path, continuous_path):
    jaxpr = jax.make_jaxpr(jax.vmap(path))(keys)
    assert "f64" not in str(jaxpr), jaxpr
discrete_paths = jax.jit(jax.vmap(discrete_path))(keys)
continuous_paths = jax.jit(jax.vmap(continuous_path))(keys)
assert discrete.transition_matrix.dtype == jnp.float32
assert continuous.generator.dtype == jnp.float32
assert continuous_paths.ts.dtype == jnp.float32
assert bool(jnp.all(continuous_paths.ok))


distribution = jnp.asarray([0.4, 0.6])


def discrete_forecast(value):
    return forecast_markov_chain(
        discrete, value, num_steps=4
    ).probabilities


def continuous_forecast(value):
    return forecast_continuous_time_markov_chain(
        continuous, 0.0, 1.0, value
    ).probabilities


for forecast in (discrete_forecast, continuous_forecast):
    check_value_jvp_vjp(forecast, distribution)
    probabilities = jax.jit(forecast)(distribution)
    assert probabilities.dtype == jnp.float32
    assert jnp.allclose(jnp.sum(probabilities), 1.0, atol=1e-5)
""")


def test_sra1_with_x64_disabled_stays_float32_and_differentiable():
    run_script(r"""
import jax
import jax.numpy as jnp

from tinydiffeq import SRA1, solve_sde


assert not jax.config.x64_enabled
n = 32
x_0 = jnp.asarray(1.0)
key = jax.random.key(0)
noise = SRA1().sample_noise(x_0, key, n, jnp.asarray(1.0 / n), x_0.dtype)


def endpoint(x_0, noise):
    return solve_sde(
        lambda x: -0.5 * x,
        lambda x: 0.1 * jnp.ones_like(x),
        SRA1(),
        0.0,
        1.0,
        x_0,
        noise=noise,
        n_steps=n,
    ).xs


jaxpr = jax.make_jaxpr(endpoint)(x_0, noise)
assert "f64" not in str(jaxpr), jaxpr

value = jax.jit(endpoint)(x_0, noise)
keyed = solve_sde(
    lambda x: -0.5 * x,
    lambda x: 0.1 * jnp.ones_like(x),
    SRA1(),
    0.0,
    1.0,
    x_0,
    key=key,
    n_steps=n,
).xs
assert value.dtype == jnp.float32
assert jnp.array_equal(value, keyed)

grad_x, grad_noise = jax.grad(endpoint, argnums=(0, 1))(x_0, noise)
assert grad_x.dtype == jnp.float32
assert jnp.isfinite(grad_x)
for leaf in jax.tree.leaves(grad_noise):
    assert leaf.dtype == jnp.float32
    assert jnp.all(jnp.isfinite(leaf))
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


def test_float32_dae_and_sdae_lowerings_contain_no_float64_under_x64():
    run_script(r"""
import os
os.environ["JAX_PLATFORMS"] = "cpu"
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from tinydiffeq import (
    EulerMaruyama,
    IController,
    LMRootSolver,
    RK4,
    Tsit5,
    solve_semi_explicit_dae,
    solve_semi_explicit_sdae,
)


DTYPE = jnp.float32
T_0 = jnp.asarray(0.0, DTYPE)
T_1 = jnp.asarray(0.2, DTYPE)
DT_0 = jnp.asarray(0.1, DTYPE)
Y_0 = jnp.asarray(1.0, DTYPE)
Z_0 = jnp.asarray(1.0, DTYPE)
P = jnp.asarray(0.2, DTYPE)


def differential(y, z, t, args, p):
    return p * z


def constraint(y, z, t, args, p):
    return z - y


def dae_endpoint(p, solver, root_solver, adaptive_loop="bounded"):
    adaptive_options = {}
    if isinstance(solver, Tsit5):
        adaptive_options = {
            "controller": IController(),
            "adaptive_loop": adaptive_loop,
        }
    return solve_semi_explicit_dae(
        differential,
        constraint,
        solver,
        T_0,
        T_1,
        Y_0,
        Z_0,
        p=p,
        dt_0=DT_0,
        max_steps=4,
        root_solver=root_solver,
        **adaptive_options,
    ).ys


def sdae_endpoint(p):
    return solve_semi_explicit_sdae(
        differential,
        lambda y, z, t, args, p: jnp.zeros_like(y),
        constraint,
        EulerMaruyama(),
        T_0,
        T_1,
        Y_0,
        Z_0,
        p=p,
        key=jax.random.key(0),
        n_steps=2,
        root_solver=LMRootSolver(),
    ).ys


def with_jvp(function):
    return lambda p: jax.jvp(function, (p,), (jnp.ones_like(p),))


def with_vjp(function):
    def transformed(p):
        value, pullback = jax.vjp(function, p)
        return value, pullback(jnp.ones_like(value))[0]

    return transformed


def assert_pure_float32(name, function):
    value = function(P)
    for leaf in jax.tree.leaves(value):
        if jnp.issubdtype(leaf.dtype, jnp.inexact):
            assert leaf.dtype == DTYPE, (name, leaf.dtype)
            assert jnp.all(jnp.isfinite(leaf)), (name, leaf)

    jaxpr = str(jax.make_jaxpr(function)(P))
    assert "f64" not in jaxpr, (name, jaxpr)
    stablehlo = str(jax.jit(function).lower(P).compiler_ir("stablehlo"))
    offending = [line for line in stablehlo.splitlines() if "f64" in line]
    assert not offending, (name, offending[:10])


default_root = LMRootSolver()
configured_root = LMRootSolver(
    solver_options={
        "init_damping": 2e-3,
        "damping_decrease": 0.4,
        "damping_increase": 3.0,
    }
)
dae_cases = {
    "rk4_primal": lambda p: dae_endpoint(p, RK4(), default_root),
    "tsit5_bounded_primal": lambda p: dae_endpoint(
        p, Tsit5(), configured_root, "bounded"
    ),
    "tsit5_forward_primal": lambda p: dae_endpoint(
        p, Tsit5(), configured_root, "forward"
    ),
}
for name, function in dae_cases.items():
    assert_pure_float32(name, function)
    assert_pure_float32(name.replace("primal", "jvp"), with_jvp(function))
    if "forward" not in name:
        assert_pure_float32(name.replace("primal", "vjp"), with_vjp(function))

assert_pure_float32("sdae_primal", sdae_endpoint)
assert_pure_float32("sdae_jvp", with_jvp(sdae_endpoint))
assert_pure_float32("sdae_vjp", with_vjp(sdae_endpoint))
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
    def differential(y, z, t, args, q, algebraic_aux):
        return q * z, algebraic_aux

    return solve_semi_explicit_dae(
        differential,
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
        has_algebraic_aux=True,
    )


for dtype in (jnp.float32, jnp.float64):
    sol = dae_endpoint(dtype)
    assert sol.ys.dtype == dtype
    assert sol.zs.dtype == dtype
    assert sol.aux["value"].dtype == dtype
    if dtype == jnp.float64:
        jaxpr = jax.make_jaxpr(lambda: dae_endpoint(dtype))()
        assert "f32" not in str(jaxpr), jaxpr


def mixed_differential(y, z, t, args, p, algebraic_aux):
    return z, algebraic_aux


mixed = solve_semi_explicit_dae(
    mixed_differential,
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
    has_algebraic_aux=True,
)
assert mixed.aux["f32"].dtype == jnp.float32
assert mixed.aux["f64"].dtype == jnp.float64


def sdae_endpoint(y_0, p):
    def drift(y, z, t, args, q, algebraic_aux):
        return q * z, algebraic_aux

    def diffusion(y, z, t, args, q, algebraic_aux):
        return jnp.asarray(0.1, y.dtype) * z

    return solve_semi_explicit_sdae(
        drift,
        diffusion,
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
        has_algebraic_aux=True,
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
