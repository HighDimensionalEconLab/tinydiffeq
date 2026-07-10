import jax
import jax.numpy as jnp

from tinydiffeq import (
    RK4,
    ConstantStepSize,
    IController,
    PIController,
    SaveAt,
    Tsit5,
    solve_ode,
)

# x' = -a x, x(t) = x_0 exp(-a t): every derivative has a closed form, and
# central finite differences of the numerical solve are the exact ground
# truth for what the AD should return.

T = 1.5
GRID = jnp.linspace(0.0, T, 7)


def field(x, t, args, p):
    return -p * x


def run(a, x_0, save_at):
    return solve_ode(
        field,
        Tsit5(),
        0.0,
        T,
        x_0,
        p=a,
        dt_0=0.1,
        controller=IController(rtol=1e-11, atol=1e-13),
        max_steps=512,
        save_at=save_at,
    ).xs


def central_difference(fn, x, eps=1e-6):
    return (fn(x + eps) - fn(x - eps)) / (2.0 * eps)


A_0 = jnp.asarray(1.3)
X_0 = jnp.asarray(0.7)

# One scalar output per save_at mode, with its closed-form derivatives.
# Steps mode reduces to the final padded row (the reached endpoint). A
# reduction over only the valid prefix is discontinuous when its length
# changes with the accept/reject pattern.
CASES = (
    (
        SaveAt(t_1=True),
        lambda a, x_0: run(a, x_0, SaveAt(t_1=True)),
        lambda a, x_0: -T * x_0 * jnp.exp(-a * T),
        lambda a, x_0: jnp.exp(-a * T),
        lambda a, x_0: T**2 * x_0 * jnp.exp(-a * T),
    ),
    (
        SaveAt(steps=True),
        lambda a, x_0: run(a, x_0, SaveAt(steps=True))[-1],
        lambda a, x_0: -T * x_0 * jnp.exp(-a * T),
        lambda a, x_0: jnp.exp(-a * T),
        lambda a, x_0: T**2 * x_0 * jnp.exp(-a * T),
    ),
    (
        SaveAt(ts=GRID),
        lambda a, x_0: jnp.sum(run(a, x_0, SaveAt(ts=GRID))),
        lambda a, x_0: jnp.sum(-GRID * x_0 * jnp.exp(-a * GRID)),
        lambda a, x_0: jnp.sum(jnp.exp(-a * GRID)),
        lambda a, x_0: jnp.sum(GRID**2 * x_0 * jnp.exp(-a * GRID)),
    ),
)


def test_jvp_and_vjp_wrt_p_and_x_0_all_save_at_modes():
    for _, output, d_f_d_a, d_f_d_x_0, _ in CASES:
        f_a = lambda a, output=output: output(a, X_0)  # noqa: E731
        jvp_a = jax.jvp(f_a, (A_0,), (jnp.asarray(1.0),))[1]
        grad_a = jax.grad(f_a)(A_0)
        assert jnp.abs(jvp_a - d_f_d_a(A_0, X_0)) < 1e-6
        assert jnp.abs(grad_a - d_f_d_a(A_0, X_0)) < 1e-6
        assert jnp.abs(jvp_a - grad_a) < 1e-9  # forward == reverse
        f_x = lambda x_0, output=output: output(A_0, x_0)  # noqa: E731
        jvp_x = jax.jvp(f_x, (X_0,), (jnp.asarray(1.0),))[1]
        grad_x = jax.grad(f_x)(X_0)
        assert jnp.abs(jvp_x - d_f_d_x_0(A_0, X_0)) < 1e-6
        assert jnp.abs(grad_x - d_f_d_x_0(A_0, X_0)) < 1e-6


def test_grad_matches_finite_differences_endpoint():
    # FD is only well-posed where the output is insensitive to accept/reject
    # pattern flips at the FD epsilon scale; the endpoint at tight tolerances
    # qualifies
    endpoint = lambda a: run(a, X_0, SaveAt(t_1=True))  # noqa: E731
    fd = central_difference(endpoint, A_0)
    assert jnp.abs(jax.grad(endpoint)(A_0) - fd) < 1e-6


def test_reverse_over_forward():
    # the LM geodesic-acceleration pattern: grad of a jvp
    for _, output, _, _, second_derivative_a in CASES:
        f_a = lambda a, output=output: output(a, X_0)  # noqa: E731

        def directional(a, f_a=f_a):
            return jax.jvp(f_a, (a,), (jnp.asarray(1.0),))[1]

        got = jax.grad(directional)(A_0)
        assert bool(jnp.isfinite(got))
        assert jnp.abs(got - second_derivative_a(A_0, X_0)) < 1e-5


def test_pi_controller_jvp_vjp_and_reverse_over_forward():
    def endpoint(a):
        return solve_ode(
            field,
            Tsit5(),
            0.0,
            T,
            X_0,
            p=a,
            dt_0=0.1,
            controller=PIController(rtol=1e-11, atol=1e-13),
            max_steps=512,
        ).xs

    tangent = jnp.asarray(1.0)
    jvp = jax.jvp(endpoint, (A_0,), (tangent,))[1]
    vjp = jax.grad(endpoint)(A_0)
    reverse_over_forward = jax.grad(lambda a: jax.jvp(endpoint, (a,), (tangent,))[1])(
        A_0
    )
    exact_first = -T * X_0 * jnp.exp(-A_0 * T)
    exact_second = T**2 * X_0 * jnp.exp(-A_0 * T)
    assert jnp.abs(jvp - exact_first) < 1e-6
    assert jnp.abs(vjp - exact_first) < 1e-6
    assert jnp.abs(reverse_over_forward - exact_second) < 1e-5


def test_grad_finite_on_flat_field():
    # exact-zero error estimate: the E**(-1/order) blow-up the controller's
    # stop_gradient exists for
    def flat_endpoint(x_0):
        return solve_ode(
            lambda x: 0.0 * x,
            Tsit5(),
            0.0,
            T,
            x_0,
            dt_0=0.1,
            controller=IController(rtol=1e-8, atol=1e-8),
            max_steps=64,
        ).xs

    grad = jax.grad(flat_endpoint)(X_0)
    assert bool(jnp.isfinite(grad))
    assert jnp.abs(grad - 1.0) < 1e-12


def test_grad_finite_with_binding_project():
    def clamped_endpoint(a):
        return solve_ode(
            field,
            Tsit5(),
            0.0,
            4.0,
            jnp.asarray(2.0),
            p=a,
            dt_0=0.05,
            controller=IController(rtol=1e-8, atol=1e-10),
            max_steps=512,
            project=lambda x: jnp.maximum(x, 0.5),
        ).xs

    value = clamped_endpoint(jnp.asarray(3.0))
    assert jnp.abs(value - 0.5) < 1e-8  # the clamp binds at the endpoint
    grad = jax.grad(clamped_endpoint)(jnp.asarray(3.0))
    assert bool(jnp.isfinite(grad))


def test_vmap_over_x_0_matches_individual_solves():
    # magnitudes spanning three orders force per-lane accepted counts to
    # differ under a fixed atol, so this exercises genuinely per-lane control
    x_0_values = jnp.asarray([0.05, 1.0, 50.0])

    def endpoint(x_0):
        return solve_ode(
            field,
            Tsit5(),
            0.0,
            T,
            x_0,
            p=A_0,
            dt_0=0.1,
            controller=IController(rtol=1e-8, atol=1e-10),
            max_steps=256,
        )

    batched = jax.vmap(endpoint)(x_0_values)
    singles = [endpoint(x_0) for x_0 in x_0_values]
    counts = [int(s.num_accepted) for s in singles]
    assert len(set(counts)) > 1
    for lane, single in enumerate(singles):
        assert jnp.array_equal(batched.xs[lane], single.xs)
        assert int(batched.num_accepted[lane]) == int(single.num_accepted)
    # gradients also vmap
    grads = jax.vmap(jax.grad(lambda x_0: endpoint(x_0).xs))(x_0_values)
    assert bool(jnp.all(jnp.isfinite(grads)))


def test_vmap_over_p():
    a_batch = jnp.asarray([0.5, 1.3, 2.0])

    def endpoint(a):
        return run(a, X_0, SaveAt(t_1=True))

    batched = jax.vmap(endpoint)(a_batch)
    expected = X_0 * jnp.exp(-a_batch * T)
    assert jnp.max(jnp.abs(batched - expected)) < 1e-9


def test_grad_through_fixed_step_rk4():
    def endpoint(a):
        return solve_ode(
            field,
            RK4(),
            0.0,
            T,
            X_0,
            p=a,
            dt_0=T / 100,
            controller=ConstantStepSize(),
            max_steps=100,
        ).xs

    fd = central_difference(endpoint, A_0)
    assert jnp.abs(jax.grad(endpoint)(A_0) - fd) < 1e-7
