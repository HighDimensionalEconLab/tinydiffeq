import jax
import jax.numpy as jnp

from tinydiffeq import RK4, ConstantStepSize, IController, SaveAt, Tsit5, solve_ode

# x' = -a x, x(t) = x0 exp(-a t): every derivative has a closed form, and
# central finite differences of the numerical solve are the exact ground
# truth for what the AD should return.

T = 1.5
GRID = jnp.linspace(0.0, T, 7)


def field(x, t, args, p):
    return -p * x


def run(a, x0, saveat):
    return solve_ode(
        field,
        Tsit5(),
        0.0,
        T,
        x0,
        p=a,
        dt0=0.1,
        controller=IController(rtol=1e-11, atol=1e-13),
        max_steps=512,
        saveat=saveat,
    ).xs


def central_difference(fn, x, eps=1e-6):
    return (fn(x + eps) - fn(x - eps)) / (2.0 * eps)


A0 = jnp.asarray(1.3)
X0 = jnp.asarray(0.7)

# One scalar output per saveat mode, with its closed-form derivatives.
# steps mode reduces to the final row (== the endpoint): any reduction over
# ALL raw rows is discontinuous in the inputs because the number of duplicate
# rows changes with the accept/reject pattern -- that is exactly why
# kernels-style collocation residuals (which vanish at every state) are the
# intended consumer of steps mode.
CASES = (
    (
        SaveAt(t1=True),
        lambda a, x0: run(a, x0, SaveAt(t1=True)),
        lambda a, x0: -T * x0 * jnp.exp(-a * T),
        lambda a, x0: jnp.exp(-a * T),
        lambda a, x0: T**2 * x0 * jnp.exp(-a * T),
    ),
    (
        SaveAt(steps=True),
        lambda a, x0: run(a, x0, SaveAt(steps=True))[-1],
        lambda a, x0: -T * x0 * jnp.exp(-a * T),
        lambda a, x0: jnp.exp(-a * T),
        lambda a, x0: T**2 * x0 * jnp.exp(-a * T),
    ),
    (
        SaveAt(ts=GRID),
        lambda a, x0: jnp.sum(run(a, x0, SaveAt(ts=GRID))),
        lambda a, x0: jnp.sum(-GRID * x0 * jnp.exp(-a * GRID)),
        lambda a, x0: jnp.sum(jnp.exp(-a * GRID)),
        lambda a, x0: jnp.sum(GRID**2 * x0 * jnp.exp(-a * GRID)),
    ),
)


def test_jvp_and_vjp_wrt_p_and_x0_all_saveat_modes():
    for _, output, dF_da, dF_dx0, _ in CASES:
        f_a = lambda a, output=output: output(a, X0)  # noqa: E731
        jvp_a = jax.jvp(f_a, (A0,), (jnp.asarray(1.0),))[1]
        grad_a = jax.grad(f_a)(A0)
        assert jnp.abs(jvp_a - dF_da(A0, X0)) < 1e-6
        assert jnp.abs(grad_a - dF_da(A0, X0)) < 1e-6
        assert jnp.abs(jvp_a - grad_a) < 1e-9  # forward == reverse
        f_x = lambda x0, output=output: output(A0, x0)  # noqa: E731
        jvp_x = jax.jvp(f_x, (X0,), (jnp.asarray(1.0),))[1]
        grad_x = jax.grad(f_x)(X0)
        assert jnp.abs(jvp_x - dF_dx0(A0, X0)) < 1e-6
        assert jnp.abs(grad_x - dF_dx0(A0, X0)) < 1e-6


def test_grad_matches_finite_differences_endpoint():
    # FD is only well-posed where the output is insensitive to accept/reject
    # pattern flips at the FD epsilon scale; the endpoint at tight tolerances
    # qualifies
    endpoint = lambda a: run(a, X0, SaveAt(t1=True))  # noqa: E731
    fd = central_difference(endpoint, A0)
    assert jnp.abs(jax.grad(endpoint)(A0) - fd) < 1e-6


def test_reverse_over_forward():
    # the LM geodesic-acceleration pattern: grad of a jvp
    for _, output, _, _, d2F_da2 in CASES:
        f_a = lambda a, output=output: output(a, X0)  # noqa: E731

        def directional(a, f_a=f_a):
            return jax.jvp(f_a, (a,), (jnp.asarray(1.0),))[1]

        got = jax.grad(directional)(A0)
        assert bool(jnp.isfinite(got))
        assert jnp.abs(got - d2F_da2(A0, X0)) < 1e-5


def test_grad_finite_on_flat_field():
    # exact-zero error estimate: the E**(-1/order) blow-up the controller's
    # stop_gradient exists for
    def flat_endpoint(x0):
        return solve_ode(
            lambda x: 0.0 * x,
            Tsit5(),
            0.0,
            T,
            x0,
            dt0=0.1,
            controller=IController(rtol=1e-8, atol=1e-8),
            max_steps=64,
        ).xs

    grad = jax.grad(flat_endpoint)(X0)
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
            dt0=0.05,
            controller=IController(rtol=1e-8, atol=1e-10),
            max_steps=512,
            project=lambda x: jnp.maximum(x, 0.5),
        ).xs

    value = clamped_endpoint(jnp.asarray(3.0))
    assert jnp.abs(value - 0.5) < 1e-8  # the clamp binds at the endpoint
    grad = jax.grad(clamped_endpoint)(jnp.asarray(3.0))
    assert bool(jnp.isfinite(grad))


def test_vmap_over_x0_matches_individual_solves():
    # magnitudes spanning three orders force per-lane accepted counts to
    # differ under a fixed atol, so this exercises genuinely per-lane control
    x0s = jnp.asarray([0.05, 1.0, 50.0])

    def endpoint(x0):
        return solve_ode(
            field,
            Tsit5(),
            0.0,
            T,
            x0,
            p=A0,
            dt0=0.1,
            controller=IController(rtol=1e-8, atol=1e-10),
            max_steps=256,
        )

    batched = jax.vmap(endpoint)(x0s)
    singles = [endpoint(x0) for x0 in x0s]
    counts = [int(s.num_accepted) for s in singles]
    assert len(set(counts)) > 1
    for lane, single in enumerate(singles):
        assert jnp.array_equal(batched.xs[lane], single.xs)
        assert int(batched.num_accepted[lane]) == int(single.num_accepted)
    # gradients also vmap
    grads = jax.vmap(jax.grad(lambda x0: endpoint(x0).xs))(x0s)
    assert bool(jnp.all(jnp.isfinite(grads)))


def test_vmap_over_p():
    a_batch = jnp.asarray([0.5, 1.3, 2.0])

    def endpoint(a):
        return run(a, X0, SaveAt(t1=True))

    batched = jax.vmap(endpoint)(a_batch)
    expected = X0 * jnp.exp(-a_batch * T)
    assert jnp.max(jnp.abs(batched - expected)) < 1e-9


def test_grad_through_fixed_step_rk4():
    def endpoint(a):
        return solve_ode(
            field,
            RK4(),
            0.0,
            T,
            X0,
            p=a,
            dt0=T / 100,
            controller=ConstantStepSize(),
            max_steps=100,
        ).xs

    fd = central_difference(endpoint, A0)
    assert jnp.abs(jax.grad(endpoint)(A0) - fd) < 1e-7
