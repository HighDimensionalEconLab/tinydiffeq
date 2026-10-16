"""Rodas5P stage kernel shared by the ODE and semi-explicit DAE solvers.

The tableau, embedded estimator, and dense-output coefficients are from
Steinebach (2023), *Construction of Rosenbrock--Wanner method Rodas5P and
numerical benchmarks within the Julia Differential Equations package*:
https://doi.org/10.1007/s10543-023-00967-x

This implementation follows SciML's authoritative open-source implementation,
especially ``Rodas5PTableau`` and the consolidated Rosenbrock step kernel:
https://github.com/SciML/OrdinaryDiffEq.jl/blob/master/lib/OrdinaryDiffEqRosenbrock/src/rosenbrock_tableaus.jl
https://github.com/SciML/OrdinaryDiffEq.jl/blob/master/lib/OrdinaryDiffEqRosenbrock/src/rosenbrock_perform_step.jl

SciML's OrdinaryDiffEq.jl is MIT-licensed. The direct source references are
kept here so future coefficient or stage-kernel changes can be checked against
the implementation maintained by the method's originating ecosystem.
"""

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
from jax.flatten_util import ravel_pytree

GAMMA = 0.21193756319429014

A = (
    (),
    (3.0,),
    (2.849394379747939, 0.45842242204463923),
    (-6.954028509809101, 2.489845061869568, -10.358996098473584),
    (
        2.8029986275628964,
        0.5072464736228206,
        -0.3988312541770524,
        -0.04721187230404641,
    ),
    (
        -7.502846399306121,
        2.561846144803919,
        -11.627539656261098,
        -0.18268767659942256,
        0.030198172008377946,
    ),
    (
        -7.502846399306121,
        2.561846144803919,
        -11.627539656261098,
        -0.18268767659942256,
        0.030198172008377946,
        1.0,
    ),
    (
        -7.502846399306121,
        2.561846144803919,
        -11.627539656261098,
        -0.18268767659942256,
        0.030198172008377946,
        1.0,
        1.0,
    ),
)

C = (
    (),
    (-14.155112264123755,),
    (-17.97296035885952, -2.859693295451294),
    (147.12150275711716, -1.41221402718213, 71.68940251302358),
    (
        165.43517024871676,
        -0.4592823456491126,
        42.90938336958603,
        -5.961986721573306,
    ),
    (
        24.854864614690072,
        -3.0009227002832186,
        47.4931110020768,
        5.5814197821558125,
        -0.6610691825249471,
    ),
    (
        30.91273214028599,
        -3.1208243349937974,
        77.79954646070892,
        34.28646028294783,
        -19.097331116725623,
        -28.087943162872662,
    ),
    (
        37.80277123390563,
        -3.2571969029072276,
        112.26918849496327,
        66.9347231244047,
        -40.06618937091002,
        -54.66780262877968,
        -9.48861652309627,
    ),
)

STAGE_TIMES = (
    0.0,
    0.6358126895828704,
    0.4095798393397535,
    0.9769306725060716,
    0.4288403609558664,
    1.0,
    1.0,
    1.0,
)

TIME_DERIVATIVE = (
    0.21193756319429014,
    -0.42387512638858027,
    -0.3384627126235924,
    1.8046452872882734,
    2.325825639765069,
    0.0,
    0.0,
    0.0,
)

DENSE = (
    (
        25.948786856663858,
        -2.5579724845846235,
        10.433815404888879,
        -2.3679251022685204,
        0.524948541321073,
        1.1241088310450404,
        0.4272876194431874,
        -0.17202221070155493,
    ),
    (
        -9.91568850695171,
        -0.9689944594115154,
        3.0438037242978453,
        -24.495224566215796,
        20.176138334709044,
        15.98066361424651,
        -6.789040303419874,
        -6.710236069923372,
    ),
    (
        11.419903575922262,
        2.8879645146136994,
        72.92137995996029,
        80.12511834622643,
        -52.072871366152654,
        -59.78993625266729,
        -0.15582684282751913,
        4.883087185713722,
    ),
)

# Rodas5P is stiffly accurate: SciML constructs b from the final A row and
# appends one. Its embedded difference has only the final stage coefficient.
SOLUTION = (*A[-1], 1.0)


def _sum_vectors(vectors, coefficients, model):
    value = jnp.zeros_like(model)
    for coefficient, vector in zip(coefficients, vectors, strict=True):
        value = value + jnp.asarray(coefficient, model.dtype) * vector
    return value


def _safe_lu_factor(matrix):
    """Factor once and replace an unusable factor before any stage solve."""
    lu, pivots = jsp_linalg.lu_factor(matrix, check_finite=False)
    diagonal = jnp.diag(lu)
    ok = jax.lax.stop_gradient(
        jnp.all(jnp.isfinite(lu)) & jnp.all(jnp.abs(diagonal) > 0.0)
    )
    identity = jnp.eye(matrix.shape[0], dtype=matrix.dtype)
    safe_lu = jnp.where(ok, lu, identity)
    safe_pivots = jnp.where(
        ok,
        pivots,
        jnp.arange(matrix.shape[0], dtype=pivots.dtype),
    )
    return (safe_lu, safe_pivots), ok


@jax.custom_jvp
def _factored_solve(matrix, lu, pivots, rhs):
    """Solve with an implicit derivative that does not differentiate pivots."""
    return jsp_linalg.lu_solve((lu, pivots), rhs, check_finite=False)


@_factored_solve.defjvp
def _factored_solve_jvp(primals, tangents):
    matrix, lu, pivots, rhs = primals
    matrix_dot, _, _, rhs_dot = tangents
    value = _factored_solve(matrix, lu, pivots, rhs)
    value_dot = _factored_solve(
        matrix,
        lu,
        pivots,
        rhs_dot - matrix_dot @ value,
    )
    return value, value_dot


def rodas5p_step(field, t, state, dt, mass_diagonal, project):
    """Take one Rodas5P attempt.

    ``field`` maps a state pytree and scalar time to a pytree with the same
    flattened size and coordinate order. ``mass_diagonal`` is the flattened
    constant diagonal of ``M``. Stage increments and the three dense
    coefficients are returned in the state structure. AD differentiates the
    complete discrete step. Linear solves use their implicit derivative, so
    pivot choices are not differentiated.
    """
    state_flat, unravel = ravel_pytree(state)
    dtype = state_flat.dtype
    mass_diagonal = jnp.asarray(mass_diagonal, dtype)

    def field_flat(value, time):
        output, _ = ravel_pytree(field(unravel(value), time))
        return output

    field_initial = field_flat(state_flat, t)
    jacobian = jax.jacfwd(lambda value: field_flat(value, t))(state_flat)
    time_derivative = jax.jacfwd(lambda time: field_flat(state_flat, time))(t)
    gamma = jnp.asarray(GAMMA, dtype)
    matrix = jnp.diag(mass_diagonal / (gamma * dt)) - jacobian
    (lu, pivots), factor_ok = _safe_lu_factor(matrix)

    def linear_solve(rhs):
        return _factored_solve(matrix, lu, pivots, rhs)

    stages = []
    first_rhs = (
        field_initial + dt * jnp.asarray(TIME_DERIVATIVE[0], dtype) * time_derivative
    )
    stages.append(linear_solve(first_rhs))
    for stage in range(1, 8):
        stage_increment = _sum_vectors(stages, A[stage], state_flat)
        stage_state = state_flat + stage_increment
        stage_field = field_flat(
            stage_state,
            t + jnp.asarray(STAGE_TIMES[stage], dtype) * dt,
        )
        coupling = _sum_vectors(stages, C[stage], state_flat) / dt
        rhs = (
            stage_field
            + dt * jnp.asarray(TIME_DERIVATIVE[stage], dtype) * time_derivative
            + mass_diagonal * coupling
        )
        stages.append(linear_solve(rhs))

    candidate_flat = state_flat + _sum_vectors(stages, SOLUTION, state_flat)
    candidate = project(unravel(candidate_flat))
    candidate_flat, _ = ravel_pytree(candidate)
    error = unravel(stages[-1])
    dense = tuple(
        unravel(_sum_vectors(stages, coefficients, state_flat))
        for coefficients in DENSE
    )
    finite = (
        jnp.all(jnp.isfinite(field_initial))
        & jnp.all(jnp.isfinite(jacobian))
        & jnp.all(jnp.isfinite(time_derivative))
        & jnp.all(jnp.isfinite(candidate_flat))
        & jnp.all(jnp.isfinite(stages[-1]))
    )
    for stage in stages:
        finite = finite & jnp.all(jnp.isfinite(stage))
    for coefficient in dense:
        for leaf in jax.tree.leaves(coefficient):
            finite = finite & jnp.all(jnp.isfinite(leaf))
    ok = factor_ok & finite
    safe_candidate = jax.tree.map(
        lambda value, fallback: jnp.where(ok, value, fallback),
        candidate,
        state,
    )
    safe_error = jax.tree.map(
        lambda value: jnp.where(ok, value, jnp.full_like(value, jnp.inf)),
        error,
    )
    safe_dense = tuple(
        jax.tree.map(lambda value: jnp.where(ok, value, jnp.zeros_like(value)), item)
        for item in dense
    )
    return safe_candidate, safe_error, safe_dense, ok


def rodas_dense_value(theta, left, right, dense):
    """Evaluate SciML's fourth-order Rodas5P continuous extension."""
    q_1, q_2, q_3 = dense

    def evaluate(value, x_0, x_1, a, b, c):
        return (1.0 - value) * x_0 + value * (
            x_1 + (1.0 - value) * (a + value * (b + value * c))
        )

    if jax.tree.structure(theta) == jax.tree.structure(left):
        return jax.tree.map(evaluate, theta, left, right, q_1, q_2, q_3)
    return jax.tree.map(
        lambda x_0, x_1, a, b, c: evaluate(theta, x_0, x_1, a, b, c),
        left,
        right,
        q_1,
        q_2,
        q_3,
    )


def rodas_dense_endpoint_derivatives(left, right, dense, dt):
    """Return time derivatives of the Rodas polynomial at both endpoints."""
    q_1, q_2, q_3 = dense
    left_dot = jax.tree.map(
        lambda x_0, x_1, a: (x_1 - x_0 + a) / dt,
        left,
        right,
        q_1,
    )
    right_dot = jax.tree.map(
        lambda x_0, x_1, a, b, c: (x_1 - x_0 - a - b - c) / dt,
        left,
        right,
        q_1,
        q_2,
        q_3,
    )
    return left_dot, right_dot
