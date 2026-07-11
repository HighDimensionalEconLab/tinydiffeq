"""Primal finite-state discrete- and continuous-time Markov simulation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from tinydiffeq._tree import asarray_state, prepend
from tinydiffeq.exponential import (
    AdaptiveKrylovExponential,
    DenseExponential,
    KrylovExponential,
    solve_linear_ode,
)
from tinydiffeq.save_at import SaveAt
from tinydiffeq.solution import Solution


def _concrete_square_matrix(matrix, name):
    value = jnp.asarray(matrix)
    if value.ndim != 2 or value.shape[0] != value.shape[1] or value.shape[0] < 1:
        raise ValueError(f"{name} must be a nonempty square matrix")
    if not jnp.issubdtype(value.dtype, jnp.floating):
        raise TypeError(f"{name} must have a real floating dtype")
    try:
        host = value.tolist()
    except Exception as error:
        raise TypeError(
            f"{name} must be concrete; prepare the Markov chain outside jit/vmap"
        ) from error
    return value, host


def _alias_tables(probabilities, dtype):
    """Build Vose alias tables row by row on the host."""
    num_states = len(probabilities)
    probability_rows = []
    alias_rows = []
    for row in probabilities:
        scaled = [num_states * probability for probability in row]
        small = [index for index, value in enumerate(scaled) if value < 1.0]
        large = [index for index, value in enumerate(scaled) if value >= 1.0]
        probability = [0.0] * num_states
        alias = list(range(num_states))
        while small and large:
            small_index = small.pop()
            large_index = large.pop()
            probability[small_index] = scaled[small_index]
            alias[small_index] = large_index
            scaled[large_index] -= 1.0 - scaled[small_index]
            destination = small if scaled[large_index] < 1.0 else large
            destination.append(large_index)
        for index in small + large:
            probability[index] = 1.0
            alias[index] = index
        probability_rows.append(probability)
        alias_rows.append(alias)
    return (
        jnp.asarray(probability_rows, dtype=dtype),
        jnp.asarray(alias_rows, dtype=jnp.int32),
    )


def _normalize_transition_rows(host, name):
    normalized = []
    for row in host:
        if any(not math.isfinite(value) for value in row):
            raise ValueError(f"{name} must contain only finite values")
        if any(value < 0.0 for value in row):
            raise ValueError(f"{name} entries must be nonnegative")
        total = sum(row)
        if not total > 0.0:
            raise ValueError(f"every {name} row must have positive mass")
        normalized.append([value / total for value in row])
    return normalized


@jax.tree_util.register_pytree_node_class
class DiscreteMarkovChain:
    """Prepared dense homogeneous finite-state transition matrix.

    Construction validates and normalizes the rows and builds Vose alias tables.
    Construct once outside transformed code, then pass the object through ``jit``
    or as a shared ``vmap`` argument.
    """

    def __init__(self, transition_matrix):
        value, host = _concrete_square_matrix(transition_matrix, "transition_matrix")
        normalized = _normalize_transition_rows(host, "transition_matrix")
        self.transition_matrix = jnp.asarray(normalized, dtype=value.dtype)
        self.alias_probability, self.alias_index = _alias_tables(
            normalized, value.dtype
        )

    @property
    def num_states(self):
        return self.transition_matrix.shape[0]

    def tree_flatten(self):
        return (
            self.transition_matrix,
            self.alias_probability,
            self.alias_index,
        ), None

    @classmethod
    def tree_unflatten(cls, auxiliary, children):
        instance = object.__new__(cls)
        (
            instance.transition_matrix,
            instance.alias_probability,
            instance.alias_index,
        ) = children
        return instance


@jax.tree_util.register_pytree_node_class
class ContinuousTimeMarkovChain:
    """Prepared dense homogeneous finite-state generator matrix.

    Off-diagonal entries must be nonnegative and each row must sum to zero.
    Zero rows are absorbing states. Construction extracts exit rates and builds
    alias tables for the embedded jump chain; construct outside transformed code.
    """

    def __init__(self, generator):
        value, host = _concrete_square_matrix(generator, "generator")
        num_states = value.shape[0]
        tolerance = 100.0 * float(jnp.finfo(value.dtype).eps)
        embedded = []
        rates = []
        for row_index, row in enumerate(host):
            if any(not math.isfinite(entry) for entry in row):
                raise ValueError("generator must contain only finite values")
            off_diagonal = [
                entry if column != row_index else 0.0
                for column, entry in enumerate(row)
            ]
            if any(entry < 0.0 for entry in off_diagonal):
                raise ValueError("generator off-diagonal entries must be nonnegative")
            rate = sum(off_diagonal)
            scale = max(1.0, rate, abs(row[row_index]))
            if abs(row[row_index] + rate) > tolerance * scale:
                raise ValueError("generator rows must sum to zero")
            rates.append(rate)
            if rate == 0.0:
                probabilities = [0.0] * num_states
                probabilities[row_index] = 1.0
            else:
                probabilities = [entry / rate for entry in off_diagonal]
            embedded.append(probabilities)
        self.generator = value
        self.exit_rates = jnp.asarray(rates, dtype=value.dtype)
        self.alias_probability, self.alias_index = _alias_tables(embedded, value.dtype)

    @property
    def num_states(self):
        return self.generator.shape[0]

    def tree_flatten(self):
        return (
            self.generator,
            self.exit_rates,
            self.alias_probability,
            self.alias_index,
        ), None

    @classmethod
    def tree_unflatten(cls, auxiliary, children):
        instance = object.__new__(cls)
        (
            instance.generator,
            instance.exit_rates,
            instance.alias_probability,
            instance.alias_index,
        ) = children
        return instance


@jax.tree_util.register_pytree_node_class
class MatrixFreeContinuousTimeMarkovChain:
    """Fixed CTMC forward generator represented by a pytree linear action.

    ``forward_generator(probabilities)`` must return the same probability-pytree
    structure and dtype and represent the forward equation ``dπ/dt = L(π)``.
    The callable is static JAX structure; close only over fixed model data or put
    changing arrays inside a callable pytree.
    """

    def __init__(self, forward_generator):
        if not callable(forward_generator):
            raise TypeError("forward_generator must be callable")
        self.forward_generator = forward_generator

    def tree_flatten(self):
        return (), self.forward_generator

    @classmethod
    def tree_unflatten(cls, forward_generator, children):
        instance = object.__new__(cls)
        instance.forward_generator = forward_generator
        return instance


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SequentialMarkov:
    """Chronological scan method; ``unroll=1`` is the CPU-oriented default."""

    unroll: int = field(default=1, metadata=dict(static=True))

    def __post_init__(self):
        if not isinstance(self.unroll, int) or isinstance(self.unroll, bool):
            raise TypeError("SequentialMarkov.unroll must be a positive int")
        if self.unroll < 1:
            raise ValueError("SequentialMarkov.unroll must be a positive int")


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class AssociativeMarkov:
    """Parallel-prefix random-map composition method."""


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class MatrixPowerMarkov:
    """Binary matrix powering for a DTMC distribution at one endpoint."""


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class MarkovDistribution:
    """Forecast probability mass, evaluation steps/times, and validity flag."""

    ts: jax.Array
    probabilities: Any
    ok: jax.Array


def _validate_simulation_inputs(chain, state_0, count, count_name, save_at):
    if not isinstance(count, int) or isinstance(count, bool):
        raise TypeError(f"{count_name} must be a static Python int")
    if count < 1:
        raise ValueError(f"{count_name} must be at least 1")
    if save_at is None:
        save_at = SaveAt(t_1=True)
    if save_at.steps and save_at.fill != "last":
        raise ValueError(
            'Markov integer states require SaveAt(steps=True, fill="last")'
        )
    state_0 = jnp.asarray(state_0)
    if state_0.ndim != 0 or not jnp.issubdtype(state_0.dtype, jnp.integer):
        raise TypeError("state_0 must be a scalar integer state index")
    valid = (state_0 >= 0) & (state_0 < chain.num_states)
    safe_state = jnp.clip(state_0, 0, chain.num_states - 1).astype(jnp.int32)
    return safe_state, valid, save_at


def _alias_sample(chain, state, uniform):
    scaled = uniform * chain.num_states
    column = jnp.minimum(scaled.astype(jnp.int32), chain.num_states - 1)
    fraction = scaled - column
    return jnp.where(
        fraction < chain.alias_probability[state, column],
        column,
        chain.alias_index[state, column],
    ).astype(jnp.int32)


def _random_maps(chain, uniforms):
    scaled = uniforms * chain.num_states
    columns = jnp.minimum(scaled.astype(jnp.int32), chain.num_states - 1)
    fractions = scaled - columns
    probabilities = jnp.take(chain.alias_probability, columns, axis=1).T
    aliases = jnp.take(chain.alias_index, columns, axis=1).T
    return jnp.where(fractions[:, None] < probabilities, columns[:, None], aliases)


def _compose_maps(earlier, later):
    return jnp.take_along_axis(later, earlier, axis=-1)


def _simulate_discrete_states(chain, state_0, uniforms, method):
    if isinstance(method, SequentialMarkov):

        def step(state, uniform):
            next_state = _alias_sample(chain, state, uniform)
            return next_state, next_state

        return jax.lax.scan(step, state_0, uniforms, unroll=method.unroll)[1]
    if isinstance(method, AssociativeMarkov):
        maps = _random_maps(chain, uniforms)
        prefixes = jax.lax.associative_scan(_compose_maps, maps, axis=0)
        return prefixes[:, state_0]
    raise TypeError("method must be SequentialMarkov or AssociativeMarkov")


def simulate_markov_chain(
    chain,
    state_0,
    *,
    key,
    num_steps,
    method=None,
    save_at=None,
):
    """Simulate a primal finite-state homogeneous discrete-time Markov chain.

    ``state_0`` is one scalar integer index and ``key`` one JAX key. Use
    ``jax.vmap`` over initial states and independently split keys for ensembles.
    ``num_steps`` is static. ``SequentialMarkov`` is the CPU-oriented default;
    ``AssociativeMarkov`` composes sampled state maps in parallel and returns the
    identical path for the same key. ``SaveAt`` selects the endpoint, all steps,
    or a one-dimensional array of integer step indices.
    """
    if not isinstance(chain, DiscreteMarkovChain):
        raise TypeError("chain must be a DiscreteMarkovChain")
    if method is None:
        method = SequentialMarkov()
    state_0, initial_ok, save_at = _validate_simulation_inputs(
        chain, state_0, num_steps, "num_steps", save_at
    )
    uniforms = jax.random.uniform(
        key, (num_steps,), dtype=chain.transition_matrix.dtype
    )
    step_states = _simulate_discrete_states(chain, state_0, uniforms, method)
    all_states = prepend(state_0, step_states)
    count = jnp.asarray(num_steps, jnp.int32)
    if save_at.t_1:
        return Solution(ts=count, xs=step_states[-1], ok=initial_ok, num_accepted=count)
    if save_at.steps:
        return Solution(
            ts=jnp.arange(num_steps + 1, dtype=jnp.int32),
            xs=all_states,
            ok=initial_ok,
            num_accepted=count,
            accepted=jnp.ones((num_steps + 1,), dtype=bool),
        )
    query_steps = jnp.asarray(save_at.ts)
    if query_steps.ndim != 1 or not jnp.issubdtype(query_steps.dtype, jnp.integer):
        raise TypeError("discrete SaveAt.ts must be a one-dimensional integer array")
    queries_ok = jnp.all((query_steps >= 0) & (query_steps <= num_steps))
    safe_queries = jnp.clip(query_steps, 0, num_steps).astype(jnp.int32)
    return Solution(
        ts=query_steps,
        xs=all_states[safe_queries],
        ok=initial_ok & queries_ok,
        num_accepted=count,
    )


def _simulate_continuous_events(chain, state_0, exponentials, uniforms, method):
    safe_rates = jnp.where(chain.exit_rates > 0, chain.exit_rates, jnp.inf)
    if isinstance(method, SequentialMarkov):

        def step(carry, random_values):
            state, time = carry
            exponential, uniform = random_values
            holding_time = jnp.where(
                chain.exit_rates[state] > 0,
                exponential / safe_rates[state],
                jnp.inf,
            )
            next_state = _alias_sample(chain, state, uniform)
            next_time = time + holding_time
            return (next_state, next_time), (next_state, next_time)

        return jax.lax.scan(
            step,
            (state_0, jnp.asarray(0.0, exponentials.dtype)),
            (exponentials, uniforms),
            unroll=method.unroll,
        )[1]
    if isinstance(method, AssociativeMarkov):
        state_maps = _random_maps(chain, uniforms)
        holding_times = jnp.where(
            chain.exit_rates[None, :] > 0,
            exponentials[:, None] / safe_rates[None, :],
            jnp.inf,
        )

        def compose(earlier, later):
            earlier_states, earlier_times = earlier
            later_states, later_times = later
            next_states = _compose_maps(earlier_states, later_states)
            next_times = earlier_times + jnp.take_along_axis(
                later_times, earlier_states, axis=-1
            )
            return next_states, next_times

        state_prefixes, time_prefixes = jax.lax.associative_scan(
            compose, (state_maps, holding_times), axis=0
        )
        return state_prefixes[:, state_0], time_prefixes[:, state_0]
    raise TypeError("method must be SequentialMarkov or AssociativeMarkov")


def simulate_continuous_time_markov_chain(
    chain,
    t_0,
    t_1,
    state_0,
    *,
    key,
    max_jumps,
    method=None,
    save_at=None,
):
    """Simulate a primal finite-state CTMC with Gillespie's direct recurrence.

    ``max_jumps`` is a static bound. Endpoint output is the default;
    ``SaveAt(steps=True)`` returns the padded event path and ``SaveAt(ts=...)``
    evaluates its right-continuous piecewise-constant state. ``sol.ok`` is false
    if the jump budget does not cover ``t_1``. Associative execution composes
    state and holding-time maps; states agree with sequential execution for the
    same key, while event times can differ by floating-point reassociation.
    """
    if not isinstance(chain, ContinuousTimeMarkovChain):
        raise TypeError("chain must be a ContinuousTimeMarkovChain")
    if method is None:
        method = SequentialMarkov()
    state_0, initial_ok, save_at = _validate_simulation_inputs(
        chain, state_0, max_jumps, "max_jumps", save_at
    )
    dtype = chain.generator.dtype
    t_0 = jnp.asarray(t_0, dtype)
    t_1 = jnp.asarray(t_1, dtype)
    time_ok = t_1 >= t_0
    exponential_key, transition_key = jax.random.split(key)
    exponentials = jax.random.exponential(exponential_key, (max_jumps,), dtype=dtype)
    uniforms = jax.random.uniform(transition_key, (max_jumps,), dtype=dtype)
    states_after, elapsed_event_times = _simulate_continuous_events(
        chain, state_0, exponentials, uniforms, method
    )
    event_times = t_0 + elapsed_event_times
    num_jumps = jnp.sum(event_times <= t_1, dtype=jnp.int32)
    all_states = prepend(state_0, states_after)
    final_state = all_states[num_jumps]
    covered = event_times[-1] >= t_1
    integration_ok = initial_ok & time_ok & covered

    if save_at.t_1:
        reached_time = jnp.where(integration_ok, t_1, event_times[-1])
        return Solution(
            ts=reached_time,
            xs=final_state,
            ok=integration_ok,
            num_accepted=num_jumps,
        )
    if save_at.steps:
        accepted = jnp.arange(max_jumps + 1) <= num_jumps
        raw_times = jnp.concatenate([t_0[None], event_times])
        output_times = jnp.where(accepted, raw_times, t_1)
        output_states = jnp.where(accepted, all_states, final_state)
        return Solution(
            ts=output_times,
            xs=output_states,
            ok=integration_ok,
            num_accepted=num_jumps,
            accepted=accepted,
        )
    query_times = jnp.asarray(save_at.ts, dtype)
    if query_times.ndim != 1:
        raise TypeError("continuous-time SaveAt.ts must be one-dimensional")
    queries_ok = jnp.all((query_times >= t_0) & (query_times <= t_1))
    event_counts = jnp.searchsorted(
        event_times, query_times, side="right", method="compare_all"
    )
    event_counts = jnp.minimum(event_counts, num_jumps)
    return Solution(
        ts=query_times,
        xs=all_states[event_counts],
        ok=integration_ok & queries_ok,
        num_accepted=num_jumps,
    )


def _prepare_distribution(chain, distribution_0):
    matrix = (
        chain.transition_matrix
        if isinstance(chain, DiscreteMarkovChain)
        else chain.generator
    )
    distribution_0 = jnp.asarray(distribution_0, matrix.dtype)
    if distribution_0.ndim != 1 or distribution_0.shape[0] != chain.num_states:
        raise ValueError("distribution_0 must have shape (chain.num_states,)")
    finite = jnp.all(jnp.isfinite(distribution_0))
    nonnegative = jnp.all(distribution_0 >= 0)
    safe = jnp.where(
        jnp.isfinite(distribution_0) & (distribution_0 >= 0), distribution_0, 0
    )
    total = jnp.sum(safe)
    tolerance = 100 * jnp.finfo(distribution_0.dtype).eps * chain.num_states
    valid_mass = jnp.abs(jnp.sum(distribution_0) - 1) <= tolerance
    valid = finite & nonnegative & (total > 0) & valid_mass
    fallback = safe / jnp.where(total > 0, total, 1)
    return jnp.where(valid, distribution_0, fallback), valid


def _flat_distribution_is_valid(probabilities):
    dtype = probabilities.dtype
    tolerance = 500 * jnp.finfo(dtype).eps * probabilities.shape[-1]
    return (
        jnp.all(jnp.isfinite(probabilities))
        & jnp.all(probabilities >= -tolerance)
        & jnp.all(jnp.abs(jnp.sum(probabilities, axis=-1) - 1) <= tolerance)
    )


def _pytree_distribution_is_valid(probabilities, *, batched):
    leaves = jax.tree.leaves(probabilities)
    dtype = leaves[0].dtype
    num_states = 0
    mass = None
    finite = jnp.asarray(True)
    minimum = jnp.asarray(jnp.inf, dtype)
    for leaf in leaves:
        finite = finite & jnp.all(jnp.isfinite(leaf))
        minimum = jnp.minimum(minimum, jnp.min(leaf))
        if batched:
            axes = tuple(range(1, leaf.ndim))
            leaf_mass = jnp.sum(leaf, axis=axes)
            num_states += math.prod(leaf.shape[1:])
        else:
            leaf_mass = jnp.sum(leaf)
            num_states += leaf.size
        mass = leaf_mass if mass is None else mass + leaf_mass
    tolerance = 500 * jnp.finfo(dtype).eps * num_states
    return finite & (minimum >= -tolerance) & jnp.all(jnp.abs(mass - 1) <= tolerance)


def _prepare_pytree_distribution(distribution_0):
    distribution_0, dtype = asarray_state(distribution_0, "distribution_0")
    flat, unravel = ravel_pytree(distribution_0)
    finite = jnp.all(jnp.isfinite(flat))
    nonnegative = jnp.all(flat >= 0)
    safe_flat = jnp.where(jnp.isfinite(flat) & (flat >= 0), flat, 0)
    total = jnp.sum(safe_flat)
    tolerance = 100 * jnp.finfo(dtype).eps * flat.size
    valid_mass = jnp.abs(jnp.sum(flat) - 1) <= tolerance
    valid = finite & nonnegative & (total > 0) & valid_mass
    fallback = safe_flat / jnp.where(total > 0, total, 1)
    prepared = unravel(jnp.where(valid, flat, fallback))
    return prepared, valid, dtype


def _discrete_distribution_steps(chain, distribution_0, num_steps, method):
    def step(distribution, _):
        next_distribution = distribution @ chain.transition_matrix
        return next_distribution, next_distribution

    if isinstance(method, SequentialMarkov):
        return jax.lax.scan(
            step,
            distribution_0,
            None,
            length=num_steps,
            unroll=method.unroll,
        )[1]
    if isinstance(method, AssociativeMarkov):
        operators = jnp.broadcast_to(
            chain.transition_matrix,
            (num_steps,) + chain.transition_matrix.shape,
        )
        prefixes = jax.lax.associative_scan(
            lambda earlier, later: earlier @ later, operators, axis=0
        )
        return jnp.einsum("i,tij->tj", distribution_0, prefixes)
    raise TypeError(
        "multi-step distributions require SequentialMarkov or AssociativeMarkov"
    )


def forecast_markov_chain(
    chain,
    distribution_0,
    *,
    num_steps,
    method=None,
    save_at=None,
):
    """Forecast a fixed DTMC probability mass function.

    Endpoint output defaults to binary matrix powering. Multi-row output defaults
    to chronological matrix-vector scan; ``AssociativeMarkov`` instead composes
    prefix transition matrices and can expose useful GPU parallelism for small
    state spaces. The deterministic forecast supports JVP/VJP with respect to
    ``distribution_0``; the prepared chain is treated as fixed.
    """
    if not isinstance(chain, DiscreteMarkovChain):
        raise TypeError("chain must be a DiscreteMarkovChain")
    if not isinstance(num_steps, int) or isinstance(num_steps, bool):
        raise TypeError("num_steps must be a static Python int")
    if num_steps < 0:
        raise ValueError("num_steps must be nonnegative")
    if save_at is None:
        save_at = SaveAt(t_1=True)
    distribution_0, initial_ok = _prepare_distribution(chain, distribution_0)

    if save_at.t_1:
        if method is None or isinstance(method, MatrixPowerMarkov):
            probabilities = distribution_0 @ jnp.linalg.matrix_power(
                chain.transition_matrix, num_steps
            )
        elif isinstance(method, SequentialMarkov):
            probabilities = jax.lax.fori_loop(
                0,
                num_steps,
                lambda _, distribution: distribution @ chain.transition_matrix,
                distribution_0,
            )
        elif isinstance(method, AssociativeMarkov):
            if num_steps == 0:
                probabilities = distribution_0
            else:
                probabilities = _discrete_distribution_steps(
                    chain, distribution_0, num_steps, method
                )[-1]
        else:
            raise TypeError(
                "method must be MatrixPowerMarkov, SequentialMarkov, or "
                "AssociativeMarkov"
            )
        return MarkovDistribution(
            ts=jnp.asarray(num_steps, jnp.int32),
            probabilities=probabilities,
            ok=initial_ok & _flat_distribution_is_valid(probabilities),
        )

    if isinstance(method, MatrixPowerMarkov):
        raise ValueError("MatrixPowerMarkov supports endpoint output only")
    if method is None:
        method = SequentialMarkov()
    if num_steps == 0:
        all_probabilities = distribution_0[None]
    else:
        step_probabilities = _discrete_distribution_steps(
            chain, distribution_0, num_steps, method
        )
        all_probabilities = prepend(distribution_0, step_probabilities)
    if save_at.steps:
        times = jnp.arange(num_steps + 1, dtype=jnp.int32)
        probabilities = all_probabilities
    else:
        query_steps = jnp.asarray(save_at.ts)
        if query_steps.ndim != 1 or not jnp.issubdtype(query_steps.dtype, jnp.integer):
            raise TypeError(
                "discrete SaveAt.ts must be a one-dimensional integer array"
            )
        queries_ok = jnp.all((query_steps >= 0) & (query_steps <= num_steps))
        safe_queries = jnp.clip(query_steps, 0, num_steps).astype(jnp.int32)
        times = query_steps
        probabilities = all_probabilities[safe_queries]
        initial_ok = initial_ok & queries_ok
    return MarkovDistribution(
        ts=times,
        probabilities=probabilities,
        ok=initial_ok & _flat_distribution_is_valid(probabilities),
    )


def forecast_continuous_time_markov_chain(
    chain,
    t_0,
    t_1,
    distribution_0,
    *,
    method=None,
    save_at=None,
):
    """Forecast a fixed CTMC probability mass with exponential actions.

    For one endpoint this evaluates ``distribution_0 @ exp((t_1-t_0) Q)``.
    ``DenseExponential`` forms the dense exponential.
    ``KrylovExponential`` applies a static Arnoldi approximation;
    ``AdaptiveKrylovExponential`` adapts its internal time slices. Both support
    ``MatrixFreeContinuousTimeMarkovChain`` probability pytrees.
    Requested times are independent exponential actions vectorized over the query
    axis. JVP/VJP with respect to ``distribution_0`` are supported.
    """
    is_dense = isinstance(chain, ContinuousTimeMarkovChain)
    is_matrix_free = isinstance(chain, MatrixFreeContinuousTimeMarkovChain)
    if not is_dense and not is_matrix_free:
        raise TypeError(
            "chain must be ContinuousTimeMarkovChain or "
            "MatrixFreeContinuousTimeMarkovChain"
        )
    if method is None:
        method = DenseExponential() if is_dense else KrylovExponential()
    if not isinstance(
        method, (DenseExponential, KrylovExponential, AdaptiveKrylovExponential)
    ):
        raise TypeError(
            "CTMC distribution forecasts require DenseExponential, "
            "KrylovExponential, or AdaptiveKrylovExponential"
        )
    if save_at is None:
        save_at = SaveAt(t_1=True)
    if save_at.steps:
        raise ValueError("CTMC distribution forecasts require endpoint or SaveAt.ts")
    if is_dense:
        distribution_0, initial_ok = _prepare_distribution(chain, distribution_0)
        dtype = chain.generator.dtype
    else:
        distribution_0, initial_ok, dtype = _prepare_pytree_distribution(distribution_0)
    if is_dense:
        operator = chain.generator.T
    else:
        operator = chain.forward_generator
    linear_solution = solve_linear_ode(
        operator,
        method,
        t_0,
        t_1,
        distribution_0,
        save_at=save_at,
    )
    probabilities = linear_solution.xs
    return MarkovDistribution(
        ts=linear_solution.ts,
        probabilities=probabilities,
        ok=(
            initial_ok
            & linear_solution.ok
            & _pytree_distribution_is_valid(probabilities, batched=not save_at.t_1)
        ),
    )
