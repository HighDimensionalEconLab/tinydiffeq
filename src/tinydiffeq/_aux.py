"""Static aux detection and failure-safe JAX evaluators."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

from tinydiffeq._tree import all_finite, asarray_aux, asarray_context


def _validate_mode(mode, name):
    if mode is not None and not isinstance(mode, bool):
        raise TypeError(f"{name} must be a static bool or None")


def _validate_saved_aux_shape(aux):
    leaves = jax.tree.leaves(aux)
    if not leaves:
        raise ValueError("aux must contain at least one array leaf")
    for leaf in leaves:
        if not jnp.issubdtype(leaf.dtype, jnp.floating):
            raise TypeError("aux leaves must have a real floating dtype")
        if leaf.size == 0:
            raise ValueError("aux leaves must not be empty")


def _validate_context_shape(context):
    leaves = jax.tree.leaves(context)
    if not leaves:
        raise ValueError("algebraic aux must contain at least one array leaf")
    for leaf in leaves:
        if leaf.size == 0:
            raise ValueError("algebraic aux leaves must not be empty")
        if not (
            jnp.issubdtype(leaf.dtype, jnp.bool_)
            or jnp.issubdtype(leaf.dtype, jnp.integer)
            or jnp.issubdtype(leaf.dtype, jnp.inexact)
        ):
            raise TypeError(
                "algebraic aux leaves must have boolean, integer, real, or "
                "complex dtypes"
            )


def shape_tree(tree):
    """Remove any nested-transform tracer identity from abstract outputs."""
    return jax.tree.map(
        lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype),
        tree,
    )


def resolve_field_aux(
    field: Callable,
    primals: tuple,
    state_structure,
    mode: bool | None,
    *,
    name: str,
):
    """Resolve whether a differential field returns ``(value, aux)``.

    A whole output matching the state pytree takes precedence, so a two-leaf
    tuple state remains unambiguous. Explicit ``False`` avoids the abstract
    detection trace and is checked naturally by the first runtime field call.
    """
    _validate_mode(mode, name)
    if mode is False:
        return False, None
    output = jax.eval_shape(field, *primals)
    output_structure = jax.tree.structure(output)
    if mode is None and output_structure == state_structure:
        return False, None
    if mode is None and (not isinstance(output, tuple) or len(output) != 2):
        return False, None
    if not isinstance(output, tuple) or len(output) != 2:
        expectation = f"{name}=True" if mode else "auto aux detection"
        raise TypeError(f"with {expectation}, the field must return (value, aux)")
    if jax.tree.structure(output[0]) != state_structure:
        raise ValueError("the differential value must match the state pytree")
    _validate_saved_aux_shape(output[1])
    return True, shape_tree(output[1])


def resolve_algebraic_aux(
    field: Callable,
    primals: tuple,
    mode: bool | None,
):
    """Resolve whether ``g`` returns ``(residual, algebraic_aux)``."""
    _validate_mode(mode, "has_algebraic_aux")
    if mode is False:
        return False, None
    output = jax.eval_shape(field, *primals)
    detected = isinstance(output, tuple) and len(output) == 2
    if mode is None and not detected:
        return False, None
    if not detected:
        raise TypeError(
            "with has_algebraic_aux=True, g must return (residual, algebraic_aux)"
        )
    _validate_context_shape(output[1])
    return True, shape_tree(output[1])


def split_field_output(output, has_aux):
    if has_aux:
        value, aux = output
        return value, asarray_aux(aux)
    return output, None


def split_algebraic_output(output, has_algebraic_aux):
    if has_algebraic_aux:
        residual, context = output
        return residual, asarray_context(context)
    return output, None


def zeros_from_shape(shape_tree):
    return jax.tree.map(lambda leaf: jnp.zeros(leaf.shape, leaf.dtype), shape_tree)


def _tree_where(condition, x, y):
    return jax.tree.map(
        lambda a, b: None if a is None else jnp.where(condition, a, b),
        x,
        y,
        is_leaf=lambda value: value is None,
    )


def _mask_tangent(condition, tangent, zero):
    def mask(value, zero_value):
        if getattr(value, "dtype", None) == jax.dtypes.float0:
            return value
        return jnp.where(condition, value, zero_value)

    return jax.tree.map(mask, tangent, zero)


def make_safe_evaluator(function, output_shape, *, check_finite=True):
    """Return ``evaluate(inputs, active, reference) -> (value, ok)``.

    ``reference`` is substituted before differentiation on inactive or failed
    lanes, preventing selected-away invalid domains from poisoning batched
    JVPs and transposed VJPs.
    """

    def zeros():
        return zeros_from_shape(output_shape)

    @jax.custom_jvp
    def evaluate(inputs, active, reference):
        safe_inputs = _tree_where(active, inputs, reference)
        raw = jax.lax.cond(active, lambda: function(safe_inputs), zeros)
        finite = all_finite(raw) if check_finite else jnp.asarray(True)
        ok = active & finite
        return _tree_where(ok, raw, zeros()), ok

    @evaluate.defjvp
    def evaluate_jvp(primals, tangents):
        inputs, active, reference = primals
        inputs_dot, _, _ = tangents
        value, ok = evaluate(inputs, active, reference)
        safe_inputs = _tree_where(ok, inputs, reference)
        value_dot = jax.jvp(function, (safe_inputs,), (inputs_dot,))[1]
        zero_dot = jax.tree.map(
            lambda leaf: (
                jnp.zeros(leaf.shape, jax.dtypes.float0)
                if not jnp.issubdtype(leaf.dtype, jnp.inexact)
                else jnp.zeros_like(leaf)
            ),
            output_shape,
        )
        value_dot = _mask_tangent(ok, value_dot, zero_dot)
        ok_dot = jnp.zeros(ok.shape, dtype=jax.dtypes.float0)
        return (value, ok), (value_dot, ok_dot)

    return evaluate


def prepare_aux_reference(reference, state, time, p):
    """Validate and stop-gradient an ODE/SDE aux-domain reference."""

    def ones(tree):
        return jax.tree.map(
            lambda value: jax.lax.stop_gradient(jnp.ones_like(value)),
            tree,
        )

    if reference is None:
        return ones(state), jnp.ones_like(time), ones(p)
    if not isinstance(reference, tuple) or len(reference) != 3:
        raise TypeError("failure_ad_reference must be an (x, t, p) tuple")

    def cast_like(candidate, model, name):
        if jax.tree.structure(candidate) != jax.tree.structure(model):
            raise ValueError(f"failure_ad_reference {name} has the wrong pytree")

        def cast(candidate_leaf, model_leaf):
            model_leaf = jnp.asarray(model_leaf)
            if not isinstance(candidate_leaf, jax.core.Tracer):
                concrete = np.asarray(candidate_leaf, dtype=np.dtype(model_leaf.dtype))
                if np.issubdtype(concrete.dtype, np.inexact) and not np.all(
                    np.isfinite(concrete)
                ):
                    raise ValueError(
                        f"failure_ad_reference {name} leaves must be finite"
                    )
            value = jnp.asarray(candidate_leaf, model_leaf.dtype)
            if value.shape != model_leaf.shape:
                raise ValueError(
                    f"failure_ad_reference {name} leaves must match model shapes"
                )
            return jax.lax.stop_gradient(value)

        return jax.tree.map(cast, candidate, model)

    state_ref, time_ref, p_ref = reference
    state_ref = cast_like(state_ref, state, "x")
    p_ref = cast_like(p_ref, p, "p")
    if not isinstance(time_ref, jax.core.Tracer):
        concrete = np.asarray(time_ref, dtype=np.dtype(time.dtype))
        if not np.all(np.isfinite(concrete)):
            raise ValueError("failure_ad_reference t must be finite")
    time_ref = jax.lax.stop_gradient(jnp.asarray(time_ref, time.dtype))
    if time_ref.shape != time.shape:
        raise ValueError("failure_ad_reference t must match the time shape")
    return state_ref, time_ref, p_ref
