"""Leaf-local arithmetic for differential-equation state pytrees."""

import jax
import jax.numpy as jnp


def asarray_state(state, name):
    """Convert leaves to arrays and enforce one real floating dtype."""
    try:
        state = jax.tree.map(jnp.asarray, state)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a pytree of array-like leaves") from error
    leaves = jax.tree.leaves(state)
    if not leaves:
        raise ValueError(f"{name} must contain at least one array leaf")
    dtype = leaves[0].dtype
    if not jnp.issubdtype(dtype, jnp.floating):
        raise TypeError(f"{name} leaves must have a real floating dtype")
    for leaf in leaves[1:]:
        if not jnp.issubdtype(leaf.dtype, jnp.floating):
            raise TypeError(f"{name} leaves must have a real floating dtype")
        if leaf.dtype != dtype:
            raise TypeError(f"all {name} leaves must have the same dtype")
    if any(leaf.size == 0 for leaf in leaves):
        raise ValueError(f"{name} leaves must not be empty")
    return state, dtype


def asarray_aux(aux):
    """Convert an aux pytree to nonempty, real floating array leaves."""
    try:
        aux = jax.tree.map(jnp.asarray, aux)
    except (TypeError, ValueError) as error:
        raise TypeError("aux must be a pytree of array-like leaves") from error
    leaves = jax.tree.leaves(aux)
    if not leaves:
        raise ValueError("aux must contain at least one array leaf")
    for leaf in leaves:
        if not jnp.issubdtype(leaf.dtype, jnp.floating):
            raise TypeError("aux leaves must have a real floating dtype")
        if leaf.size == 0:
            raise ValueError("aux leaves must not be empty")
    return aux


def assert_same_structure(reference, value, name):
    if jax.tree.structure(reference) != jax.tree.structure(value):
        raise ValueError(f"{name} must have the same pytree structure as the state")


def zeros_like(tree):
    return jax.tree.map(jnp.zeros_like, tree)


def where(condition, x, y):
    return jax.tree.map(lambda a, b: jnp.where(condition, a, b), x, y)


def add_scaled(x, *terms):
    """Return ``x + sum(scale * value for scale, value in terms)``."""
    return jax.tree.map(
        lambda base, *values: (
            base
            + sum(
                scale * value for (scale, _), value in zip(terms, values, strict=True)
            )
        ),
        x,
        *(value for _, value in terms),
    )


def weighted_sum(values, coefficients):
    return jax.tree.map(
        lambda *leaves: sum(
            coefficient * leaf
            for coefficient, leaf in zip(coefficients, leaves, strict=True)
        ),
        *values,
    )


def multiply(x, y):
    return jax.tree.map(lambda a, b: a * b, x, y)


def prepend(initial, rows):
    return jax.tree.map(lambda x, xs: jnp.concatenate([x[None], xs]), initial, rows)


def take(tree, indices):
    return jax.tree.map(lambda x: x[indices], tree)


def fill_rows(values, accepted, last, fill):
    def fill_leaf(x, final):
        mask = accepted.reshape(accepted.shape + (1,) * (x.ndim - 1))
        replacement = jnp.asarray(jnp.inf, x.dtype) if fill == "inf" else final
        return jnp.where(mask, x, replacement)

    return jax.tree.map(fill_leaf, values, last)


def error_ratio(x_0, x_1, err, rtol, atol):
    ratios = jax.tree.leaves(
        jax.tree.map(
            lambda a, b, e: jnp.max(
                jnp.abs(e) / (atol + rtol * jnp.maximum(jnp.abs(a), jnp.abs(b)))
            ),
            x_0,
            x_1,
            err,
        )
    )
    value = ratios[0]
    for ratio in ratios[1:]:
        value = jnp.maximum(value, ratio)
    return value


def full_like(tree, value):
    return jax.tree.map(lambda x: jnp.full_like(x, value), tree)
