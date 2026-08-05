import jax
import numpy as np
from jax.extend import core
from jax.interpreters import ad, batching, mlir

# A lax.cond whose predicate is batched is lowered under vmap to a select
# that executes BOTH branches for every lane, so the adaptive loops'
# skip-the-frozen-tail conds do no skipping in a vmapped solve: every attempt
# slot up to max_steps runs for every lane. unvmap_all reduces its boolean
# across any vmapped axes (its batching rule maps to an all-reduce and an
# unbatched result), so a cond gated on it keeps a scalar predicate under
# vmap and skips for real once every lane is finished. Its output is a
# nondifferentiable bool, hence the zero-JVP rule.

unvmap_all_p = core.Primitive("unvmap_all")


def unvmap_all(x):
    """All-reduce a boolean over any vmapped axes to a scalar predicate."""
    return unvmap_all_p.bind(x)


def unvmap_all_impl(x):
    return jax.numpy.all(x)


unvmap_all_p.def_impl(unvmap_all_impl)
unvmap_all_p.def_abstract_eval(lambda x: jax.core.ShapedArray((), np.bool_))


def unvmap_all_batch(args, dims):
    (x,) = args
    (dim,) = dims
    # None is JAX's not_mapped sentinel: the reduced output has no batch axis.
    return unvmap_all(jax.numpy.all(x, axis=dim)), None


batching.primitive_batchers[unvmap_all_p] = unvmap_all_batch
# The output is a nondifferentiable bool: no tangent contribution.
ad.defjvp(unvmap_all_p, None)
mlir.register_lowering(
    unvmap_all_p, mlir.lower_fun(unvmap_all_impl, multiple_results=False)
)
