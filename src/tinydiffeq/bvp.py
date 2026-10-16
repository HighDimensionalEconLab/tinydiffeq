"""Collocation boundary value solver ported from scipy.integrate._bvp."""

import inspect
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from tinydiffeq._aux import resolve_field_aux, split_field_output
from tinydiffeq._bvp_core import (
    LOBATTO_OFFSET,
    LOBATTO_WEIGHT_MIDDLE,
    LOBATTO_WEIGHT_SIDE,
    MAX_ITERATION,
    MAX_NEWTON_ITERATIONS,
    MAX_NJEV,
    N_TRIAL,
    REFINE_FACTOR,
    RUNNING,
    SIGMA,
    STATUS_BC_TOL,
    STATUS_CONVERGED,
    STATUS_MAX_NODES,
    STATUS_SINGULAR,
    TAU,
    TOL_FLOOR_FACTOR,
    TOL_R_FACTOR,
    collocation_jacobian_blocks,
    hermite_pair,
    pad_tail,
    refined_mesh,
)
from tinydiffeq._tree import asarray_state, assert_same_structure, zero_tangent
from tinydiffeq._unvmap import unvmap_all
from tinydiffeq.babd import (
    babd_dense,
    babd_matvec,
    structured_qr_factor,
    structured_qr_solve,
    structured_qr_transpose_solve,
)
from tinydiffeq.interpolation import hermite_interpolate
from tinydiffeq.solution import BVPSolution


def bvp_arity(f, name, forms):
    # Count positional parameters; *args or uninspectable means the full form.
    try:
        signature = inspect.signature(f)
    except (TypeError, ValueError):
        return 5
    arity = 0
    for parameter in signature.parameters.values():
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            arity += 1
        elif parameter.kind == inspect.Parameter.VAR_POSITIONAL:
            return 5
    if arity < 2 or arity > 5:
        raise ValueError(f"{name} must take 2 to 5 positional arguments: {forms}")
    return arity


def canonicalize_bvp_fun(f, arity):
    if arity == 2:
        return lambda t, y, z, args, p: f(t, y)
    if arity == 3:
        return lambda t, y, z, args, p: f(t, y, z)
    if arity == 4:
        return lambda t, y, z, args, p: f(t, y, z, args)
    return f


def canonicalize_bvp_bc(f, arity):
    if arity == 2:
        return lambda ya, yb, z, args, p: f(ya, yb)
    if arity == 3:
        return lambda ya, yb, z, args, p: f(ya, yb, z)
    if arity == 4:
        return lambda ya, yb, z, args, p: f(ya, yb, z, args)
    return f


@dataclass(frozen=True)
class _BVPConfig:
    fun: Any
    bc: Any
    fun_arity: int
    bc_arity: int
    max_nodes: int
    n: int
    k: int
    has_aux: bool
    fun_jac_ad: str
    bc_jac_ad: str
    has_singular_term: bool
    y_treedef: Any
    y_leaf_specs: tuple
    z_treedef: Any
    z_leaf_specs: Any
    # Derived per call and fully determined by the compared fields; excluding
    # them keeps equal configurations sharing one jit compilation.
    fun_canon: Any = field(compare=False, repr=False)
    bc_canon: Any = field(compare=False, repr=False)
    unravel_y: Any = field(compare=False, repr=False)
    unravel_z: Any = field(compare=False, repr=False)


def resolve_jacobian_mode(mode, num_inputs, num_outputs):
    # "auto" seeds the small side: forward when tall or square, reverse when
    # strictly fat (the nlls-gram convention).
    if mode != "auto":
        return mode
    return "jvp" if num_inputs <= num_outputs else "vjp"


def jac_transform(mode):
    return jax.jacfwd if mode == "jvp" else jax.jacrev


def build_system(cfg, args, S, t_a):
    """Closures evaluating the wrapped field, bc, and Jacobians on flat rows.

    Shared by the primal loops and the implicit-differentiation rule so the
    Newton Jacobian and the AD Jacobian are the same assembly.
    """
    n, k = cfg.n, cfg.k
    if cfg.has_singular_term:
        S = jax.lax.stop_gradient(S)
        eye = jnp.eye(n, dtype=S.dtype)
        # rtol pins numpy's pinv cutoff so rank decisions agree with scipy.
        B = eye - jnp.linalg.pinv(S, rtol=1e-15) @ S
        D = jnp.linalg.pinv(eye - S, rtol=1e-15)
    else:
        B = D = None

    def raw_node_field(t_j, y_flat, z_flat, p):
        y = cfg.unravel_y(y_flat)
        out = cfg.fun_canon(t_j, y, cfg.unravel_z(z_flat), args, p)
        value, _ = split_field_output(out, cfg.has_aux)
        value, value_dtype = asarray_state(value, "fun(t, y, ...)")
        assert_same_structure(y, value, "fun(t, y, ...)")
        if value_dtype != y_flat.dtype:
            raise TypeError("fun(t, y, ...) must preserve the state dtype")
        return ravel_pytree(value)[0]

    def node_field(t_j, y_flat, z_flat, p, is_left):
        value = raw_node_field(t_j, y_flat, z_flat, p)
        if not cfg.has_singular_term:
            return value
        # double-where: the left-endpoint denominator never reaches the ratio.
        denominator = jnp.where(is_left, jnp.ones_like(t_j), t_j - t_a)
        interior = value + (S @ y_flat) / denominator
        return jnp.where(is_left, D @ value, interior)

    def left_mask(size, at_left):
        if at_left:
            return jnp.arange(size) == 0
        return jnp.zeros(size, bool)

    def field_values(t_vec, y_rows, z_flat, p, at_left):
        is_left = left_mask(t_vec.shape[0], at_left)
        return jax.vmap(node_field, in_axes=(0, 0, None, None, 0))(
            t_vec, y_rows, z_flat, p, is_left
        )

    def collocation_parts(t, h, Y, Z, p):
        f = field_values(t, Y, Z, p, True)
        y_middle = 0.5 * (Y[1:] + Y[:-1]) - 0.125 * h[:, None] * (f[1:] - f[:-1])
        f_middle = field_values(t[:-1] + 0.5 * h, y_middle, Z, p, False)
        col_res = Y[1:] - Y[:-1] - h[:, None] / 6.0 * (f[:-1] + f[1:] + 4.0 * f_middle)
        return col_res, y_middle, f, f_middle

    def bc_values(ya_flat, yb_flat, z_flat, p):
        residual = cfg.bc_canon(
            cfg.unravel_y(ya_flat),
            cfg.unravel_y(yb_flat),
            cfg.unravel_z(z_flat),
            args,
            p,
        )
        residual = jnp.asarray(residual)
        if residual.shape != (n + k,):
            raise ValueError(f"bc must return {n + k} residuals, got {residual.shape}")
        if residual.dtype != ya_flat.dtype:
            raise TypeError("bc(ya, yb, ...) must preserve the state dtype")
        return residual

    mode_fy = resolve_jacobian_mode(cfg.fun_jac_ad, n, n)
    mode_fz = resolve_jacobian_mode(cfg.fun_jac_ad, k, n)
    # bc is differentiated jointly with respect to (ya, yb, z).
    mode_bc = resolve_jacobian_mode(cfg.bc_jac_ad, 2 * n + k, n + k)

    def field_jacobians(t_vec, y_rows, z_flat, p, at_left):
        is_left = left_mask(t_vec.shape[0], at_left)
        if k == 0:

            def single(t_j, y_flat, il):
                def fn(yy):
                    return node_field(t_j, yy, z_flat, p, il)

                return jac_transform(mode_fy)(fn)(y_flat)

            return jax.vmap(single)(t_vec, y_rows, is_left), None
        if mode_fy == mode_fz:

            def single(t_j, y_flat, il):
                def fn(yy, zz):
                    return node_field(t_j, yy, zz, p, il)

                return jac_transform(mode_fy)(fn, argnums=(0, 1))(y_flat, z_flat)

            return jax.vmap(single)(t_vec, y_rows, is_left)

        def single_split(t_j, y_flat, il):
            def fn(yy, zz):
                return node_field(t_j, yy, zz, p, il)

            df_dy = jac_transform(mode_fy)(fn, argnums=0)(y_flat, z_flat)
            df_dz = jac_transform(mode_fz)(fn, argnums=1)(y_flat, z_flat)
            return df_dy, df_dz

        return jax.vmap(single_split)(t_vec, y_rows, is_left)

    def bc_jacobians(ya_flat, yb_flat, z_flat, p):
        if k == 0:

            def fn(a, b):
                return bc_values(a, b, z_flat, p)

            dbc_dya, dbc_dyb = jac_transform(mode_bc)(fn, argnums=(0, 1))(
                ya_flat, yb_flat
            )
            return dbc_dya, dbc_dyb, None

        def fn(a, b, zz):
            return bc_values(a, b, zz, p)

        return jac_transform(mode_bc)(fn, argnums=(0, 1, 2))(ya_flat, yb_flat, z_flat)

    def jacobian_blocks(t, h, Y, Z, y_middle, p):
        df_dy, df_dz = field_jacobians(t, Y, Z, p, True)
        df_dy_middle, df_dz_middle = field_jacobians(
            t[:-1] + 0.5 * h, y_middle, Z, p, False
        )
        dbc_dya, dbc_dyb, dbc_dz = bc_jacobians(Y[0], Y[-1], Z, p)
        return collocation_jacobian_blocks(
            h,
            df_dy,
            df_dy_middle,
            df_dz,
            df_dz_middle,
            dbc_dya,
            dbc_dyb,
            dbc_dz,
        )

    def jacobian_at(t, h, Y, Z, y_middle, p):
        return babd_dense(jacobian_blocks(t, h, Y, Z, y_middle, p))

    def project(Y):
        if not cfg.has_singular_term:
            return Y
        return Y.at[0].set(B @ Y[0])

    def aux_values(t, Y, Z, p):
        def node_aux(t_j, y_flat):
            out = cfg.fun_canon(t_j, cfg.unravel_y(y_flat), cfg.unravel_z(Z), args, p)
            return split_field_output(out, True)[1]

        return jax.vmap(node_aux)(t, Y)

    return dict(
        field_values=field_values,
        collocation_parts=collocation_parts,
        bc_values=bc_values,
        jacobian_blocks=jacobian_blocks,
        jacobian_at=jacobian_at,
        project=project,
        aux_values=aux_values,
    )


def estimate_rms_residuals(system, t, h, h_safe, active, Y, Z, f, f_middle, col_res, p):
    # 5-point Lobatto quadrature of the relative residual over each interval;
    # endpoint terms vanish because the spline collocates at the nodes.
    r_middle = 1.5 * col_res / h_safe[:, None]
    tau_right = 0.5 + LOBATTO_OFFSET
    tau_left = 0.5 - LOBATTO_OFFSET
    y1, y1_prime = hermite_pair(tau_right, h, h_safe, Y[:-1], Y[1:], f[:-1], f[1:])
    y2, y2_prime = hermite_pair(tau_left, h, h_safe, Y[:-1], Y[1:], f[:-1], f[1:])
    f1 = system["field_values"](t[:-1] + tau_right * h, y1, Z, p, False)
    f2 = system["field_values"](t[:-1] + tau_left * h, y2, Z, p, False)
    r1 = (y1_prime - f1) / (1.0 + jnp.abs(f1))
    r2 = (y2_prime - f2) / (1.0 + jnp.abs(f2))
    r_middle = r_middle / (1.0 + jnp.abs(f_middle))
    rms = jnp.sqrt(
        0.5
        * (
            LOBATTO_WEIGHT_MIDDLE * jnp.sum(r_middle * r_middle, axis=1)
            + LOBATTO_WEIGHT_SIDE
            * (jnp.sum(r1 * r1, axis=1) + jnp.sum(r2 * r2, axis=1))
        )
    )
    # The padded cubic has derivative f_b / 7 at the Lobatto points, not zero,
    # so inactive intervals must be masked rather than trusted to vanish.
    return jnp.where(active, rms, 0.0)


def solve_newton(cfg, system, t, h, Y, Z, num_nodes, p, tol, bc_tol, live):
    max_nodes, n, k = cfg.max_nodes, cfg.n, cfg.k
    size = max_nodes * n + k
    dtype = Y.dtype
    active = jnp.arange(max_nodes - 1) < num_nodes - 1
    tol_r = TOL_R_FACTOR * h * tol
    # The copy chain duplicates the last active node's step over the tail, so
    # the affine-invariant cost must count active variables only.
    active_variable = jnp.concatenate(
        [jnp.repeat(jnp.arange(max_nodes) < num_nodes, n), jnp.ones(k, bool)]
    )

    def masked_cost(step):
        return jnp.sum(jnp.where(active_variable, step * step, 0.0))

    def stack_residual(col_res, bc_res):
        return jnp.concatenate([col_res.reshape(-1), bc_res])

    col_res, y_middle, _, f_middle = system["collocation_parts"](t, h, Y, Z, p)
    bc_res = system["bc_values"](Y[0], Y[-1], Z, p)
    # The carried solver state is whatever pytree the linear solver's init
    # returns; its structure is recovered abstractly for the placeholder.
    state_shapes = jax.eval_shape(
        lambda tt, hh, YY, ZZ, ym, pp: structured_qr_factor(
            system["jacobian_blocks"](tt, hh, YY, ZZ, ym, pp)
        )[0],
        t,
        h,
        Y,
        Z,
        y_middle,
        p,
    )
    carry = dict(
        Y=Y,
        Z=Z,
        y_middle=y_middle,
        col_res=col_res,
        f_middle=f_middle,
        bc_res=bc_res,
        res=stack_residual(col_res, bc_res),
        state=jax.tree.map(lambda s: jnp.zeros(s.shape, s.dtype), state_shapes),
        step=jnp.zeros(size, dtype),
        cost=jnp.zeros((), dtype),
        recompute=jnp.asarray(True),
        njev=jnp.zeros((), jnp.int32),
        iteration=jnp.zeros((), jnp.int32),
        singular=jnp.asarray(False),
        # A lane whose outer iteration already terminated starts stopped, so
        # a batched Newton loop only runs until the live lanes finish.
        stop=~live,
    )

    def newton_cond(c):
        return (~c["stop"]) & (c["iteration"] < MAX_NEWTON_ITERATIONS)

    def newton_body(c):
        def refactor(c):
            blocks = system["jacobian_blocks"](t, h, c["Y"], c["Z"], c["y_middle"], p)
            state, ok = structured_qr_factor(blocks)
            step = structured_qr_solve(state, c["res"])
            return state, ok, step, masked_cost(step), c["njev"] + 1

        def reuse(c):
            return c["state"], jnp.asarray(True), c["step"], c["cost"], c["njev"]

        def per_lane(c):
            return jax.lax.cond(c["recompute"], refactor, reuse, c)

        # A batched cond runs both branches as a select; the unvmap_all gate
        # keeps a scalar predicate so an all-reusing batch skips the
        # factorization for real (see _unvmap). Stopped lanes must not veto:
        # a lane that finished on a damped step carries recompute=True
        # forever, and its frozen carry would otherwise pin the gate false.
        solver_state, factor_ok, step, cost, njev = jax.lax.cond(
            unvmap_all(~c["recompute"] | c["stop"]), reuse, per_lane, c
        )
        singular = ~factor_ok

        def trial_cond(tc):
            return (~tc["accepted"]) & (tc["trial"] <= N_TRIAL)

        def trial_body(tc):
            # alpha shrinks by exact multiplication, matching scipy's
            # ``alpha *= tau`` (powers of two, bitwise).
            alpha = tc["alpha"]
            y_candidate = c["Y"] - alpha * step[: max_nodes * n].reshape(max_nodes, n)
            y_candidate = system["project"](y_candidate)
            y_candidate = pad_tail(y_candidate, num_nodes)
            z_candidate = c["Z"] - alpha * step[max_nodes * n :]
            col_res, y_middle, _, f_middle = system["collocation_parts"](
                t, h, y_candidate, z_candidate, p
            )
            bc_res = system["bc_values"](
                y_candidate[0], y_candidate[-1], z_candidate, p
            )
            res = stack_residual(col_res, bc_res)
            step_new = structured_qr_solve(solver_state, res)
            cost_new = masked_cost(step_new)
            return dict(
                trial=tc["trial"] + 1,
                trial_used=tc["trial"],
                alpha=alpha * TAU,
                accepted=cost_new < (1.0 - 2.0 * alpha * SIGMA) * cost,
                Y=y_candidate,
                Z=z_candidate,
                y_middle=y_middle,
                col_res=col_res,
                f_middle=f_middle,
                bc_res=bc_res,
                res=res,
                step_new=step_new,
                cost_new=cost_new,
            )

        trial = jax.lax.while_loop(
            trial_cond,
            trial_body,
            dict(
                trial=jnp.zeros((), jnp.int32),
                trial_used=jnp.zeros((), jnp.int32),
                alpha=jnp.ones((), dtype),
                # A singular factorization takes no trials, as scipy's break.
                accepted=singular,
                Y=c["Y"],
                Z=c["Z"],
                y_middle=c["y_middle"],
                col_res=c["col_res"],
                f_middle=c["f_middle"],
                bc_res=c["bc_res"],
                res=c["res"],
                step_new=step,
                cost_new=cost,
            ),
        )

        # A singular factorization stops before taking any step, as scipy does.
        def keep_current(tr):
            return (
                c["Y"],
                c["Z"],
                c["y_middle"],
                c["col_res"],
                c["f_middle"],
                c["bc_res"],
                c["res"],
            )

        def take_candidate(tr):
            return (
                tr["Y"],
                tr["Z"],
                tr["y_middle"],
                tr["col_res"],
                tr["f_middle"],
                tr["bc_res"],
                tr["res"],
            )

        Y_next, Z_next, y_middle, col_res, f_middle, bc_res, res = jax.lax.cond(
            singular, keep_current, take_candidate, trial
        )

        # Inactive intervals have tol_r == 0 == col_res, so mask them true.
        col_ok = jnp.all(
            jnp.where(
                active[:, None],
                jnp.abs(col_res) < tol_r[:, None] * (1.0 + jnp.abs(f_middle)),
                True,
            )
        )
        bc_ok = jnp.all(jnp.abs(bc_res) < bc_tol)
        stop = singular | (njev == MAX_NJEV) | (col_ok & bc_ok)
        # A full step keeps the frozen Jacobian; a damped one forces refresh.
        full_step = trial["accepted"] & (trial["trial_used"] == 0)
        return dict(
            Y=Y_next,
            Z=Z_next,
            y_middle=y_middle,
            col_res=col_res,
            f_middle=f_middle,
            bc_res=bc_res,
            res=res,
            state=solver_state,
            step=jnp.where(full_step, trial["step_new"], step),
            cost=jnp.where(full_step, trial["cost_new"], cost),
            recompute=~full_step,
            njev=njev,
            iteration=c["iteration"] + 1,
            singular=singular,
            stop=stop,
        )

    final = jax.lax.while_loop(newton_cond, newton_body, carry)
    return final["Y"], final["Z"], final["singular"]


def _solve_bvp_impl(cfg, t, Y, Z, num_nodes, p, args, S, tol, bc_tol):
    max_nodes, n = cfg.max_nodes, cfg.n
    dtype = Y.dtype
    tol = jnp.maximum(jnp.asarray(tol, dtype), TOL_FLOOR_FACTOR * jnp.finfo(dtype).eps)
    bc_tol = jnp.asarray(bc_tol, dtype)
    bc_tol = jnp.where(jnp.isnan(bc_tol), tol, bc_tol)
    system = build_system(cfg, args, S, t[0])
    Y = system["project"](Y)

    def outer_cond(c):
        return (c["status"] == RUNNING) & (c["iteration"] < max_nodes + MAX_ITERATION)

    def outer_body(c):
        t_c, num_c = c["t"], c["num_nodes"]
        h = t_c[1:] - t_c[:-1]
        active = jnp.arange(max_nodes - 1) < num_c - 1
        h_safe = jnp.where(active, h, jnp.ones_like(h))
        Y_c, Z_c, singular = solve_newton(
            cfg,
            system,
            t_c,
            h,
            c["Y"],
            c["Z"],
            num_c,
            p,
            tol,
            bc_tol,
            c["status"] == RUNNING,
        )
        iteration = c["iteration"] + 1
        col_res, _, f, f_middle = system["collocation_parts"](t_c, h, Y_c, Z_c, p)
        bc_res = system["bc_values"](Y_c[0], Y_c[-1], Z_c, p)
        max_bc_res = jnp.max(jnp.abs(bc_res))
        rms = estimate_rms_residuals(
            system, t_c, h, h_safe, active, Y_c, Z_c, f, f_middle, col_res, p
        )
        insert_1 = active & (rms > tol) & (rms < REFINE_FACTOR * tol)
        insert_2 = active & (rms >= REFINE_FACTOR * tol)
        nodes_added = (jnp.sum(insert_1) + 2 * jnp.sum(insert_2)).astype(jnp.int32)
        # The max_nodes test precedes any modification: a status-1 result
        # reports the unmodified mesh, exactly as scipy does.
        overflow = num_c + nodes_added > max_nodes
        live = ~singular & ~overflow
        refine = live & (nodes_added > 0)
        converged = live & (nodes_added == 0) & (max_bc_res <= bc_tol)
        # ~(<=) rather than (>): a NaN boundary residual must follow scipy's
        # elif chain to status 3 instead of looping to the safety bound.
        stalled = (
            live
            & (nodes_added == 0)
            & ~(max_bc_res <= bc_tol)
            & (iteration >= MAX_ITERATION)
        )
        status = jnp.where(
            singular,
            STATUS_SINGULAR,
            jnp.where(
                overflow,
                STATUS_MAX_NODES,
                jnp.where(
                    converged,
                    STATUS_CONVERGED,
                    jnp.where(stalled, STATUS_BC_TOL, RUNNING),
                ),
            ),
        ).astype(jnp.int32)
        # Refinement is computed unconditionally and selected: a batched cond
        # would run both branches under vmap anyway, and the sort is cheap.
        t_refined, num_refined = refined_mesh(t_c, num_c, insert_1, insert_2)
        y_refined = pad_tail(hermite_interpolate(t_refined, t_c, Y_c, f), num_refined)
        return dict(
            t=jnp.where(refine, t_refined, t_c),
            Y=jnp.where(refine, y_refined, Y_c),
            Z=Z_c,
            num_nodes=jnp.where(refine, num_refined, num_c),
            iteration=iteration,
            status=status,
            f=f,
            rms=rms,
        )

    final = jax.lax.while_loop(
        outer_cond,
        outer_body,
        dict(
            t=t,
            Y=Y,
            Z=Z,
            num_nodes=num_nodes,
            iteration=jnp.zeros((), jnp.int32),
            status=jnp.asarray(RUNNING, jnp.int32),
            f=jnp.zeros((max_nodes, n), dtype),
            rms=jnp.zeros(max_nodes - 1, dtype),
        ),
    )
    aux = (
        system["aux_values"](final["t"], final["Y"], final["Z"], p)
        if cfg.has_aux
        else None
    )
    return (
        final["t"],
        final["Y"],
        final["Z"],
        final["f"],
        final["rms"],
        final["num_nodes"],
        final["iteration"],
        final["status"],
        aux,
    )


_solve_bvp_jit = jax.jit(_solve_bvp_impl, static_argnums=(0,))


def mask_tangent(condition, tangent, zero):
    def mask(value, zero_value):
        if getattr(value, "dtype", None) == jax.dtypes.float0:
            return value
        return jnp.where(condition, value, zero_value)

    return jax.tree.map(mask, tangent, zero)


def _run_with_ad(cfg, t, Y, Z, num_nodes, p, args, S, tol, bc_tol):
    def primal(t, Y, Z, num_nodes, p, args, S, tol, bc_tol):
        return _solve_bvp_jit(cfg, t, Y, Z, num_nodes, p, args, S, tol, bc_tol)

    run = jax.custom_jvp(primal)

    @run.defjvp
    def run_jvp(primals, tangents):
        result = run(*primals)
        zeros = zero_tangent(result)
        t_in, Y_in, Z_in, _, p_in, args_in, S_in, _, _ = primals
        p_dot = tangents[4]
        if p_in is None:
            return result, zeros
        t_f, Y_f, Z_f, _, _, num_f, _, status, _ = result

        # Implicit function theorem at the solution on the frozen final mesh;
        # a failed lane's tangent program runs at the inert initial guess so
        # batched JVPs and transposed VJPs stay finite, then masks to zero.
        # The solution, p, and the Jacobian stay differentiable: an outer
        # transform of this rule recurses through the same custom_jvp, so
        # higher-order derivatives (hessians, reverse-over-forward) are exact
        # on the frozen mesh. Only the mesh and the failed-lane substitutes
        # are stop-gradiented.
        ad_ok = status == STATUS_CONVERGED
        t_c = jax.lax.stop_gradient(t_f)
        num_c = jax.lax.stop_gradient(num_f)
        Y_sol = jnp.where(ad_ok, Y_f, jax.lax.stop_gradient(Y_in))
        Z_sol = jnp.where(ad_ok, Z_f, jax.lax.stop_gradient(Z_in))
        p_c = p_in
        args_c = jax.lax.stop_gradient(args_in)
        S_c = jax.lax.stop_gradient(S_in) if cfg.has_singular_term else None
        system = build_system(cfg, args_c, S_c, t_c[0])
        h_c = t_c[1:] - t_c[:-1]
        _, y_middle, _, _ = system["collocation_parts"](t_c, h_c, Y_sol, Z_sol, p_c)
        blocks = system["jacobian_blocks"](t_c, h_c, Y_sol, Z_sol, y_middle, p_c)
        # The factorization is solver state, not a differentiation path:
        # tangents of the matrix flow through custom_linear_solve's matvec.
        state, factor_ok = structured_qr_factor(jax.lax.stop_gradient(blocks))
        p_dot_masked = mask_tangent(ad_ok, p_dot, zero_tangent(p_in))

        def residual_of_p(p_value):
            col_res, _, _, _ = system["collocation_parts"](
                t_c, h_c, Y_sol, Z_sol, p_value
            )
            bc_res = system["bc_values"](Y_sol[0], Y_sol[-1], Z_sol, p_value)
            return jnp.concatenate([col_res.reshape(-1), bc_res])

        rhs_dot = jax.jvp(residual_of_p, (p_c,), (p_dot_masked,))[1]
        delta = jax.lax.custom_linear_solve(
            lambda u: babd_matvec(blocks, u),
            -rhs_dot,
            lambda _, b: structured_qr_solve(state, b),
            transpose_solve=lambda _, b: structured_qr_transpose_solve(state, b),
        )
        y_dot = delta[: cfg.max_nodes * cfg.n].reshape(cfg.max_nodes, cfg.n)
        y_dot = system["project"](y_dot)
        y_dot = pad_tail(y_dot, num_c)
        z_dot = delta[cfg.max_nodes * cfg.n :]

        # A failed solve has exact-zero tangents: the where-select (not a
        # multiplication) kills any non-finite garbage from the inert-guess
        # tangent program. A converged solve whose final-mesh Jacobian fails
        # to factor has no computable derivative: scale by NaN rather than
        # substituting a constant NaN, so the failure survives transposition
        # into reverse mode instead of dropping to a silent zero cotangent.
        def finalize(dot):
            dot = jnp.where(ad_ok, dot, jnp.zeros_like(dot))
            scale = jnp.where(ad_ok & ~factor_ok, jnp.nan, 1.0)
            return dot * jnp.asarray(scale, dot.dtype)

        y_dot = finalize(y_dot)
        z_dot = finalize(z_dot)
        f_dot = jax.jvp(
            lambda yv, zv, pv: system["field_values"](t_c, yv, zv, pv, True),
            (Y_sol, Z_sol, p_c),
            (y_dot, z_dot, p_dot_masked),
        )[1]
        f_dot = finalize(f_dot)
        if cfg.has_aux:
            aux_dot = jax.jvp(
                lambda yv, zv, pv: system["aux_values"](t_c, yv, zv, pv),
                (Y_sol, Z_sol, p_c),
                (y_dot, z_dot, p_dot_masked),
            )[1]
            aux_dot = jax.tree.map(finalize, aux_dot)
        else:
            aux_dot = None
        tangent = (
            zeros[0],
            y_dot,
            z_dot,
            f_dot,
            zeros[4],
            zeros[5],
            zeros[6],
            zeros[7],
            aux_dot,
        )
        return result, tangent

    return run(t, Y, Z, num_nodes, p, args, S, tol, bc_tol)


def _unravel_empty(z_flat):
    return None


def solve_bvp(
    fun,
    bc,
    t,
    y_0,
    z_0=None,
    *,
    p=None,
    args=None,
    S=None,
    fun_jac_ad="auto",
    bc_jac_ad="auto",
    tol=1e-3,
    bc_tol=None,
    max_nodes=128,
    has_aux=None,
):
    """Solve ``dy/dt = fun(t, y, z, args, p) + S y / (t - t[0])`` with
    two-point boundary conditions ``bc(y(t_a), y(t_b), z, args, p) = 0``.

    A faithful JAX port of :func:`scipy.integrate.solve_bvp` (4th-order Lobatto
    IIIA collocation with residual-controlled mesh refinement and a damped
    Newton method), with scipy's algorithm, constants, and default tolerances.
    ``fun`` and ``bc`` are pointwise — a scalar ``t`` and one node's state
    pytree — and may be declared with 2 to 5 positional arguments in the
    orders above. ``z_0`` is the guess for scipy's unknown parameters (any
    pytree), solved jointly with ``y`` and returned as ``sol.z``; ``bc`` must
    then return ``n + size(z)`` residuals as a 1-D array. ``p`` holds known
    differentiable parameters — the only AD input: JVP/VJP rules (composing
    to higher order) differentiate ``sol.y``, ``sol.yp``, ``sol.z``, and
    ``sol.aux`` with respect to ``p`` implicitly at the solution, never
    through the iterations, and the guesses ``t``, ``y_0``, ``z_0`` (and
    ``args``, ``S``) are differentiation-inert.
    ``args`` is inert pass-through data. Local Jacobians come from AD instead
    of scipy's finite differences; ``fun_jac_ad``/``bc_jac_ad`` choose
    ``"jvp"``, ``"vjp"``, or ``"auto"`` (forward when square or tall, reverse
    when strictly fat). ``max_nodes`` (static, default 128) fixes the padded
    output length: the mesh ``t`` starts from the given guess and grows under
    refinement, the returned tail repeats ``t[-1]`` and the last active rows,
    and ``sol.num_nodes`` counts active nodes, so
    ``hermite_interpolate(ts, sol.t, sol.y, sol.yp)`` evaluates exactly
    scipy's returned C1 cubic spline (``hermite_derivative`` its derivative).
    ``fun`` may return ``(value, aux)``; aux is evaluated once at the solution
    over all padded nodes and participates in AD. ``tol`` is floored at
    ``100 * eps`` of the working dtype (taken from ``y_0``), silently.
    Failures never raise inside the solve: ``sol.status`` carries scipy's
    codes and a singular collocation Jacobian is reported as status 2 with the
    last iterate returned. The collocation system is factored once per Newton
    refresh by a structured orthogonal factorization of its bordered
    almost-block-diagonal form (``O(max_nodes)`` instead of the dense cubic,
    where scipy uses sparse LU). The whole solve is compiled with
    ``lax.while_loop`` loops, keyed on the identity of ``fun`` and ``bc``
    (reuse module-level functions rather than rebuilding closures per call);
    for repeated solves call it inside an outer ``jax.jit`` so the wrapper's
    per-call validation and dispatch trace away.
    """
    if not isinstance(max_nodes, int) or isinstance(max_nodes, bool) or max_nodes < 2:
        raise ValueError("max_nodes must be a static int of at least 2")
    for name, mode in (("fun_jac_ad", fun_jac_ad), ("bc_jac_ad", bc_jac_ad)):
        if mode not in ("auto", "jvp", "vjp"):
            raise ValueError(f'{name} must be "auto", "jvp", or "vjp"')

    y_0, dtype = asarray_state(y_0, "y_0")
    t = jnp.asarray(t, dtype)
    if t.ndim != 1:
        raise ValueError("t must be 1-dimensional")
    m_0 = t.shape[0]
    if m_0 < 2:
        raise ValueError("t must contain at least two nodes")
    if m_0 > max_nodes:
        raise ValueError(f"t has {m_0} nodes, more than max_nodes={max_nodes}")
    if not isinstance(t, jax.core.Tracer):
        if np.any(np.diff(np.asarray(t)) <= 0):
            raise ValueError("t must be strictly increasing")
    for leaf in jax.tree.leaves(y_0):
        if leaf.shape[0] != m_0:
            raise ValueError("y_0 leaves must have leading axis len(t)")

    node_template = jax.tree.map(lambda leaf: leaf[0], y_0)
    flat_node, unravel_y = ravel_pytree(node_template)
    n = flat_node.size
    Y_0 = jax.vmap(lambda node: ravel_pytree(node)[0])(y_0)

    if z_0 is None:
        k = 0
        Z_0 = jnp.zeros((0,), dtype)
        unravel_z = _unravel_empty
        z_treedef = None
        z_leaf_specs = None
    else:
        z_0, z_dtype = asarray_state(z_0, "z_0")
        if z_dtype != dtype:
            raise TypeError("z_0 must have the same dtype as y_0")
        Z_0, unravel_z = ravel_pytree(z_0)
        k = Z_0.size
        z_treedef = jax.tree.structure(z_0)
        z_leaf_specs = tuple(
            (leaf.shape, str(leaf.dtype)) for leaf in jax.tree.leaves(z_0)
        )

    if S is not None:
        S = jnp.asarray(S, dtype)
        if S.shape != (n, n):
            raise ValueError(f"S must have shape {(n, n)}, got {S.shape}")

    fun_arity = bvp_arity(
        fun, "fun", "(t, y), (t, y, z), (t, y, z, args), or (t, y, z, args, p)"
    )
    bc_arity = bvp_arity(
        bc, "bc", "(ya, yb), (ya, yb, z), (ya, yb, z, args), or (ya, yb, z, args, p)"
    )
    # Silently dropping p or args would make derivatives silently zero.
    if p is not None and fun_arity < 5 and bc_arity < 5:
        raise ValueError("p was passed but neither fun nor bc takes it")
    if args is not None and fun_arity < 4 and bc_arity < 4:
        raise ValueError("args was passed but neither fun nor bc takes it")
    if z_0 is not None and fun_arity < 3 and bc_arity < 3:
        raise ValueError("z_0 was passed but neither fun nor bc takes it")
    fun_canon = canonicalize_bvp_fun(fun, fun_arity)
    bc_canon = canonicalize_bvp_bc(bc, bc_arity)

    has_aux, _ = resolve_field_aux(
        fun_canon,
        (t[0], node_template, z_0, args, p),
        jax.tree.structure(node_template),
        has_aux,
        name="has_aux",
    )
    bc_shape = jax.eval_shape(
        lambda *operands: jnp.asarray(bc_canon(*operands)),
        node_template,
        node_template,
        z_0,
        args,
        p,
    )
    if bc_shape.shape != (n + k,):
        raise ValueError(f"bc must return a 1-D array of {n + k} residuals")

    cfg = _BVPConfig(
        fun=fun,
        bc=bc,
        fun_arity=fun_arity,
        bc_arity=bc_arity,
        max_nodes=max_nodes,
        n=n,
        k=k,
        has_aux=has_aux,
        fun_jac_ad=fun_jac_ad,
        bc_jac_ad=bc_jac_ad,
        has_singular_term=S is not None,
        y_treedef=jax.tree.structure(node_template),
        y_leaf_specs=tuple(
            (leaf.shape, str(leaf.dtype)) for leaf in jax.tree.leaves(node_template)
        ),
        z_treedef=z_treedef,
        z_leaf_specs=z_leaf_specs,
        fun_canon=fun_canon,
        bc_canon=bc_canon,
        unravel_y=unravel_y,
        unravel_z=unravel_z,
    )

    pad = max_nodes - m_0
    t_pad = jnp.concatenate([t, jnp.full((pad,), t[-1], dtype)])
    Y_pad = jnp.concatenate([Y_0, jnp.broadcast_to(Y_0[-1], (pad, n))])
    bc_tol_value = jnp.asarray(jnp.nan if bc_tol is None else bc_tol, dtype)
    t_f, Y_f, Z_f, f_f, rms, num_nodes, num_iterations, status, aux = _run_with_ad(
        cfg,
        t_pad,
        Y_pad,
        Z_0,
        jnp.asarray(m_0, jnp.int32),
        p,
        args,
        S,
        jnp.asarray(tol, dtype),
        bc_tol_value,
    )
    return BVPSolution(
        t=t_f,
        y=jax.vmap(unravel_y)(Y_f),
        yp=jax.vmap(unravel_y)(f_f),
        z=unravel_z(Z_f),
        rms_residuals=rms,
        num_nodes=num_nodes,
        num_iterations=num_iterations,
        status=status,
        ok=status == STATUS_CONVERGED,
        aux=aux,
    )
