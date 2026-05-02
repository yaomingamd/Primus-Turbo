###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from functools import partial

import jax
import jax.numpy as jnp

from primus_turbo.jax.primitive.normalization.normalization import (
    adarmsnorm_fwd_p,
    rmsnorm_bwd_p,
    rmsnorm_fwd_p,
)

__all__ = ["rmsnorm", "adarmsnorm"]


@partial(jax.custom_vjp, nondiff_argnums=(2,))
def rmsnorm(x: jnp.ndarray, gamma: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    return rmsnorm_fwd_p.bind(x, gamma, eps=eps)


# Ref: https://docs.jax.dev/en/latest/_autosummary/jax.custom_vjp.defvjp.html
# Input : same input signature as the underlying primal function
# Output: out, ctx
def _rmsnorm_fwd(x, gamma, eps):
    y = rmsnorm_fwd_p.bind(x, gamma, eps=eps)
    ctx = (x, gamma)
    return y, ctx


# input: nondiff_argnums, ctx, grad
# output: input grad
def _rmsnorm_bwd(eps, ctx, dy):
    x, gamma = ctx
    dx, dgamma = rmsnorm_bwd_p.bind(dy, x, gamma, eps=eps)
    # Sum per-row partial dgamma in float32 to avoid low-precision accumulation error
    return dx, jnp.sum(dgamma.astype(jnp.float32), axis=0).astype(x.dtype)


rmsnorm.defvjp(_rmsnorm_fwd, _rmsnorm_bwd)


def adarmsnorm(
    x: jnp.ndarray,
    gamma: jnp.ndarray,
    ada_scale: jnp.ndarray,
    ada_shift: jnp.ndarray,
    eps: float = 1e-6,
) -> jnp.ndarray:
    return adarmsnorm_fwd_p.bind(x, gamma, ada_scale, ada_shift, eps=eps)
