###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from primus_turbo.jax.lax.normalization import rmsnorm


def rmsnorm_ref(x, gamma, eps):
    norm = jnp.sqrt(jnp.mean(x**2, axis=-1, keepdims=True) + eps)
    return x / norm * gamma


def get_tolerances(dtype):
    if dtype == jnp.float32:
        return dict(rtol=1e-4, atol=1e-4)
    elif dtype in (jnp.float16, jnp.bfloat16):
        return dict(rtol=1e-2, atol=1e-2)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


@pytest.mark.parametrize("shape", [(1024, 4096)])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float16, jnp.bfloat16])
def test_rmsnorm_lax(shape, dtype):
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, shape, jnp.float32).astype(dtype)
    gamma = jax.random.normal(key, (shape[-1],), jnp.float32).astype(dtype)
    eps = 1e-6

    tols = get_tolerances(dtype)

    #######################################
    # Fwd
    y = rmsnorm(x, gamma, eps)
    y_ref = rmsnorm_ref(x, gamma, eps)

    np.testing.assert_allclose(np.array(y, dtype=np.float32), np.array(y_ref, dtype=np.float32), **tols)

    #######################################
    # Backward w.r.t both x and gamma
    def loss_fn(x, gamma):
        return jnp.sum(rmsnorm(x, gamma, eps))

    def loss_fn_ref(x, gamma):
        return jnp.sum(rmsnorm_ref(x, gamma, eps))

    grad_fn = jax.grad(loss_fn, argnums=(0, 1))
    grad_fn_ref = jax.grad(loss_fn_ref, argnums=(0, 1))

    grad_x, grad_gamma = grad_fn(x, gamma)
    grad_x_ref, grad_gamma_ref = grad_fn_ref(x, gamma)

    np.testing.assert_allclose(np.array(grad_x, dtype=np.float32), np.array(grad_x_ref, dtype=np.float32), **tols)
    np.testing.assert_allclose(np.array(grad_gamma, dtype=np.float32), np.array(grad_gamma_ref, dtype=np.float32), **tols)
