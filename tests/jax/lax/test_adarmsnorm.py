###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from primus_turbo.jax.lax.normalization import adarmsnorm


def adarmsnorm_ref(x, gamma, ada_scale, ada_shift, eps):
    norm = jnp.sqrt(jnp.mean(x**2, axis=-1, keepdims=True) + eps)
    y = x / norm * gamma
    return y * (1 + ada_scale) + ada_shift


def get_tolerances(dtype):
    if dtype == jnp.float32:
        return dict(rtol=1e-4, atol=1e-4)
    elif dtype in (jnp.float16, jnp.bfloat16):
        return dict(rtol=1e-2, atol=1e-2)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


@pytest.mark.parametrize("shape", [(1024, 4096)])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float16, jnp.bfloat16])
def test_adarmsnorm_lax(shape, dtype):
    key = jax.random.PRNGKey(0)
    # Generate in float32 then cast to avoid NaN from direct low-precision sampling
    x = jax.random.normal(key, shape, jnp.float32).astype(dtype)
    gamma = jax.random.normal(key, (shape[-1],), jnp.float32).astype(dtype)
    ada_scale = jax.random.normal(key, (shape[-1],), jnp.float32).astype(dtype)
    ada_shift = jax.random.normal(key, (shape[-1],), jnp.float32).astype(dtype)
    eps = 1e-6

    y = adarmsnorm(x, gamma, ada_scale, ada_shift, eps)
    y_ref = adarmsnorm_ref(x, gamma, ada_scale, ada_shift, eps)

    tols = get_tolerances(dtype)
    np.testing.assert_allclose(np.array(y, dtype=np.float32), np.array(y_ref, dtype=np.float32), **tols)
