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


@pytest.mark.parametrize("shape", [(1024, 4096)])
@pytest.mark.parametrize("dtype", [jnp.float32])
def test_adarmsnorm_lax(shape, dtype):
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, shape, dtype)
    gamma = jax.random.normal(key, (shape[-1],), dtype)
    ada_scale = jax.random.normal(key, (shape[-1],), dtype)
    ada_shift = jax.random.normal(key, (shape[-1],), dtype)
    eps = 1e-6

    y = adarmsnorm(x, gamma, ada_scale, ada_shift, eps)
    y_ref = adarmsnorm_ref(x, gamma, ada_scale, ada_shift, eps)

    print("y     ", y)
    print("y_ref ", y_ref)
    np.testing.assert_allclose(y, y_ref, rtol=1e-4, atol=1e-4)
