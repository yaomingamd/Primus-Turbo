###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import pytest
import torch
import torch.nn.functional as F

from primus_turbo.pytorch.ops.normalization import adarmsnorm
from tests.pytorch.test_utils import get_tolerances


def adarmsnorm_ref(x, gamma, ada_scale, ada_shift, eps):
    y = F.rms_norm(x, [x.shape[-1]], gamma, eps)
    return y * (1 + ada_scale) + ada_shift


# @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("outer_shape", [(1,), (511,), (4096,), (8192,), (16384,)])
@pytest.mark.parametrize("inner_shape", [33, 513, 4096, 5120, 7168, 8192])
def test_adarmsnorm_ops(dtype, outer_shape, inner_shape):
    torch.manual_seed(1)
    device = "cuda:0"
    eps = 1e-6

    shape = outer_shape + (inner_shape,)
    x = torch.randn(shape, dtype=dtype, device=device)
    gamma = torch.randn(inner_shape, dtype=dtype, device=device)
    ada_scale = torch.randn(inner_shape, dtype=dtype, device=device)
    ada_shift = torch.randn(inner_shape, dtype=dtype, device=device)

    y = adarmsnorm(x, gamma, ada_scale, ada_shift, eps)
    y_ref = adarmsnorm_ref(x, gamma, ada_scale, ada_shift, eps)

    torch.testing.assert_close(y, y_ref, **get_tolerances(dtype))
