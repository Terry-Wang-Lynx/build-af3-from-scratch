"""Test inputs + helpers for ``feature_embedding/`` modules.

``feature_embedding/`` 章的固定测试输入。
"""
from __future__ import annotations

import math
import os

import torch

from runtime.checks import (  # noqa: F401
    controlled_execution,
    controlled_forward,
    test_module_forward,
    test_module_method,
    test_module_shape,
)


CONTROL_FOLDER = os.path.dirname(os.path.abspath(__file__))


# Small constants reused across feature-embedding tests.
N_batch  = 2
N_token  = 6
c_z      = 16
c_noise  = 12
r_max    = 3
s_max    = 2


def _ramp(shape: tuple, idx: int) -> torch.Tensor:
    lo = -2.0 - idx / 5
    hi = +2.0 + idx / 5
    n = math.prod(shape)
    return torch.linspace(lo, hi, n).reshape(shape).double()


# Pre-computed relp feature (channel width matches RelativePositionEncoding(r_max=3, s_max=2) = 4*3+2*2+7 = 23).
test_inputs = {
    "relp_feature": _ramp((N_token, N_token, 4 * r_max + 2 * s_max + 7), 0),
    "noise_level": torch.linspace(0.5, 4.0, 3).double(),
}
