"""Test inputs + helpers for ``confidence/`` modules.

``confidence/`` 章的固定测试输入。
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


N_token = 6
c_z = 12
no_bins = 64


def _ramp(shape: tuple, idx: int) -> torch.Tensor:
    lo = -2.0 - idx / 5
    hi = +2.0 + idx / 5
    n = math.prod(shape)
    return torch.linspace(lo, hi, n).reshape(shape).double()


test_inputs = {
    "z": _ramp((N_token, N_token, c_z), 0),
}
