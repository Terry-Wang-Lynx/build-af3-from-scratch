"""Test inputs + helpers for ``diffusion/`` modules.

``diffusion/`` 章的固定测试输入。
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


# Small constants for diffusion-side blocks.
N_batch = 1
N_token = 6
N_sample = 1
c_a     = 24
c_s     = 24
c_z     = 12
n_heads = 4
n_blocks = 2


def _ramp(shape: tuple, idx: int) -> torch.Tensor:
    lo = -2.0 - idx / 5
    hi = +2.0 + idx / 5
    n = math.prod(shape)
    return torch.linspace(lo, hi, n).reshape(shape).double()


feature_shapes = {
    "a":  (N_sample, N_token, c_a),
    "s":  (N_sample, N_token, c_s),
    "z":  (N_sample, N_token, N_token, c_z),
}
test_inputs = {name: _ramp(shape, i) for i, (name, shape) in enumerate(feature_shapes.items())}


# ----- Geometry helpers (Algorithms 19 + 29) -------------------------------
# Geometry test inputs are independent of the Transformer ones above.
# 几何 helper 的测试输入与上面的 Transformer 输入相互独立。
N_atom_geo  = 7
N_frame_geo = 3
test_inputs["coords"]      = _ramp((N_atom_geo, 3), 4)
test_inputs["frame_atoms"] = _ramp((N_frame_geo, 3, 3), 5)
