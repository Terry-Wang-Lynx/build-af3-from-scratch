"""Test inputs + helpers for ``pairformer/`` modules.

``pairformer/`` 章的固定测试输入与辅助。
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


# Tiny shapes — kept independent from the attention chapter so each chapter
# can vary them.
# 每章独立的极小尺寸常量。
N_batch  = 2
N_token  = 5     # pair grid side
N_msa    = 4     # MSA sequences (sub-sampled)
c_m      = 8     # MSA channel
c_z      = 12    # pair channel
c_hidden = 6     # triangle hidden
no_heads_pair = 4   # triangle attention heads


feature_shapes = {
    "z":        (N_token, N_token, c_z),
    "m":        (N_msa, N_token, c_m),
    "pair_mask": (N_token, N_token),
    "msa_mask":  (N_msa, N_token),
}
batched_feature_shapes = {k: (N_batch,) + v for k, v in feature_shapes.items()}


def _ramp(shape: tuple, idx: int) -> torch.Tensor:
    lo = -2.0 - idx / 5
    hi = +2.0 + idx / 5
    n = math.prod(shape)
    return torch.linspace(lo, hi, n).reshape(shape).double()


test_inputs = {name: _ramp(shape, i) for i, (name, shape) in enumerate(feature_shapes.items())}
# Masks are 0/1, not the ramp:
test_inputs["pair_mask"] = torch.ones(feature_shapes["pair_mask"]).double()
test_inputs["msa_mask"]  = torch.ones(feature_shapes["msa_mask"]).double()
