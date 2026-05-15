"""Test inputs + chapter-level helpers for ``attention/``.

``attention/`` 章的固定测试输入与本章辅助函数。

Shapes are intentionally small so every test runs in well under a
second on CPU. They are NOT meant to represent realistic AF3 sizes —
only to exercise every code path.

形状刻意取得很小，所有测试在 CPU 上都不到一秒。它们的目的只是覆盖
每条代码路径，并不反映真实的 AF3 规模。
"""
from __future__ import annotations

import math
import os

import torch

# Make the shared helpers importable as ``from attention.control_values.attention_checks import ...``.
# 让共享 helper 可以从本模块再导出。
from runtime.checks import (  # noqa: F401
    controlled_execution,
    controlled_forward,
    test_module_forward,
    test_module_method,
    test_module_shape,
)


CONTROL_FOLDER = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------------------------
# Tiny shape constants — all tests share these.
# 极小的形状常量 —— 所有测试共用。
# ----------------------------------------------------------------------------
N_batch  = 2     # leading batch size
N_token  = 5     # tokens / atoms (Q / K / V dimension)
N_head   = 4     # number of attention heads
c_hidden = 6     # per-head hidden width
c_a      = N_head * c_hidden   # token / atom channel = 24
c_s      = 16    # single channel
c_z      = 12    # pair channel
n_factor = 4     # Transition widening factor


feature_shapes = {
    # ---- generic ----
    "x_a":    (N_token, c_a),                       # token activation
    "x_s":    (N_token, c_s),                       # single activation
    "x_z":    (N_token, N_token, c_z),              # pair activation

    # ---- multi-head attention ----
    "q_x":    (N_token, c_a),                       # query input
    "kv_x":   (N_token, c_a),                       # key/value input
    "attn_bias":   (N_head, N_token, N_token),      # per-head bias

    # ---- raw _attention(q, k, v) helper ----
    "q_raw":  (N_head, N_token, c_hidden),
    "k_raw":  (N_head, N_token, c_hidden),
    "v_raw":  (N_head, N_token, c_hidden),
}

# Batched variants prepend an extra leading dim.
# 批量版本前面再补一维。
batched_feature_shapes = {k: (N_batch,) + v for k, v in feature_shapes.items()}


def _ramp(shape: tuple, idx: int) -> torch.Tensor:
    """Deterministic ramp ``linspace(-2 - idx/5, 2 + idx/5, prod(shape))``.

    确定性等差张量 —— 用不同 ``idx`` 让不同特征互不重合。
    """
    lo = -2.0 - idx / 5
    hi = +2.0 + idx / 5
    n = math.prod(shape)
    return torch.linspace(lo, hi, n).reshape(shape).double()


test_inputs = {name: _ramp(shape, i) for i, (name, shape) in enumerate(feature_shapes.items())}
