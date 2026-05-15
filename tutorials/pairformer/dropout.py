# Copyright 2024 ByteDance and/or its affiliates.
# Licensed under the Apache License 2.0.
"""Dropout helpers used by the Pairformer / MSAModule blocks.

Pairformer 与 MSAModule 用到的 dropout 工具。

:func:`dropout_add_rowwise` fuses "row-shared dropout on ``x`` + residual
add to ``z``" into a single call. :func:`fixed_length_chunk` splits a tensor
along a dimension into fixed-size pieces (used by the MSA stack).

:func:`dropout_add_rowwise` 把 "对 ``x`` 做按行共享掩码的 dropout、再残差
加到 ``z``" 合并成一次调用。:func:`fixed_length_chunk` 沿某维把张量切成
定长 pieces (MSA stack 使用)。
"""
from __future__ import annotations

import torch


def dropout_add_rowwise(
    residual: torch.Tensor,
    x: torch.Tensor,
    p_drop: float,
    training: bool,
) -> torch.Tensor:
    """``residual + dropout_rowwise(x)`` in one call.

    一次完成 "residual + 按行共享掩码 dropout(x)"。

    The dropout mask is shared along dim -3 (i.e. all entries in the same row
    receive the same Bernoulli decision), matching :class:`DropoutRowwise`.

    Dropout 掩码沿 dim -3 共享 (同一行的所有元素采用同一个 Bernoulli 结果)，
    与 :class:`DropoutRowwise` 行为一致。

    Args / 参数:
        residual: tensor to add to. shape ``[..., R, C, D]``.
        x:        tensor that gets dropout applied. same shape as *residual*.
        p_drop:   dropout probability.
        training: if False or ``p_drop == 0``, the dropout is a no-op and the
                  function returns ``residual + x`` directly.
                  若为 False 或 p_drop=0，直接返回 ``residual + x``。

    Returns / 返回:
        ``residual + dropout_rowwise(x)`` with shape ``[..., R, C, D]``.
    """
    if not training or p_drop == 0.0:
        return residual + x

    shape = list(x.shape)
    shape[-3] = 1
    mask = x.new_ones(shape)
    mask = torch.nn.functional.dropout(mask, p=p_drop, training=True)
    return residual + x * mask


def fixed_length_chunk(x: torch.Tensor, chunk_size: int, dim: int = 0) -> list[torch.Tensor]:
    """Split ``x`` along ``dim`` into pieces of size at most ``chunk_size``.

    把 ``x`` 沿 ``dim`` 切成长度不超过 ``chunk_size`` 的若干段。

    Used by the MSA stack to process large MSAs in fixed-size batches. The
    last chunk may be smaller; we do not pad it.

    MSA stack 用它把大型 MSA 切成定长批次处理。最后一段可能短一些，不补齐。
    """
    n = x.size(dim)
    if chunk_size <= 0 or chunk_size >= n:
        return [x]
    return list(torch.split(x, chunk_size, dim=dim))
