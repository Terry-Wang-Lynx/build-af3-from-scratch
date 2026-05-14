# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from functools import partial
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from attention.linear import Linear, LinearNoBias
from feature_embedding.local_attention import (
    _local_attention,
    create_local_attn_bias,
    optimized_concat_split,
    rearrange_to_dense_trunk,
)
from pairformer.triangle_ops import LayerNorm, trunc_normal_init_
from model.utils import (
    chunk_layer,
    flatten_final_dims,
    move_final_dim_to_dim,
    pad_at_dim,
    reshape_at_dim,
)



def _attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_bias: Optional[torch.Tensor] = None,
    use_efficient_implementation: bool = True,
    inplace_safe: bool = False,
) -> torch.Tensor:
    """Attention.

    Args:
        q (torch.Tensor): query tensor of shape [..., n_q, d]
        k (torch.Tensor): key tensor of shape [..., n_kv, d]
        v (torch.Tensor): value tensor of shape[..., n_kv, d]
        attn_bias (torch.Tensor, optional): attention bias tensor of shape [..., n_q, n_kv]. Defaults to None.
        use_efficient_implementation (bool): whether to use the torch.nn.functional.scaled_dot_product_attention, Defaults to True.

    Returns:
        torch.Tensor: output of tensor [..., n_q, d]
    """
    assert k.shape == v.shape

    # Upcast to compute attn_weights
    input_dtype = q.dtype
    q = q.to(dtype=torch.float32)
    k = k.to(dtype=torch.float32)
    if attn_bias is not None:
        attn_bias = attn_bias.to(dtype=torch.float32)

    if use_efficient_implementation:
        attn_output = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=attn_bias,
            scale=1.0,
        )
        return attn_output

    with torch.amp.autocast("cuda", enabled=False):
        # [..., n_kv, d] -> [..., d, n_kv]
        k = k.transpose(-1, -2)

        # [..., n_q, d], [..., d, n_kv] -> [..., n_q, n_kv]
        attn_weights = q @ k

        if attn_bias is not None:
            if inplace_safe:
                attn_weights += attn_bias
            else:
                attn_weights = attn_weights + attn_bias

        # [..., n_q, n_kv]
        attn_weights = F.softmax(attn_weights, dim=-1)

    # [..., n_q, n_kv], [..., n_kv, d] -> [..., n_q, d]
    attn_output = attn_weights.to(dtype=input_dtype) @ v

    return attn_output


class Attention(nn.Module):
    """Standard multi-head attention
    Ref to openfold:
    https://github.com/aqlaboratory/openfold/blob/feb45a521e11af1db241a33d58fb175e207f8ce0/openfold/model/primitives.py#L340

    Args:
        c_q (int): Input dimension of query data
        c_k (int): Input dimension of key data
        c_v (int): Input dimension of value data
        c_hidden (int): Per-head hidden dimension
        num_heads (int): Number of attention heads
        gating (bool, optional): Whether the output should be gated using query data. Defaults to True.
        q_linear_bias (bool, optional): whether use Linear with bias as in AF3. Defaults to True.
        local_attention_method (str, optional): local attention method, options:
          - global_attention_with_bias: use full size global attention with sparse attention bias
          - local_cross_attention: use local cross attention to minimize computation
        use_efficient_implementation (bool): whether to use the torch.nn.functional.scaled_dot_product_attention, Defaults to True.
        zero_init (bool, optional): whether to zero-initialize the output layer. Defaults to True.

    Notes:
        if use_efficient_implementation == True, torch.nn.functional.scaled_dot_product_attention will
        be used to compute attention efficiently
        There are currently three supported implementations of scaled dot product attention:
            1. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness

            2. Memory-Efficient Attention

            3. A PyTorch implementation defined in C++ matching the above formulation

        The function may call optimized kernels for improved performance when using the CUDA backend.
        For all other backends, the PyTorch implementation will be used.All implementations are enabled by default.
        Scaled dot product attention attempts to automatically select the most optimal implementation based on the inputs.
    """

    def __init__(
        self,
        c_q: int,
        c_k: int,
        c_v: int,
        c_hidden: int,
        num_heads: int,
        gating: bool = True,
        q_linear_bias: bool = True,
        local_attention_method: str = "global_attention_with_bias",
        use_efficient_implementation: bool = True,
        zero_init: bool = True,
    ) -> None:
        super(Attention, self).__init__()
        self.c_q = c_q
        self.c_k = c_k
        self.c_v = c_v
        self.c_hidden = c_hidden
        self.num_heads = num_heads
        self.gating = gating
        self.local_attention_method = local_attention_method
        self.use_efficient_implementation = use_efficient_implementation
        self.zero_init = zero_init

        ##########################################################################
        # TODO: Initialize the query/key/value/output linear layers.             #
        #   Query uses bias iff ``q_linear_bias`` is True (default for AF3).     #
        #   Key/value/output use no bias. Output dim = c_hidden * num_heads.     #
        #   If ``gating``, initialize ``linear_g`` (zero-init) and a sigmoid.    #
        #   Names must be linear_q / linear_k / linear_v / linear_o / linear_g.  #
        #                                                                        #
        # TODO: 初始化 query / key / value / output 线性层。                     #
        #   query 是否带 bias 由 ``q_linear_bias`` 决定 (AF3 默认带)。            #
        #   key / value / output 均不带 bias，输出维度 = c_hidden * num_heads。  #
        #   若 ``gating``，初始化 ``linear_g`` (zero-init) 和 sigmoid。           #
        #   命名必须是 linear_q / linear_k / linear_v / linear_o / linear_g。    #
        ##########################################################################

        out_h = self.c_hidden * self.num_heads
        if q_linear_bias:
            self.linear_q = Linear(in_features=self.c_q, out_features=out_h)
        else:
            self.linear_q = LinearNoBias(self.c_q, out_h)
        self.linear_k = LinearNoBias(self.c_k, out_h)
        self.linear_v = LinearNoBias(self.c_v, out_h)
        self.linear_o = LinearNoBias(out_h, self.c_q)
        self.linear_g = None
        if self.gating:
            self.linear_g = LinearNoBias(self.c_q, out_h, initializer="zeros")
            self.sigmoid = nn.Sigmoid()
        if self.zero_init:
            nn.init.zeros_(self.linear_o.weight)

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

    def _prep_qkv(
        self, q_x: torch.Tensor, kv_x: torch.Tensor, apply_scale: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare qkv

        Args:
            q_x (torch.Tensor): the input x for q
                [..., c_q]
            kv_x (torch.Tensor): the input x for kv
                [..., c_k]
                [..., c_v]
            apply_scale (bool, optional): apply scale to dot product qk. Defaults to True.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: the return q/k/v
                # [..., H, Q/K/V, C_hidden]
        """
        # [*, Q/K/V, H * C_hidden]
        q = self.linear_q(q_x)
        k = self.linear_k(kv_x)
        v = self.linear_v(kv_x)

        # [*, Q/K/V, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.num_heads, -1))
        k = k.view(k.shape[:-1] + (self.num_heads, -1))
        v = v.view(v.shape[:-1] + (self.num_heads, -1))

        # [*, H, Q/K/V, C_hidden]
        q = q.transpose(-2, -3)
        k = k.transpose(-2, -3)
        v = v.transpose(-2, -3)

        if apply_scale:
            q = q / math.sqrt(self.c_hidden)

        return q, k, v

    def _wrap_up(self, o: torch.Tensor, q_x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            o (torch.Tensor): the output of attention
                [..., G/Q, H, C_hidden]
            q_x (torch.Tensor): the input for gated g
                [..., Q, c_q]

        Returns:
            torch.Tensor: the output of attention
        """
        if self.linear_g is not None:
            g = self.sigmoid(self.linear_g(q_x))

            # [*, G/Q, H, C_hidden]
            g = g.view(g.shape[:-1] + (self.num_heads, -1))
            o = o * g

        # [*, Q, H * C_hidden]
        o = flatten_final_dims(o, num_dims=2)

        # [*, Q, C_q]
        o = self.linear_o(o)

        return o

    def forward(
        self,
        q_x: torch.Tensor,
        kv_x: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
        trunked_attn_bias: Optional[torch.Tensor] = None,
        n_queries: Optional[int] = None,
        n_keys: Optional[int] = None,
        inf: Optional[float] = 1e10,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """

        Args:
            q_x (torch.Tensor): the input x for q
                [..., Q, C_q]
            kv_x (torch.Tensor): the input x for k/v
                [..., K, C_k]
            attn_bias (torch.Tensor, optional): the input biases for attention. Defaults to None.
                [..., H, Q, K] or [..., Q, K]
            trunked_attn_bias (torch.Tensor, optional): the input biases where shape has been rearranged to dense trunks. Defaults to None.
                [..., H, n_trunks, n_queries, n_keys] or [..., n_trunks, n_queries, n_keys]
            n_queries (int, optional): local window size of query tensor. If not None, will perform local attention. Defaults to None.
            n_keys (int, optional): local window size of key tensor. Defaults to None.

        Returns:
            torch.Tensor: attention update
                [*, Q, C_q]
        """
        ##########################################################################
        # TODO: Multi-head attention forward pass.                               #
        #   - Project Q/K/V via ``self._prep_qkv`` (scales Q by 1/sqrt(c_hidden))#
        #   - Add a head dimension to ``attn_bias`` / ``trunked_attn_bias``      #
        #   - Local-attention branch (n_queries/n_keys set):                     #
        #       * "global_attention_with_bias": build a local mask via          #
        #          ``create_local_attn_bias`` and call ``_attention``           #
        #       * "local_cross_attention": call ``_local_attention``             #
        #   - Otherwise: call ``_attention``                                     #
        #   - Transpose to [*, Q, H, C_hidden] and project out with ``_wrap_up`` #
        #                                                                        #
        # TODO: 多头注意力前向。                                                 #
        #   - 通过 ``self._prep_qkv`` 投影 Q/K/V (Q 已按 1/sqrt(c_hidden) 缩放)。#
        #   - 给 ``attn_bias`` / ``trunked_attn_bias`` 加 head 维度。             #
        #   - 若给了 n_queries / n_keys 则走局部分支:                            #
        #       * "global_attention_with_bias": 用 ``create_local_attn_bias``    #
        #          构造局部 mask，再调 ``_attention``。                         #
        #       * "local_cross_attention": 调 ``_local_attention``。            #
        #   - 否则: 直接 ``_attention``。                                        #
        #   - 转置到 [*, Q, H, C_hidden] 后通过 ``_wrap_up`` 出输出。            #
        ##########################################################################

        q, k, v = self._prep_qkv(q_x=q_x, kv_x=kv_x, apply_scale=True)

        if attn_bias is not None and len(attn_bias.shape) != len(q.shape):
            attn_bias = attn_bias.unsqueeze(dim=-3)
        if trunked_attn_bias is not None and len(trunked_attn_bias.shape) != len(q.shape) + 1:
            trunked_attn_bias = trunked_attn_bias.unsqueeze(dim=-4)

        if n_queries and n_keys:
            if self.local_attention_method == "global_attention_with_bias":
                local_attn_bias = create_local_attn_bias(
                    q.shape[-2], n_queries, n_keys, inf=inf, device=q.device
                )
                local_attn_bias = local_attn_bias.reshape(
                    (1,) * len(q.shape[:-2]) + local_attn_bias.shape
                )
                if attn_bias is not None:
                    local_attn_bias = local_attn_bias + attn_bias
                o = _attention(q, k, v, attn_bias=local_attn_bias,
                               use_efficient_implementation=self.use_efficient_implementation,
                               inplace_safe=inplace_safe)
            elif self.local_attention_method == "local_cross_attention":
                o = _local_attention(
                    q=q, k=k, v=v, n_queries=n_queries, n_keys=n_keys,
                    attn_bias=attn_bias, trunked_attn_bias=trunked_attn_bias,
                    inf=inf, use_efficient_implementation=self.use_efficient_implementation,
                    inplace_safe=inplace_safe, chunk_size=chunk_size,
                )
            else:
                raise ValueError(f"Invalid local_attention_method: {self.local_attention_method}")
        else:
            o = _attention(q, k, v, attn_bias=attn_bias,
                           use_efficient_implementation=self.use_efficient_implementation,
                           inplace_safe=inplace_safe)
        o = o.transpose(-2, -3)
        o = self._wrap_up(o, q_x)

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

        return o


def drop_path(
    x: torch.Tensor,
    drop_prob: float = 0.0,
    training: bool = False,
    scale_by_keep: bool = True,
) -> torch.Tensor:
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob,3):0.3f}"