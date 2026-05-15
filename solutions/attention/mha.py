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
    ##########################################################################
    # TODO: Scaled dot-product attention — the math heart of every attention #
    #   block. Two compute paths: the efficient                              #
    #   ``F.scaled_dot_product_attention`` (kernel-fused), and an explicit   #
    #   matmul + softmax fallback.                                            #
    #                                                                        #
    #   Step 0 — Sanity check + upcast to fp32 for numerical stability:      #
    #       assert k.shape == v.shape                                        #
    #       input_dtype = q.dtype                                            #
    #       q = q.to(dtype=torch.float32)                                    #
    #       k = k.to(dtype=torch.float32)                                    #
    #       if attn_bias is not None:                                        #
    #           attn_bias = attn_bias.to(dtype=torch.float32)                #
    #                                                                        #
    #   Step 1 — Efficient path. The caller has already scaled Q by          #
    #     1/sqrt(d_head), so pass scale=1.0:                                  #
    #       if use_efficient_implementation:                                  #
    #           return F.scaled_dot_product_attention(                        #
    #               query=q, key=k, value=v,                                  #
    #               attn_mask=attn_bias, scale=1.0,                           #
    #           )                                                             #
    #                                                                        #
    #   Step 2 — Explicit math path under autocast(False):                   #
    #     with torch.amp.autocast("cuda", enabled=False):                     #
    #         k = k.transpose(-1, -2)              # [..., d, n_kv]          #
    #         attn_weights = q @ k                 # [..., n_q, n_kv]        #
    #         if attn_bias is not None:                                       #
    #             if inplace_safe:                                            #
    #                 attn_weights += attn_bias                               #
    #             else:                                                       #
    #                 attn_weights = attn_weights + attn_bias                 #
    #         attn_weights = F.softmax(attn_weights, dim=-1)                  #
    #                                                                        #
    #   Step 3 — Cast weights back, weighted-sum across keys:                 #
    #     attn_output = attn_weights.to(dtype=input_dtype) @ v               #
    #     return attn_output                                                  #
    #                                                                        #
    # TODO: 缩放点积注意力 —— 所有 attention 块的数学核心。两条路径:           #
    #   高效路径 ``F.scaled_dot_product_attention`` (内核融合)；               #
    #   显式 matmul + softmax 路径 (清晰可移植)。                              #
    #                                                                        #
    #   步骤 0 — 形状校验 + 数值稳定升 fp32:                                    #
    #       assert k.shape == v.shape                                        #
    #       input_dtype = q.dtype                                            #
    #       q = q.to(dtype=torch.float32)                                    #
    #       k = k.to(dtype=torch.float32)                                    #
    #       if attn_bias is not None:                                        #
    #           attn_bias = attn_bias.to(dtype=torch.float32)                #
    #                                                                        #
    #   步骤 1 — 高效路径。Q 已在调用前按 1/sqrt(d_head) 缩放，scale=1.0:       #
    #       if use_efficient_implementation:                                  #
    #           return F.scaled_dot_product_attention(                        #
    #               query=q, key=k, value=v,                                  #
    #               attn_mask=attn_bias, scale=1.0,                           #
    #           )                                                             #
    #                                                                        #
    #   步骤 2 — autocast(False) 下的显式数学路径:                              #
    #     with torch.amp.autocast("cuda", enabled=False):                     #
    #         k = k.transpose(-1, -2)              # [..., d, n_kv]          #
    #         attn_weights = q @ k                 # [..., n_q, n_kv]        #
    #         if attn_bias is not None:                                       #
    #             if inplace_safe:                                            #
    #                 attn_weights += attn_bias                               #
    #             else:                                                       #
    #                 attn_weights = attn_weights + attn_bias                 #
    #         attn_weights = F.softmax(attn_weights, dim=-1)                  #
    #                                                                        #
    #   步骤 3 — 把权重转回输入 dtype，沿 key 维加权和:                          #
    #     attn_output = attn_weights.to(dtype=input_dtype) @ v               #
    #     return attn_output                                                  #
    ##########################################################################

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

    ##########################################################################
    #               END OF YOUR CODE                                         #
    ##########################################################################


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
        # TODO: Initialize the Q / K / V / output (and optional gate) linears.   #
        #   Let ``out_h = c_hidden * num_heads`` (total per-token hidden width). #
        #                                                                        #
        #   self.linear_q:                                                       #
        #       Linear      (c_q -> out_h)   if ``q_linear_bias`` is True        #
        #       LinearNoBias(c_q -> out_h)   otherwise                           #
        #   self.linear_k = LinearNoBias(c_k    -> out_h)                        #
        #   self.linear_v = LinearNoBias(c_v    -> out_h)                        #
        #   self.linear_o = LinearNoBias(out_h  -> c_q)                          #
        #                                                                        #
        #   Gating branch (when ``gating`` is True, default for AF3):            #
        #       self.linear_g = LinearNoBias(c_q -> out_h, initializer="zeros") #
        #       self.sigmoid  = nn.Sigmoid()                                     #
        #   Else: set ``self.linear_g = None`` (used to skip gating in forward). #
        #                                                                        #
        #   Zero-init branch (when ``zero_init`` is True): zero out              #
        #       ``self.linear_o.weight`` so the attention block starts as a      #
        #       no-op residual (AF3 standard init).                              #
        #                                                                        #
        #   The attribute names linear_q / linear_k / linear_v / linear_o /      #
        #   linear_g are **load-bearing** — they must match the Protenix         #
        #   checkpoint state_dict keys exactly.                                  #
        #                                                                        #
        # TODO: 初始化 Q / K / V / 输出(可选门控) 五个线性层。                   #
        #   令 ``out_h = c_hidden * num_heads`` (每 token 的总隐藏宽度)。         #
        #                                                                        #
        #   self.linear_q:                                                       #
        #       Linear      (c_q -> out_h)   若 ``q_linear_bias`` 为 True        #
        #       LinearNoBias(c_q -> out_h)   否则                                #
        #   self.linear_k = LinearNoBias(c_k    -> out_h)                        #
        #   self.linear_v = LinearNoBias(c_v    -> out_h)                        #
        #   self.linear_o = LinearNoBias(out_h  -> c_q)                          #
        #                                                                        #
        #   门控分支 (``gating`` 为 True，AF3 默认):                              #
        #       self.linear_g = LinearNoBias(c_q -> out_h, initializer="zeros") #
        #       self.sigmoid  = nn.Sigmoid()                                     #
        #   否则: ``self.linear_g = None`` (用于在 forward 跳过门控)。           #
        #                                                                        #
        #   零初始化分支 (``zero_init`` 为 True): 将                              #
        #       ``self.linear_o.weight`` 置零，使该 attention 块以                #
        #       残差恒等开始 (AF3 标准初始化)。                                  #
        #                                                                        #
        #   linear_q / linear_k / linear_v / linear_o / linear_g 这五个属性名     #
        #   是**关键约束** —— 必须与 Protenix 权重 state_dict 的 key 完全一致。 #
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
        ##########################################################################
        # TODO: Q / K / V preparation for multi-head attention.                  #
        #                                                                        #
        #   Step 1 — Project from token features into the head-fused hidden     #
        #     width ``H * C_hidden``. Q comes from ``q_x`` (cross-attention     #
        #     query), K and V from ``kv_x`` (cross-attention key/value; for     #
        #     self-attention the caller passes the same tensor twice):          #
        #       q = self.linear_q(q_x)        # [*, Q,   H*C_hidden]            #
        #       k = self.linear_k(kv_x)       # [*, K,   H*C_hidden]            #
        #       v = self.linear_v(kv_x)       # [*, V=K, H*C_hidden]            #
        #                                                                        #
        #   Step 2 — Split the trailing head dim:                                #
        #       q = q.view(q.shape[:-1] + (self.num_heads, -1))                 #
        #       k = k.view(k.shape[:-1] + (self.num_heads, -1))                 #
        #       v = v.view(v.shape[:-1] + (self.num_heads, -1))                 #
        #                                                                        #
        #   Step 3 — Move the head dim ahead of the token dim so attention      #
        #     sums over the right axis:                                          #
        #       q = q.transpose(-2, -3)       # [*, H, Q,   C_hidden]           #
        #       k = k.transpose(-2, -3)       # [*, H, K,   C_hidden]           #
        #       v = v.transpose(-2, -3)       # [*, H, V=K, C_hidden]           #
        #                                                                        #
        #   Step 4 — Scale Q by 1/sqrt(c_hidden) ahead of the dot product so    #
        #     the actual attention math can pass scale=1.0:                     #
        #       if apply_scale:                                                  #
        #           q = q / math.sqrt(self.c_hidden)                            #
        #                                                                        #
        #   Return ``(q, k, v)``.                                                #
        #                                                                        #
        # TODO: 为多头注意力准备 Q / K / V。                                      #
        #                                                                        #
        #   步骤 1 — 把 token 特征投到融合了头维的隐藏宽度 ``H * C_hidden``。     #
        #     Q 来自 ``q_x``，K / V 来自 ``kv_x`` (自注意力时调用方传同一张量):    #
        #       q = self.linear_q(q_x)        # [*, Q,   H*C_hidden]            #
        #       k = self.linear_k(kv_x)       # [*, K,   H*C_hidden]            #
        #       v = self.linear_v(kv_x)       # [*, V=K, H*C_hidden]            #
        #                                                                        #
        #   步骤 2 — 拆出头维:                                                     #
        #       q = q.view(q.shape[:-1] + (self.num_heads, -1))                 #
        #       k = k.view(k.shape[:-1] + (self.num_heads, -1))                 #
        #       v = v.view(v.shape[:-1] + (self.num_heads, -1))                 #
        #                                                                        #
        #   步骤 3 — 把头维移到 token 维之前 (注意力沿正确轴求和):                  #
        #       q = q.transpose(-2, -3)       # [*, H, Q,   C_hidden]           #
        #       k = k.transpose(-2, -3)       # [*, H, K,   C_hidden]           #
        #       v = v.transpose(-2, -3)       # [*, H, V=K, C_hidden]           #
        #                                                                        #
        #   步骤 4 — 提前用 1/sqrt(c_hidden) 缩放 Q (下游 _attention 才能传      #
        #     scale=1.0):                                                       #
        #       if apply_scale:                                                  #
        #           q = q / math.sqrt(self.c_hidden)                            #
        #                                                                        #
        #   返回 ``(q, k, v)``。                                                  #
        ##########################################################################

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

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

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
        ##########################################################################
        # TODO: Post-attention "wrap up" — optional sigmoid gating + output     #
        #   projection back to ``c_q``.                                          #
        #                                                                        #
        #   Step 1 — Gating (only when ``self.linear_g is not None``,           #
        #     i.e. ``gating=True``). The gate is sigmoid(linear over the        #
        #     **original** query ``q_x``, not the attention output), reshaped   #
        #     per head, then multiplied element-wise:                            #
        #       if self.linear_g is not None:                                    #
        #           g = self.sigmoid(self.linear_g(q_x))   # [*, Q, H*C_hidden] #
        #           g = g.view(g.shape[:-1] + (self.num_heads, -1))             #
        #                                              # [*, Q, H, C_hidden]    #
        #           o = o * g                          # element-wise gate      #
        #                                                                        #
        #   Step 2 — Flatten the last two dims (head, hidden) back into a       #
        #     single (H * C_hidden) channel:                                    #
        #       o = flatten_final_dims(o, num_dims=2)   # [*, Q, H*C_hidden]    #
        #                                                                        #
        #   Step 3 — Project to ``c_q`` (the attention input width):            #
        #       o = self.linear_o(o)                    # [*, Q, c_q]           #
        #   Return ``o``.                                                        #
        #                                                                        #
        # TODO: 注意力后的"收尾" —— 可选 sigmoid 门控 + 输出投影回 ``c_q``。      #
        #                                                                        #
        #   步骤 1 — 门控 (仅 ``self.linear_g is not None``，即 ``gating=True``)。 #
        #     门由 sigmoid(linear over 原始 ``q_x``) 算 (注意是 q_x 而非 attn   #
        #     输出 o)，reshape 出头维后逐元素相乘:                                #
        #       if self.linear_g is not None:                                    #
        #           g = self.sigmoid(self.linear_g(q_x))   # [*, Q, H*C_hidden] #
        #           g = g.view(g.shape[:-1] + (self.num_heads, -1))             #
        #                                              # [*, Q, H, C_hidden]    #
        #           o = o * g                                                   #
        #                                                                        #
        #   步骤 2 — 把最后两维 (head, hidden) 展平回单通道 (H * C_hidden):        #
        #       o = flatten_final_dims(o, num_dims=2)   # [*, Q, H*C_hidden]    #
        #                                                                        #
        #   步骤 3 — 投到 ``c_q`` (attention 输入宽度):                            #
        #       o = self.linear_o(o)                    # [*, Q, c_q]           #
        #   返回 ``o``。                                                          #
        ##########################################################################

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

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

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
        #   Step 1 — Project Q/K/V from q_x / kv_x:                              #
        #       q, k, v = self._prep_qkv(q_x=q_x, kv_x=kv_x, apply_scale=True)   #
        #     This linearly projects, splits into heads, transposes              #
        #     [*, T, H*C] -> [*, H, T, C], and scales q by 1/sqrt(c_hidden).     #
        #                                                                        #
        #   Step 2 — Broadcast bias to the head dim if necessary:                #
        #       if attn_bias is not None and                                     #
        #          len(attn_bias.shape) != len(q.shape):                         #
        #           attn_bias = attn_bias.unsqueeze(dim=-3)                      #
        #       Same with trunked_attn_bias (one extra trunk dim, so compare     #
        #          against len(q.shape) + 1 and unsqueeze at dim=-4).            #
        #                                                                        #
        #   Step 3 — Branch on whether we run **local** or **full** attention:   #
        #     if n_queries and n_keys:                                           #
        #         if self.local_attention_method == "global_attention_with_bias":#
        #             local_attn_bias = create_local_attn_bias(                  #
        #                 q.shape[-2], n_queries, n_keys,                        #
        #                 inf=inf, device=q.device,                              #
        #             )  # [n_q_total, n_kv_total] with -inf outside the window  #
        #             # broadcast to q's leading dims:                           #
        #             local_attn_bias = local_attn_bias.reshape(                 #
        #                 (1,) * len(q.shape[:-2]) + local_attn_bias.shape       #
        #             )                                                          #
        #             if attn_bias is not None:                                  #
        #                 local_attn_bias = local_attn_bias + attn_bias          #
        #             o = _attention(q, k, v, attn_bias=local_attn_bias,         #
        #                            use_efficient_implementation=               #
        #                                 self.use_efficient_implementation,     #
        #                            inplace_safe=inplace_safe)                  #
        #         elif self.local_attention_method == "local_cross_attention":   #
        #             o = _local_attention(                                      #
        #                 q=q, k=k, v=v,                                         #
        #                 n_queries=n_queries, n_keys=n_keys,                    #
        #                 attn_bias=attn_bias,                                   #
        #                 trunked_attn_bias=trunked_attn_bias,                   #
        #                 inf=inf,                                               #
        #                 use_efficient_implementation=                          #
        #                     self.use_efficient_implementation,                 #
        #                 inplace_safe=inplace_safe,                             #
        #                 chunk_size=chunk_size,                                 #
        #             )                                                          #
        #         else:                                                          #
        #             raise ValueError(...)                                      #
        #     else:                                                              #
        #         o = _attention(q, k, v, attn_bias=attn_bias,                   #
        #                        use_efficient_implementation=                   #
        #                            self.use_efficient_implementation,          #
        #                        inplace_safe=inplace_safe)                      #
        #                                                                        #
        #   Step 4 — Permute heads back and project out:                         #
        #       o = o.transpose(-2, -3)            # [*, Q, H, C_hidden]         #
        #       o = self._wrap_up(o, q_x)          # gate (if any) + linear_o    #
        #   Return o ([*, Q, c_q]).                                              #
        #                                                                        #
        # TODO: 多头注意力前向。                                                 #
        #   步骤 1 — 投影 Q/K/V：                                                 #
        #       q, k, v = self._prep_qkv(q_x=q_x, kv_x=kv_x, apply_scale=True)   #
        #     该方法做线性投影、拆头、转置                                       #
        #     [*, T, H*C] -> [*, H, T, C]，并将 q 缩放 1/sqrt(c_hidden)。         #
        #                                                                        #
        #   步骤 2 — 给 bias 广播 head 维度：                                     #
        #       if attn_bias is not None and                                     #
        #          len(attn_bias.shape) != len(q.shape):                         #
        #           attn_bias = attn_bias.unsqueeze(dim=-3)                      #
        #       trunked_attn_bias 多一个 trunk 维度，                             #
        #         比较 len(q.shape) + 1，在 dim=-4 unsqueeze。                    #
        #                                                                        #
        #   步骤 3 — 在“局部”和“全局”注意力之间分支:                              #
        #     if n_queries and n_keys:                                           #
        #         若 self.local_attention_method == "global_attention_with_bias"#
        #           调 create_local_attn_bias 生成窗口掩码（-inf 落在窗口外），   #
        #           reshape 到 q 的 leading dims 后 (若有) 加上 attn_bias，      #
        #           再调 _attention(q,k,v, attn_bias=local_attn_bias, ...)。    #
        #         若 self.local_attention_method == "local_cross_attention"     #
        #           调 _local_attention(q, k, v, n_queries, n_keys,             #
        #             attn_bias, trunked_attn_bias, inf, ...)。                 #
        #         其余值: raise ValueError。                                      #
        #     else: 直接调 _attention(q, k, v, attn_bias=attn_bias, ...)。       #
        #                                                                        #
        #   步骤 4 — 把 head 维换回，再做输出投影:                                #
        #       o = o.transpose(-2, -3)            # [*, Q, H, C_hidden]         #
        #       o = self._wrap_up(o, q_x)          # 门控(可选) + linear_o       #
        #   返回 o ([*, Q, c_q])。                                                #
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