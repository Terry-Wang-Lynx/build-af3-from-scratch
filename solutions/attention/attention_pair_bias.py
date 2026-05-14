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

from functools import partial
from typing import Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from attention.transition import AdaptiveLayerNorm
from attention.mha import Attention, DropPath
from attention.linear import BiasInitLinear, LinearNoBias
from feature_embedding.local_attention import broadcast_token_to_local_atom_pair, rearrange_qk_to_dense_trunk
from pairformer.triangle_ops import LayerNorm
from model.utils import (
    aggregate_atom_to_token,
    broadcast_token_to_atom,
    checkpoint_blocks,
    permute_final_dims,
)



class AttentionPairBias(nn.Module):
    """
    Implements Algorithm 24 in AF3

    Args:
        has_s (bool, optional):  whether s is None as stated in Algorithm 24 Line1. Defaults to True.
        create_offset_ln_z (bool, optional): the value of create_offset for the LayerNorm applied to z. Defaults to False.
        n_heads (int, optional): number of attention-like head in AttentionPairBias. Defaults to 16.
        c_a (int, optional): the embedding dim of a(single feature aggregated atom info). Defaults to 768.
        c_s (int, optional):  hidden dim [for single embedding]. Defaults to 384.
        c_z (int, optional): hidden dim [for pair embedding]. Defaults to 128.
        biasinit (float, optional): biasinit for BiasInitLinear. Defaults to -2.0.
        cross_attention_mode (bool, optional): If cross_attention_model = True, the adaptive layernorm will be applied
            to query and key/value seperately. Defaults to False.
    """

    def __init__(
        self,
        has_s: bool = True,
        create_offset_ln_z: bool = False,
        n_heads: int = 16,
        c_a: int = 768,
        c_s: int = 384,
        c_z: int = 128,
        biasinit: float = -2.0,
        cross_attention_mode: bool = False,
    ) -> None:
        super(AttentionPairBias, self).__init__()
        assert c_a % n_heads == 0
        self.n_heads = n_heads
        self.has_s = has_s
        self.create_offset_ln_z = create_offset_ln_z
        self.cross_attention_mode = cross_attention_mode
        if has_s:
            # Line2
            self.layernorm_a = AdaptiveLayerNorm(c_a=c_a, c_s=c_s)
            if self.cross_attention_mode:
                self.layernorm_kv = AdaptiveLayerNorm(c_a=c_a, c_s=c_s)
        else:
            self.layernorm_a = LayerNorm(c_a)
            if self.cross_attention_mode:
                self.layernorm_kv = LayerNorm(c_a)

        # Line 6-11
        self.local_attention_method = "local_cross_attention"
        self.attention = Attention(
            c_q=c_a,
            c_k=c_a,
            c_v=c_a,
            c_hidden=c_a // n_heads,
            num_heads=n_heads,
            gating=True,
            q_linear_bias=True,
            local_attention_method=self.local_attention_method,
            zero_init=not self.has_s,  # Adaptive zero init
        )
        self.layernorm_z = LayerNorm(c_z, create_offset=self.create_offset_ln_z)
        # Alg24. Line8 is scalar, but this is different for different heads
        self.linear_nobias_z = LinearNoBias(in_features=c_z, out_features=n_heads)

        # Line 13
        if self.has_s:
            self.linear_a_last = BiasInitLinear(
                in_features=c_s, out_features=c_a, bias=True, biasinit=biasinit
            )

    def local_multihead_attention(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        z: torch.Tensor,
        n_queries: int = 32,
        n_keys: int = 128,
    ) -> torch.Tensor:
        """Algorithm 24 — local (windowed) attention with pair bias. Used inside
        AtomTransformer where the bias `z` arrives already in dense-trunk form.

        Args:
            q:        [..., N_atom, c_a]                                query
            kv:       [..., N_atom, c_a]                                key/value
            z:        [..., n_blocks, n_queries, n_keys, c_z]           atom-pair feature
        Returns:
            [..., N_atom, c_a] updated query
        """
        assert n_queries == z.size(-3)
        assert n_keys == z.size(-2)
        assert len(z.shape) == len(q.shape) + 2

        bias = self.linear_nobias_z(self.layernorm_z(z))
        bias = permute_final_dims(bias, [3, 0, 1, 2])  # [..., H, n_blocks, n_q, n_k]
        return self.attention(
            q_x=q, kv_x=kv, trunked_attn_bias=bias,
            n_queries=n_queries, n_keys=n_keys,
        )

    def standard_multihead_attention(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """Algorithms 7/20 — full attention with pair bias.

        Args:
            q:  [..., N_token, c_a]              query
            kv: [..., N_token, c_a]              key/value
            z:  [..., N_token, N_token, c_z]     pair representation
        Returns:
            [..., N_token, c_a] updated query
        """
        bias = self.linear_nobias_z(self.layernorm_z(z))
        bias = permute_final_dims(bias, [2, 0, 1])  # [..., H, N, N]
        return self.attention(q_x=q, kv_x=kv, attn_bias=bias)

    def forward(
        self,
        a: torch.Tensor,
        s: Optional[torch.Tensor],
        z: torch.Tensor,
        n_queries: Optional[int] = None,
        n_keys: Optional[int] = None,
    ) -> torch.Tensor:
        """AttentionPairBias forward (adaLN-Zero output gate from Peebles & Xie '23).

        Routes to ``local_multihead_attention`` if ``(n_queries, n_keys)`` is
        given, otherwise ``standard_multihead_attention``.
        """
        ##########################################################################
        # TODO: Algorithm 24. Steps:                                              #
        #   1. Normalize the query input ``a`` (adaptive if has_s, plain LN o.w.).#
        #   2. If ``cross_attention_mode``, build ``kv`` from a second LN; else  #
        #      reuse ``a`` for both query and kv.                                #
        #   3. Dispatch to local vs standard multi-head attention based on       #
        #      whether ``n_queries``/``n_keys`` are passed.                      #
        #   4. If has_s, apply the adaLN-Zero output gate                        #
        #      ``sigmoid(linear_a_last(s)) * a``.                                #
        # TODO: Algorithm 24. 步骤:                                              #
        #   1. 归一化 query 输入 ``a`` (has_s 时用 adaptive，否则普通 LN)。       #
        #   2. 若 ``cross_attention_mode``，再过一次 LN 得到 ``kv``；否则        #
        #      query 与 kv 共用 ``a``。                                          #
        #   3. 根据是否给了 ``n_queries`` / ``n_keys`` 分派到 local / standard 。 #
        #   4. has_s 时再叠一次 adaLN-Zero 输出门                                 #
        #      ``sigmoid(linear_a_last(s)) * a``。                               #
        ##########################################################################

        if self.has_s:
            a = self.layernorm_a(a=a, s=s)
        else:
            a = self.layernorm_a(a)

        if self.cross_attention_mode:
            kv = self.layernorm_kv(a=a, s=s) if self.has_s else self.layernorm_kv(a)
        else:
            kv = a

        if n_queries and n_keys:
            a = self.local_multihead_attention(a, kv, z, n_queries, n_keys)
        else:
            a = self.standard_multihead_attention(a, kv, z)

        if self.has_s:
            a = a * torch.sigmoid(self.linear_a_last(s))

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################
        return a

