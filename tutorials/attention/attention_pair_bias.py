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
        ##########################################################################
        # TODO: AttentionPairBias (Algorithm 24) — build the sub-modules.        #
        #                                                                        #
        #   Step 1 — Query-stream LayerNorm. ``has_s=True`` (we're inside        #
        #     DiffusionTransformer, ``s`` is available) -> AdaLN. Otherwise     #
        #     (e.g. PairformerBlock) -> plain LN. In cross-attention mode add   #
        #     a parallel ``layernorm_kv`` for the kv stream:                     #
        #       if has_s:                                                        #
        #           self.layernorm_a = AdaptiveLayerNorm(c_a=c_a, c_s=c_s)       #
        #           if self.cross_attention_mode:                                #
        #               self.layernorm_kv = AdaptiveLayerNorm(c_a=c_a, c_s=c_s) #
        #       else:                                                            #
        #           self.layernorm_a = LayerNorm(c_a)                            #
        #           if self.cross_attention_mode:                                #
        #               self.layernorm_kv = LayerNorm(c_a)                       #
        #                                                                        #
        #   Step 2 — Underlying multi-head attention. ``c_a // n_heads`` is the  #
        #     per-head hidden dim. ``zero_init=True`` only when the adaLN-Zero  #
        #     output gate is missing (``has_s=False`` means the residual must   #
        #     start at zero via ``linear_o``):                                   #
        #       self.local_attention_method = "local_cross_attention"            #
        #       self.attention = Attention(                                      #
        #           c_q=c_a, c_k=c_a, c_v=c_a,                                   #
        #           c_hidden=c_a // n_heads,                                     #
        #           num_heads=n_heads,                                           #
        #           gating=True,                                                 #
        #           q_linear_bias=True,                                          #
        #           local_attention_method=self.local_attention_method,          #
        #           zero_init=not self.has_s,                                    #
        #       )                                                                #
        #                                                                        #
        #   Step 3 — Pair-bias projection: LayerNorm over ``z`` (offset is       #
        #     configurable: Pairformer wants offset=True, DiffusionTransformer  #
        #     leaves it off) + LinearNoBias to ``n_heads``:                      #
        #       self.layernorm_z = LayerNorm(                                    #
        #           c_z, create_offset=self.create_offset_ln_z)                 #
        #       self.linear_nobias_z = LinearNoBias(                            #
        #           in_features=c_z, out_features=n_heads)                      #
        #                                                                        #
        #   Step 4 — adaLN-Zero output gate. Only built when ``has_s``:          #
        #     BiasInitLinear(c_s -> c_a) with bias starting at ``biasinit``     #
        #     (default -2 -> sigmoid(-2)≈0.12, gate closed at init):            #
        #       if self.has_s:                                                   #
        #           self.linear_a_last = BiasInitLinear(                        #
        #               in_features=c_s, out_features=c_a,                      #
        #               bias=True, biasinit=biasinit)                            #
        #                                                                        #
        # TODO: AttentionPairBias (算法 24) —— 构造子模块。                       #
        #                                                                        #
        #   步骤 1 — 查询流 LayerNorm。``has_s=True`` (在 DiffusionTransformer    #
        #     里能拿到 ``s``) -> AdaLN；否则 -> 普通 LN。cross-attention 时       #
        #     另起一支 ``layernorm_kv``:                                          #
        #       if has_s:                                                        #
        #           self.layernorm_a = AdaptiveLayerNorm(c_a=c_a, c_s=c_s)       #
        #           if self.cross_attention_mode:                                #
        #               self.layernorm_kv = AdaptiveLayerNorm(c_a=c_a, c_s=c_s) #
        #       else:                                                            #
        #           self.layernorm_a = LayerNorm(c_a)                            #
        #           if self.cross_attention_mode:                                #
        #               self.layernorm_kv = LayerNorm(c_a)                       #
        #                                                                        #
        #   步骤 2 — 底层多头注意力。``c_a // n_heads`` 是每头隐藏维。             #
        #     ``zero_init=True`` 只在缺 adaLN-Zero 输出门时 (``has_s=False``)   #
        #     使用 —— 让残差从 0 起手:                                            #
        #       self.local_attention_method = "local_cross_attention"            #
        #       self.attention = Attention(                                      #
        #           c_q=c_a, c_k=c_a, c_v=c_a,                                   #
        #           c_hidden=c_a // n_heads,                                     #
        #           num_heads=n_heads,                                           #
        #           gating=True,                                                 #
        #           q_linear_bias=True,                                          #
        #           local_attention_method=self.local_attention_method,          #
        #           zero_init=not self.has_s,                                    #
        #       )                                                                #
        #                                                                        #
        #   步骤 3 — pair bias 投影: 对 ``z`` LayerNorm (offset 由调用方控制) +  #
        #     LinearNoBias 投到 ``n_heads`` 维 (每头一个标量偏置):                 #
        #       self.layernorm_z = LayerNorm(                                    #
        #           c_z, create_offset=self.create_offset_ln_z)                 #
        #       self.linear_nobias_z = LinearNoBias(                            #
        #           in_features=c_z, out_features=n_heads)                      #
        #                                                                        #
        #   步骤 4 — adaLN-Zero 输出门。仅 ``has_s`` 时构建。BiasInitLinear      #
        #     (c_s -> c_a)，bias 初始为 ``biasinit`` (默认 -2):                  #
        #       if self.has_s:                                                   #
        #           self.linear_a_last = BiasInitLinear(                        #
        #               in_features=c_s, out_features=c_a,                      #
        #               bias=True, biasinit=biasinit)                            #
        ##########################################################################

        # Replace "pass" statement with your code
        pass

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

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
        ##########################################################################
        # TODO: Local (windowed) Attention with Pair Bias for AtomTransformer.   #
        #   The bias ``z`` arrives **already in dense-trunk form**:               #
        #   ``[..., n_blocks, n_queries, n_keys, c_z]``.                          #
        #                                                                        #
        #   Step 1 — Sanity check the trunk shape matches the configured        #
        #     window sizes:                                                      #
        #       assert n_queries == z.size(-3)                                   #
        #       assert n_keys    == z.size(-2)                                   #
        #       assert len(z.shape) == len(q.shape) + 2                          #
        #                                                                        #
        #   Step 2 — Project the per-pair bias to per-head logits, then          #
        #     permute the channel (now head) axis ahead of the trunk / query /  #
        #     key axes so the underlying attention can broadcast it cleanly:    #
        #       bias = self.linear_nobias_z(self.layernorm_z(z))                #
        #                                       # [..., n_blocks, n_q, n_k, H]  #
        #       bias = permute_final_dims(bias, [3, 0, 1, 2])                   #
        #                                       # [..., H, n_blocks, n_q, n_k]  #
        #                                                                        #
        #   Step 3 — Call the underlying multi-head attention with the trunked  #
        #     bias path turned on:                                               #
        #       return self.attention(                                           #
        #           q_x=q, kv_x=kv,                                              #
        #           trunked_attn_bias=bias,                                      #
        #           n_queries=n_queries, n_keys=n_keys,                          #
        #       )                                                                #
        #                                                                        #
        # TODO: 局部 (窗口) Attention with Pair Bias，供 AtomTransformer 使用。   #
        #   bias 张量 ``z`` 已是 dense-trunk 形状:                                #
        #   ``[..., n_blocks, n_queries, n_keys, c_z]``。                         #
        #                                                                        #
        #   步骤 1 — trunk 形状与窗口配置一致:                                     #
        #       assert n_queries == z.size(-3)                                   #
        #       assert n_keys    == z.size(-2)                                   #
        #       assert len(z.shape) == len(q.shape) + 2                          #
        #                                                                        #
        #   步骤 2 — 把每 pair 的 bias 投到每头 logits，再把通道(head)轴前移       #
        #     到 trunk / q / k 之前，便于下游 attention 广播:                      #
        #       bias = self.linear_nobias_z(self.layernorm_z(z))                #
        #                                       # [..., n_blocks, n_q, n_k, H]  #
        #       bias = permute_final_dims(bias, [3, 0, 1, 2])                   #
        #                                       # [..., H, n_blocks, n_q, n_k]  #
        #                                                                        #
        #   步骤 3 — 调底层多头注意力，启用 trunked bias 路径:                      #
        #       return self.attention(                                           #
        #           q_x=q, kv_x=kv,                                              #
        #           trunked_attn_bias=bias,                                      #
        #           n_queries=n_queries, n_keys=n_keys,                          #
        #       )                                                                #
        ##########################################################################

        # Replace "pass" statement with your code
        pass

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

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
        ##########################################################################
        # TODO: Standard (full) Attention with Pair Bias. Used by PairformerBlock#
        #   and the token-level DiffusionTransformer; here ``z`` is the regular #
        #   square pair tensor ``[..., N_token, N_token, c_z]``.                 #
        #                                                                        #
        #   Step 1 — LayerNorm + project ``z`` to per-head bias logits:          #
        #       bias = self.linear_nobias_z(self.layernorm_z(z))                #
        #                                       # [..., N, N, H]                #
        #                                                                        #
        #   Step 2 — Permute heads ahead of the (q, k) grid for downstream      #
        #     attention:                                                         #
        #       bias = permute_final_dims(bias, [2, 0, 1])                      #
        #                                       # [..., H, N, N]                #
        #                                                                        #
        #   Step 3 — Run the full-attention path (no n_queries / n_keys):       #
        #       return self.attention(q_x=q, kv_x=kv, attn_bias=bias)           #
        #                                                                        #
        # TODO: 全连接 Attention with Pair Bias，供 PairformerBlock 和             #
        #   token 级 DiffusionTransformer 使用；``z`` 是正方形 pair 张量          #
        #   ``[..., N_token, N_token, c_z]``。                                    #
        #                                                                        #
        #   步骤 1 — LayerNorm + 投到每头偏置 logits:                              #
        #       bias = self.linear_nobias_z(self.layernorm_z(z))                #
        #                                       # [..., N, N, H]                #
        #                                                                        #
        #   步骤 2 — 把 head 维移到 (q, k) 网格之前:                                #
        #       bias = permute_final_dims(bias, [2, 0, 1])                      #
        #                                       # [..., H, N, N]                #
        #                                                                        #
        #   步骤 3 — 走全连接路径 (不传 n_queries / n_keys):                       #
        #       return self.attention(q_x=q, kv_x=kv, attn_bias=bias)           #
        ##########################################################################

        # Replace "pass" statement with your code
        pass

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

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
        # TODO: Algorithm 24 — Attention with Pair Bias (+ optional adaLN-Zero). #
        #                                                                        #
        #   Step 1 — Normalize the query stream ``a``:                           #
        #       if self.has_s:                                                   #
        #           a = self.layernorm_a(a=a, s=s)   # AdaptiveLayerNorm (Alg 26)#
        #       else:                                                           #
        #           a = self.layernorm_a(a)          # plain LayerNorm           #
        #                                                                        #
        #   Step 2 — Build the key/value stream:                                 #
        #       if self.cross_attention_mode:                                    #
        #           # cross-attention: kv comes from a separate LN over ``a``    #
        #           kv = self.layernorm_kv(a=a, s=s) if self.has_s              #
        #                else self.layernorm_kv(a)                              #
        #       else:                                                            #
        #           # self-attention: kv reuses the normalized query             #
        #           kv = a                                                       #
        #                                                                        #
        #   Step 3 — Run attention with pair-derived bias. The helper methods    #
        #     handle the bias projection ``Linear(LayerNorm(z))`` and the head-  #
        #     dim permutation internally:                                        #
        #       if n_queries and n_keys:                                         #
        #           # AtomTransformer path — windowed local attention, ``z`` is  #
        #           # already in dense-trunk form [..., n_blocks, n_q, n_k, c_z] #
        #           a = self.local_multihead_attention(                          #
        #                 a, kv, z, n_queries, n_keys)                           #
        #       else:                                                            #
        #           # full attention — ``z`` is [..., N_token, N_token, c_z]     #
        #           a = self.standard_multihead_attention(a, kv, z)              #
        #                                                                        #
        #   Step 4 — adaLN-Zero output gate (only when ``has_s``):               #
        #       if self.has_s:                                                   #
        #           a = a * torch.sigmoid(self.linear_a_last(s))                 #
        #     ``linear_a_last`` is a BiasInitLinear with bias initialized to     #
        #     ``biasinit = -2.0`` so sigmoid(-2)≈0.12, i.e. the gate is closed   #
        #     at init and the residual branch starts near zero (Peebles & Xie   #
        #     2023, DiT). When ``has_s`` is False the gate is fused into the    #
        #     zero-initialized output projection of ``self.attention``.         #
        #                                                                        #
        # TODO: 算法 24 — 带 pair bias 的注意力 (+ 可选 adaLN-Zero 输出门)。      #
        #                                                                        #
        #   步骤 1 — 归一化查询流 ``a``:                                          #
        #       if self.has_s:                                                   #
        #           a = self.layernorm_a(a=a, s=s)   # AdaptiveLayerNorm(算法 26)#
        #       else:                                                           #
        #           a = self.layernorm_a(a)          # 普通 LayerNorm            #
        #                                                                        #
        #   步骤 2 — 构造 key/value 流:                                           #
        #       if self.cross_attention_mode:                                    #
        #           # 交叉注意力: kv 走另一支 LN                                 #
        #           kv = self.layernorm_kv(a=a, s=s) if self.has_s              #
        #                else self.layernorm_kv(a)                              #
        #       else:                                                            #
        #           # 自注意力: kv 直接复用归一化后的查询                         #
        #           kv = a                                                       #
        #                                                                        #
        #   步骤 3 — 跑带 pair bias 的注意力。bias 投影 ``Linear(LayerNorm(z))`` #
        #     以及 head 维 permute 由下面的辅助方法内部完成:                      #
        #       if n_queries and n_keys:                                         #
        #           # AtomTransformer 路径 — 局部窗口注意力，``z`` 已是             #
        #           # dense-trunk 形状 [..., n_blocks, n_q, n_k, c_z]            #
        #           a = self.local_multihead_attention(                          #
        #                 a, kv, z, n_queries, n_keys)                           #
        #       else:                                                            #
        #           # 全连接注意力 — ``z`` 形状 [..., N_token, N_token, c_z]     #
        #           a = self.standard_multihead_attention(a, kv, z)              #
        #                                                                        #
        #   步骤 4 — adaLN-Zero 输出门 (仅 ``has_s`` 时):                          #
        #       if self.has_s:                                                   #
        #           a = a * torch.sigmoid(self.linear_a_last(s))                 #
        #     ``linear_a_last`` 是 BiasInitLinear，bias 初始化为                  #
        #     ``biasinit = -2.0``，sigmoid(-2)≈0.12，                            #
        #     初始残差分支接近 0 (Peebles & Xie 2023, DiT)。                      #
        #     若 ``has_s`` 为 False，输出门并入 ``self.attention``                #
        #     零初始化的输出投影，无需再乘。                                      #
        ##########################################################################

        # Replace "pass" statement with your code
        pass

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################
        return a

