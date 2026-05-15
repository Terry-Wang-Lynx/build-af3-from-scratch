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

from attention.attention_pair_bias import AttentionPairBias
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



class DiffusionTransformerBlock(nn.Module):
    """
    Implements Algorithm 23[Line2-Line3] in AF3

    Args:
        c_a (int): single embedding dimension.
        c_s (int): single embedding dimension.
        c_z (int): pair embedding dimension.
        n_heads (int): number of heads for DiffusionTransformerBlock.
        biasinit (float, optional): bias initialization value. Defaults to -2.0.
        drop_path_rate (float, optional): drop path rate. Defaults to 0.0.
        cross_attention_mode (bool, optional): whether to use cross attention. Defaults to False.
    """

    def __init__(
        self,
        c_a: int,  # could be 128 or 768 in AF3
        c_s: int,  # could be c_s or c_atom
        c_z: int,  # could be c_z or c_atompair
        n_heads: int,  # could be 16 or 4 or ... in AF3
        biasinit: float = -2.0,
        drop_path_rate: float = 0.0,
        cross_attention_mode: bool = False,
    ) -> None:
        super(DiffusionTransformerBlock, self).__init__()
        self.n_heads = n_heads
        self.c_a = c_a
        self.c_s = c_s
        self.c_z = c_z
        self.attention_pair_bias = AttentionPairBias(
            has_s=True,
            create_offset_ln_z=False,
            n_heads=n_heads,
            c_a=c_a,
            c_s=c_s,
            c_z=c_z,
            biasinit=biasinit,
            cross_attention_mode=cross_attention_mode,
        )
        self.conditioned_transition_block = ConditionedTransitionBlock(
            n=2, c_a=c_a, c_s=c_s, biasinit=biasinit
        )
        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )

    def forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        n_queries: Optional[int] = None,
        n_keys: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Algorithm 23 line 2-3 — a single DiffusionTransformer block.

        Args:
            a: [..., N, c_a]                       atom/token activations
            s: [..., N, c_s]                       single conditioning
            z: [..., N, N, c_z] OR
               [..., n_blocks, n_queries, n_keys, c_z] for local attention

        Returns: (a_out, s, z)  — s/z forwarded so they survive checkpointing.
        """
        ##########################################################################
        # TODO: Algorithm 23 lines 2-3 — one DiffusionTransformer block.         #
        #                                                                        #
        #   Step 1 — Run the AttentionPairBias sub-block. ``has_s=True`` so it  #
        #     uses AdaLN + adaLN-Zero output gate internally; we wrap its       #
        #     output in DropPath (stochastic depth) and add as residual:        #
        #       attn_out = self.drop_path(                                       #
        #           self.attention_pair_bias(                                   #
        #               a=a, s=s, z=z,                                          #
        #               n_queries=n_queries, n_keys=n_keys,                     #
        #           )                                                            #
        #       )                                                                #
        #       attn_out = attn_out + a                                          #
        #                                                                        #
        #   Step 2 — Run the ConditionedTransitionBlock (Algorithm 25) on the   #
        #     post-attention state, again with DropPath and residual:           #
        #       ff_out = self.drop_path(                                         #
        #           self.conditioned_transition_block(a=attn_out, s=s)          #
        #       )                                                                #
        #                                                                        #
        #   Step 3 — Return ``(a, s, z)`` so ``s`` and ``z`` survive activation #
        #     checkpointing (only ``a`` is updated):                             #
        #       return ff_out + attn_out, s, z                                   #
        #                                                                        #
        # TODO: 算法 23 第 2-3 行 —— DiffusionTransformer 的一个 block。           #
        #                                                                        #
        #   步骤 1 — 跑 AttentionPairBias 子块。``has_s=True`` 自动启用 AdaLN +    #
        #     adaLN-Zero 输出门；外面包一层 DropPath 后做残差:                     #
        #       attn_out = self.drop_path(                                       #
        #           self.attention_pair_bias(                                   #
        #               a=a, s=s, z=z,                                          #
        #               n_queries=n_queries, n_keys=n_keys,                     #
        #           )                                                            #
        #       )                                                                #
        #       attn_out = attn_out + a                                          #
        #                                                                        #
        #   步骤 2 — 跑 ConditionedTransitionBlock (算法 25)，                     #
        #     同样 DropPath + 残差:                                                #
        #       ff_out = self.drop_path(                                         #
        #           self.conditioned_transition_block(a=attn_out, s=s)          #
        #       )                                                                #
        #                                                                        #
        #   步骤 3 — 返回 ``(a, s, z)``，方便激活检查点存活 (只 ``a`` 更新):        #
        #       return ff_out + attn_out, s, z                                   #
        ##########################################################################

        attn_out = self.drop_path(
            self.attention_pair_bias(a=a, s=s, z=z, n_queries=n_queries, n_keys=n_keys)
        )
        attn_out = attn_out + a
        ff_out = self.drop_path(self.conditioned_transition_block(a=attn_out, s=s))
        return ff_out + attn_out, s, z

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################



class DiffusionTransformer(nn.Module):
    """
    Implements Algorithm 23 in AF3

    Args:
        c_a (int): single embedding dimension.
        c_s (int): single embedding dimension.
        c_z (int): pair embedding dimension.
        n_blocks (int): number of blocks in DiffusionTransformer.
        n_heads (int): number of heads in attention.
        cross_attention_mode (bool, optional): whether to use cross attention. Defaults to False.
        drop_path_rate (float, optional): drop skip connection path rate. Defaults to 0.0.
        blocks_per_ckpt (int, optional): number of DiffusionTransformer blocks in each activation checkpoint. Defaults to None.
    """

    def __init__(
        self,
        c_a: int,  # could be 128 or 768 in AF3
        c_s: int,  # could be c_s or c_atom
        c_z: int,  # could be c_z or c_atompair
        n_blocks: int,  # could be 3 or 24 in AF3
        n_heads: int,  # could be 16 or 4 or ... in AF3
        cross_attention_mode: bool = False,
        drop_path_rate: float = 0.0,  # drop skip connection path
        blocks_per_ckpt: Optional[int] = None,
    ) -> None:
        super(DiffusionTransformer, self).__init__()
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.c_a = c_a
        self.c_s = c_s
        self.c_z = c_z
        self.blocks_per_ckpt = blocks_per_ckpt

        self.blocks = nn.ModuleList()
        drop_path_rates = [
            drop_path_value.item()
            for drop_path_value in torch.linspace(0, drop_path_rate, n_blocks)
        ]
        for i in range(n_blocks):
            block = DiffusionTransformerBlock(
                n_heads=n_heads,
                c_a=c_a,
                c_s=c_s,
                c_z=c_z,
                cross_attention_mode=cross_attention_mode,
                drop_path_rate=drop_path_rates[i],
            )
            self.blocks.append(block)

    def forward(
        self,
        a: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        n_queries: Optional[int] = None,
        n_keys: Optional[int] = None,
    ) -> torch.Tensor:
        """Algorithm 23 — stacked DiffusionTransformer blocks.

        Args:
            a: [..., N, c_a]
            s: [..., N, c_s] conditioning
            z: pair representation; full or local-trunked depending on n_queries/n_keys
        Returns:
            [..., N, c_a]
        """
        ##########################################################################
        # TODO: Algorithm 23 — stacked DiffusionTransformer blocks. Forwards    #
        #   ``s`` / ``z`` unchanged between blocks so they survive activation   #
        #   checkpointing; only ``a`` accumulates updates:                       #
        #       for block in self.blocks:                                        #
        #           a, s, z = block(                                             #
        #               a, s, z,                                                 #
        #               n_queries=n_queries, n_keys=n_keys,                     #
        #           )                                                            #
        #       return a                                                         #
        #                                                                        #
        # TODO: 算法 23 —— 堆叠 DiffusionTransformer blocks。``s`` / ``z`` 在     #
        #   block 之间原样透传 (为方便激活检查点)，只更新 ``a``:                   #
        #       for block in self.blocks:                                        #
        #           a, s, z = block(                                             #
        #               a, s, z,                                                 #
        #               n_queries=n_queries, n_keys=n_keys,                     #
        #           )                                                            #
        #       return a                                                         #
        ##########################################################################

        for block in self.blocks:
            a, s, z = block(a, s, z, n_queries=n_queries, n_keys=n_keys)
        return a

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################



class ConditionedTransitionBlock(nn.Module):
    """
    Implements Algorithm 25 in AF3

    Args:
        c_a (int): single embedding dim (single feature aggregated atom info).
        c_s (int):  single embedding dim.
        n (int, optional): channel scale factor. Defaults to 2.
        biasinit (float, optional): bias initialization value. Defaults to -2.0.
    """

    def __init__(self, c_a: int, c_s: int, n: int = 2, biasinit: float = -2.0) -> None:
        super(ConditionedTransitionBlock, self).__init__()
        self.c_a = c_a
        self.c_s = c_s
        self.n = n
        self.adaln = AdaptiveLayerNorm(c_a=c_a, c_s=c_s)
        self.linear_nobias_a1 = LinearNoBias(
            in_features=c_a, out_features=n * c_a, initializer="relu"
        )
        self.linear_nobias_a2 = LinearNoBias(
            in_features=c_a, out_features=n * c_a, initializer="relu"
        )
        self.linear_nobias_b = LinearNoBias(in_features=n * c_a, out_features=c_a)
        self.linear_s = BiasInitLinear(
            in_features=c_s, out_features=c_a, bias=True, biasinit=biasinit
        )

    def forward(self, a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        Args:
            a (torch.Tensor): the single feature aggregate per-atom representation
                [..., N, c_a]
            s (torch.Tensor): single embedding
                [..., N, c_s]

        Returns:
            torch.Tensor: the updated a from ConditionedTransitionBlock
                [..., N, c_a]
        """
        ##########################################################################
        # TODO: Algorithm 25 — ConditionedTransitionBlock (SwiGLU FFN with     #
        #   adaLN-Zero output gate).                                            #
        #                                                                        #
        #   Step 1 — Adaptive LayerNorm modulates ``a`` by ``s`` (Alg 26):       #
        #       a = self.adaln(a, s)                # [..., N, c_a]              #
        #                                                                        #
        #   Step 2 — SwiGLU gate-value FFN: two parallel widening linears       #
        #     (``relu`` init, no bias) feed a silu * b gate, output stays at    #
        #     ``n * c_a`` width:                                                 #
        #       b = F.silu(self.linear_nobias_a1(a)) * self.linear_nobias_a2(a)  #
        #                                       # [..., N, n*c_a]                #
        #                                                                        #
        #   Step 3 — adaLN-Zero output gate (from Peebles & Xie 2023):          #
        #     a sigmoid gate driven by ``s`` (``linear_s`` is BiasInitLinear    #
        #     with bias=-2 -> gate ≈ 0.12 at init), multiplied with the         #
        #     projected FFN output:                                              #
        #       a = torch.sigmoid(self.linear_s(s)) * self.linear_nobias_b(b)    #
        #                                       # [..., N, c_a]                  #
        #   Return ``a``.                                                        #
        #                                                                        #
        # TODO: 算法 25 —— ConditionedTransitionBlock (带 adaLN-Zero 输出门的     #
        #   SwiGLU FFN)。                                                        #
        #                                                                        #
        #   步骤 1 — Adaptive LayerNorm 用 ``s`` 调制 ``a`` (算法 26):             #
        #       a = self.adaln(a, s)                # [..., N, c_a]              #
        #                                                                        #
        #   步骤 2 — SwiGLU gate-value FFN: 两路 relu-init 扩宽线性层 (无 bias)、 #
        #     silu * b 门，宽度 ``n * c_a``:                                       #
        #       b = F.silu(self.linear_nobias_a1(a)) * self.linear_nobias_a2(a)  #
        #                                       # [..., N, n*c_a]                #
        #                                                                        #
        #   步骤 3 — adaLN-Zero 输出门 (Peebles & Xie 2023)：由 ``s`` 驱动的       #
        #     sigmoid 门 (``linear_s`` 是 BiasInitLinear，bias=-2，初始 ≈0.12)   #
        #     乘以投回 c_a 的 FFN 输出:                                            #
        #       a = torch.sigmoid(self.linear_s(s)) * self.linear_nobias_b(b)    #
        #                                       # [..., N, c_a]                  #
        #   返回 ``a``。                                                          #
        ##########################################################################

        a = self.adaln(a, s)
        b = F.silu((self.linear_nobias_a1(a))) * self.linear_nobias_a2(a)
        # Output projection (from adaLN-Zero [27])
        a = torch.sigmoid(self.linear_s(s)) * self.linear_nobias_b(b)
        return a

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

