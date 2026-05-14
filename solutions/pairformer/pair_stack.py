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

# pylint: disable=C0114
from functools import partial
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from feature_extraction.constants import STD_RESIDUES_WITH_GAP
from attention.linear import LinearNoBias
from attention.transition import Transition
from attention.attention_pair_bias import AttentionPairBias
from pairformer.dropout import dropout_add_rowwise
from pairformer.triangle_ops import DropoutRowwise, LayerNorm, OuterProductMean
from pairformer.triangle import (
    TriangleAttention,
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)
from model.utils import (
    checkpoint_blocks,
    expand_at_dim,
    get_checkpoint_fn,
    pad_at_dim,
    sample_msa_feature_dict_random_without_replacement,
)



class PairformerBlock(nn.Module):
    """One Pairformer block — AF3 Algorithm 17, lines 2-8.

    Pairformer 单个 block —— AF3 Algorithm 17 第 2-8 行。

    Updates the pair representation ``z`` via:
    依次用以下四步更新 pair 表示 ``z``：

        1. TriangleMultiplicationOutgoing  (Alg 12)
        2. TriangleMultiplicationIncoming  (Alg 13)
        3. TriangleAttentionStartingNode + TriangleAttentionEndingNode  (Alg 13/14)
        4. Pair transition (FFN)

    If ``c_s > 0`` it additionally updates the single representation ``s``
    via an AttentionPairBias (using ``z`` as bias) and a Transition.
    若 ``c_s > 0`` 还会再走一次 AttentionPairBias (用 ``z`` 作为偏置) +
    Transition 来更新 single 表示 ``s``。

    Args / 参数:
        n_heads:         number of attention heads in AttentionPairBias / 单注意力的头数.
        c_z:             pair representation hidden size / pair 隐藏维度.
        c_s:             single representation hidden size / single 隐藏维度.
        c_hidden_mul:    hidden size inside TriangleMultiplicative / 三角形乘法的隐藏维度.
        c_hidden_pair_att: hidden size inside TriangleAttention / 三角形注意力的隐藏维度.
        no_heads_pair:   number of heads inside TriangleAttention / 三角形注意力的头数.
        num_intermediate_factor: FFN expansion factor / 前馈层扩展因子.
        dropout:         (training-only) dropout rate / 训练时的 dropout 概率.
        hidden_scale_up: if True, scale internal hidden dims with c_z /
                         若为 True，内部隐藏维度随 c_z 一起放大.
    """

    def __init__(
        self,
        n_heads: int = 16,
        c_z: int = 128,
        c_s: int = 384,
        c_hidden_mul: int = 128,
        c_hidden_pair_att: int = 32,
        no_heads_pair: int = 4,
        num_intermediate_factor: int = 4,
        dropout: float = 0.25,
        hidden_scale_up: bool = False,
    ) -> None:
        super(PairformerBlock, self).__init__()
        self.n_heads = n_heads
        if hidden_scale_up:
            no_heads_pair = c_z // c_hidden_pair_att
            c_hidden_mul = c_z
        self.tri_mul_out = TriangleMultiplicationOutgoing(
            c_z=c_z, c_hidden=c_hidden_mul
        )
        self.tri_mul_in = TriangleMultiplicationIncoming(c_z=c_z, c_hidden=c_hidden_mul)
        self.tri_att_start = TriangleAttention(
            c_in=c_z,
            c_hidden=c_hidden_pair_att,
            no_heads=no_heads_pair,
        )
        self.tri_att_end = TriangleAttention(
            c_in=c_z,
            c_hidden=c_hidden_pair_att,
            no_heads=no_heads_pair,
        )
        self.dropout_row = DropoutRowwise(dropout)
        self.p_drop = dropout
        self.pair_transition = Transition(c_in=c_z, n=num_intermediate_factor)
        self.c_s = c_s
        if self.c_s > 0:
            self.attention_pair_bias = AttentionPairBias(
                has_s=False, create_offset_ln_z=True, n_heads=n_heads, c_a=c_s, c_z=c_z
            )
            self.single_transition = Transition(c_in=c_s, n=4)

    def forward(
        self,
        s: Optional[torch.Tensor],
        z: torch.Tensor,
        pair_mask: Optional[torch.Tensor] = None,
    ) -> tuple[Optional[torch.Tensor], torch.Tensor]:
        """One Pairformer block update.

        Args:
            s:         [..., N_token, c_s]              single feature (used iff c_s > 0)
            z:         [..., N_token, N_token, c_z]     pair representation
            pair_mask: [..., N_token, N_token]          optional pair mask

        Returns:
            (s_updated, z_updated) — same shapes as inputs. ``s_updated`` is
            ``None`` if this block has no single-track (``c_s == 0``).
        """
        ##########################################################################
        # TODO: Algorithm 17 lines 2-8. Update z with four sub-blocks, then     #
        #   (iff c_s > 0) update s with one AttentionPairBias + Transition.      #
        #     1. z += TriangleMultiplicationOutgoing(z)                          #
        #     2. z += TriangleMultiplicationIncoming(z)                          #
        #     3. z += TriangleAttentionStartingNode(z)                           #
        #     4. z += TriangleAttentionEndingNode(z) (transposed twice)          #
        #     5. z += pair_transition(z)                                         #
        #     6. (c_s>0) s += AttentionPairBias(s, z)                            #
        #     7. (c_s>0) s += single_transition(s)                               #
        # TODO: Algorithm 17 第 2-8 行。先用四步更新 z，再 (c_s>0 时) 用一次     #
        #   AttentionPairBias + Transition 更新 s。                              #
        ##########################################################################

        z = self.tri_mul_out(z, mask=pair_mask, inplace_safe=True, _add_with_inplace=True)
        z = self.tri_mul_in(z, mask=pair_mask, inplace_safe=True, _add_with_inplace=True)
        z += self.tri_att_start(z, mask=pair_mask, inplace_safe=True)
        z = z.transpose(-2, -3).contiguous()
        z += self.tri_att_end(
            z,
            mask=pair_mask.transpose(-1, -2) if pair_mask is not None else None,
            inplace_safe=True,
        )
        z = z.transpose(-2, -3).contiguous()
        z += self.pair_transition(z)

        if self.c_s > 0:
            s = s + self.attention_pair_bias(a=s, s=None, z=z)
            s = s + self.single_transition(s)

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################
        return s, z



class PairformerStack(nn.Module):
    """A stack of ``n_blocks`` PairformerBlocks — AF3 Algorithm 17.

    由 ``n_blocks`` 个 PairformerBlock 串联而成 —— AF3 Algorithm 17。

    Defaults to 48 blocks for the Base model; Tiny/Mini use 8/16 blocks
    (set via ``configs/configs_model_type.py``).
    Base 模型默认 48 块，Tiny / Mini 分别用 8 / 16 块 (由
    ``configs/configs_model_type.py`` 配置)。

    Args / 参数:
        n_blocks: how many PairformerBlocks to stack / block 数量.
        其它参数含义见 PairformerBlock / see PairformerBlock above.
    """

    def __init__(
        self,
        n_blocks: int = 48,
        n_heads: int = 16,
        c_z: int = 128,
        c_s: int = 384,
        num_intermediate_factor: int = 4,
        dropout: float = 0.25,
        blocks_per_ckpt: Optional[int] = None,
        hidden_scale_up: bool = False,
    ) -> None:
        super(PairformerStack, self).__init__()
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.blocks_per_ckpt = blocks_per_ckpt
        self.blocks = nn.ModuleList()

        for _ in range(n_blocks):
            block = PairformerBlock(
                n_heads=n_heads,
                c_z=c_z,
                c_s=c_s,
                num_intermediate_factor=num_intermediate_factor,
                dropout=dropout,
                hidden_scale_up=hidden_scale_up,
            )
            self.blocks.append(block)

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        pair_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sequentially apply n_blocks PairformerBlocks."""
        for block in self.blocks:
            s, z = block(s, z, pair_mask=pair_mask)
        return s, z

