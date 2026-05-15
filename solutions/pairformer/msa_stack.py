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
from pairformer.pair_stack import PairformerBlock, PairformerStack  # noqa: F401
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



class MSAPairWeightedAveraging(nn.Module):
    """
    Implements Algorithm 10 [MSAPairWeightedAveraging] in AF3

    Args:
        c_m (int, optional): hidden dim [for msa embedding]. Defaults to 64.
        c (int, optional): hidden dim [for MSAPairWeightedAveraging]. Defaults to 32.
        c_z (int, optional): hidden dim [for pair embedding]. Defaults to 128.
        n_heads (int, optional): number of heads [for MSAPairWeightedAveraging]. Defaults to 8.
    """

    def __init__(
        self, c_m: int = 64, c: int = 32, c_z: int = 128, n_heads: int = 8
    ) -> None:
        super(MSAPairWeightedAveraging, self).__init__()
        self.c_m = c_m
        self.c = c
        self.n_heads = n_heads
        self.c_z = c_z
        ##########################################################################
        # TODO: MSAPairWeightedAveraging (Algorithm 10 inside MSAModule). Set up #
        #   the input projections, the softmax used to turn pair logits into     #
        #   attention weights, and the output projection.                        #
        #                                                                        #
        #   - MSA-side LayerNorm + value projection (multi-head):                #
        #       self.layernorm_m = LayerNorm(self.c_m)                          #
        #       self.linear_no_bias_mv = LinearNoBias(                          #
        #           in_features=self.c_m,                                       #
        #           out_features=self.c * self.n_heads,                          #
        #       )                                                                #
        #                                                                        #
        #   - Pair-side LayerNorm + per-head logits (c_z -> n_heads):            #
        #       self.layernorm_z = LayerNorm(self.c_z)                          #
        #       self.linear_no_bias_z = LinearNoBias(                           #
        #           in_features=self.c_z, out_features=self.n_heads,             #
        #       )                                                                #
        #                                                                        #
        #   - MSA-side gate projection (c_m -> c*n_heads), zero-init so the     #
        #     output residual starts closed (sigmoid(0)=0.5):                    #
        #       self.linear_no_bias_mg = LinearNoBias(                          #
        #           in_features=self.c_m,                                       #
        #           out_features=self.c * self.n_heads,                          #
        #           initializer="zeros",                                         #
        #       )                                                                #
        #                                                                        #
        #   - Softmax along the "second residue" axis of the pair grid (dim=-2):#
        #       self.softmax_w = nn.Softmax(dim=-2)                              #
        #                                                                        #
        #   - Output projection back to c_m, zero-init so the block starts as a #
        #     no-op:                                                              #
        #       self.linear_no_bias_out = LinearNoBias(                          #
        #           in_features=self.c * self.n_heads,                           #
        #           out_features=self.c_m,                                       #
        #           initializer="zeros",                                         #
        #       )                                                                #
        #                                                                        #
        # TODO: MSAPairWeightedAveraging (MSAModule 内的算法 10)。                #
        #   构造输入投影、把 pair logits 变成权重的 softmax、以及输出投影。       #
        #                                                                        #
        #   - MSA 侧 LayerNorm + 多头 value 投影:                                  #
        #       self.layernorm_m = LayerNorm(self.c_m)                          #
        #       self.linear_no_bias_mv = LinearNoBias(                          #
        #           in_features=self.c_m,                                       #
        #           out_features=self.c * self.n_heads,                          #
        #       )                                                                #
        #                                                                        #
        #   - pair 侧 LayerNorm + 每头 logits (c_z -> n_heads):                   #
        #       self.layernorm_z = LayerNorm(self.c_z)                          #
        #       self.linear_no_bias_z = LinearNoBias(                           #
        #           in_features=self.c_z, out_features=self.n_heads,             #
        #       )                                                                #
        #                                                                        #
        #   - MSA 侧门控投影 (c_m -> c*n_heads)，零初始化使输出门初闭:              #
        #       self.linear_no_bias_mg = LinearNoBias(                          #
        #           in_features=self.c_m,                                       #
        #           out_features=self.c * self.n_heads,                          #
        #           initializer="zeros",                                         #
        #       )                                                                #
        #                                                                        #
        #   - 沿 pair 网格的"第二条 residue 轴" (dim=-2) 做 softmax:                #
        #       self.softmax_w = nn.Softmax(dim=-2)                              #
        #                                                                        #
        #   - 输出投影回 c_m，零初始化使块从恒等开始:                                #
        #       self.linear_no_bias_out = LinearNoBias(                          #
        #           in_features=self.c * self.n_heads,                           #
        #           out_features=self.c_m,                                       #
        #           initializer="zeros",                                         #
        #       )                                                                #
        ##########################################################################

        # Input projections
        self.layernorm_m = LayerNorm(self.c_m)
        self.linear_no_bias_mv = LinearNoBias(
            in_features=self.c_m, out_features=self.c * self.n_heads
        )
        self.layernorm_z = LayerNorm(self.c_z)
        self.linear_no_bias_z = LinearNoBias(
            in_features=self.c_z, out_features=self.n_heads
        )
        self.linear_no_bias_mg = LinearNoBias(
            in_features=self.c_m,
            out_features=self.c * self.n_heads,
            initializer="zeros",
        )
        # Weighted average with gating
        self.softmax_w = nn.Softmax(dim=-2)
        # Output projection
        self.linear_no_bias_out = LinearNoBias(
            in_features=self.c * self.n_heads,
            out_features=self.c_m,
            initializer="zeros",
        )

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

    def forward(self, m: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            m: [..., N_msa, N_token, c_m] MSA embedding.
            z: [..., N_token, N_token, c_z] pair embedding.

        Returns:
            [..., N_msa, N_token, c_m] updated MSA embedding.
        """
        ##########################################################################
        # TODO: Algorithm 10 — MSA Pair-Weighted Averaging.                      #
        #   Send the pair representation back into the MSA: each pair (i, j)    #
        #   contributes a per-head weight ``w_ij^h`` that is used to average     #
        #   value vectors across the residue (``j``) axis of the MSA.            #
        #                                                                        #
        #   Step 1 — Pre-LayerNorm and value projection from MSA:                #
        #       m = self.layernorm_m(m)                  # [*, N_msa, N_tok, c_m]#
        #       v = self.linear_no_bias_mv(m).reshape(                           #
        #               *m.shape[:-1], self.n_heads, self.c                      #
        #           )                                    # [*, N_msa, N_tok, H, c]#
        #                                                                        #
        #   Step 2 — Per-head logits from the pair representation:               #
        #       b = self.linear_no_bias_z(self.layernorm_z(z))                   #
        #                                                # [*, N_tok, N_tok, H]  #
        #     ``linear_no_bias_z`` is LinearNoBias(c_z -> H).                    #
        #                                                                        #
        #   Step 3 — Per-position output gate from MSA (zero-init, opens later): #
        #       g = torch.sigmoid(self.linear_no_bias_mg(m)).reshape(            #
        #               *m.shape[:-1], self.n_heads, self.c                      #
        #           )                                    # [*, N_msa, N_tok, H, c]#
        #                                                                        #
        #   Step 4 — Softmax across the second residue axis of the pair grid:    #
        #       w = self.softmax_w(b)                    # softmax over dim=-2   #
        #     This gives w_ij^h ∝ exp(b_ij^h), normalized over j for each (i,h). #
        #                                                                        #
        #   Step 5 — Weighted average of MSA value vectors along the ``j`` axis: #
        #       wv = torch.einsum('...ijh,...mjhc->...mihc', w, v)               #
        #     i.e. ``wv[*, m, i, h, c] = sum_j w[*, i, j, h] * v[*, m, j, h, c]``#
        #                                                                        #
        #   Step 6 — Apply the gate, flatten the head dim, project back to c_m:  #
        #       o = (g * wv).reshape(*g.shape[:-2], self.n_heads * self.c)       #
        #       m = self.linear_no_bias_out(o)            # [*, N_msa, N_tok, c_m]#
        #   Return ``m`` (zero-init projection -> residual starts as identity). #
        #                                                                        #
        # TODO: 算法 10 —— MSA Pair-Weighted Averaging。                          #
        #   把 pair 表示送回 MSA: 每个 pair (i, j) 给出一个每头权重 w_ij^h，      #
        #   用它沿 MSA 的 residue (j) 轴对 value 加权平均。                       #
        #                                                                        #
        #   步骤 1 — Pre-LayerNorm 并从 MSA 投影出 value:                          #
        #       m = self.layernorm_m(m)                  # [*, N_msa, N_tok, c_m]#
        #       v = self.linear_no_bias_mv(m).reshape(                           #
        #               *m.shape[:-1], self.n_heads, self.c                      #
        #           )                                    # [*, N_msa, N_tok, H, c]#
        #                                                                        #
        #   步骤 2 — 从 pair 表示得到每头 logits:                                   #
        #       b = self.linear_no_bias_z(self.layernorm_z(z))                   #
        #                                                # [*, N_tok, N_tok, H]  #
        #     ``linear_no_bias_z`` 是 LinearNoBias(c_z -> H)。                    #
        #                                                                        #
        #   步骤 3 — 从 MSA 得到每位点输出门控 (零初始化，逐步学开):                #
        #       g = torch.sigmoid(self.linear_no_bias_mg(m)).reshape(            #
        #               *m.shape[:-1], self.n_heads, self.c                      #
        #           )                                    # [*, N_msa, N_tok, H, c]#
        #                                                                        #
        #   步骤 4 — 沿 pair 网格的“第二条 residue 轴”做 softmax:                  #
        #       w = self.softmax_w(b)                    # softmax(dim=-2)       #
        #     即 w_ij^h ∝ exp(b_ij^h)，对每个 (i, h) 沿 j 归一。                  #
        #                                                                        #
        #   步骤 5 — 沿 j 轴对 value 做加权平均:                                    #
        #       wv = torch.einsum('...ijh,...mjhc->...mihc', w, v)               #
        #     即 ``wv[*, m, i, h, c] = sum_j w[*, i, j, h] * v[*, m, j, h, c]``  #
        #                                                                        #
        #   步骤 6 — 应用门控、扁平化头维、投回 c_m:                                #
        #       o = (g * wv).reshape(*g.shape[:-2], self.n_heads * self.c)       #
        #       m = self.linear_no_bias_out(o)            # [*, N_msa, N_tok, c_m]#
        #   返回 ``m`` (零初始化的输出投影 -> 残差从恒等开始)。                    #
        ##########################################################################

        m = self.layernorm_m(m)
        v = self.linear_no_bias_mv(m).reshape(*m.shape[:-1], self.n_heads, self.c)
        b = self.linear_no_bias_z(self.layernorm_z(z))
        g = torch.sigmoid(self.linear_no_bias_mg(m)).reshape(*m.shape[:-1], self.n_heads, self.c)
        w = self.softmax_w(b)
        wv = torch.einsum("...ijh,...mjhc->...mihc", w, v)
        o = (g * wv).reshape(*g.shape[:-2], self.n_heads * self.c)
        m = self.linear_no_bias_out(o)

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################
        return m



class MSAStack(nn.Module):
    """
    Implements MSAStack Line7-Line8 in Algorithm 8

    Args:
        c_m (int, optional): hidden dim [for msa embedding]. Defaults to 64.
        c_z (int, optional): hidden dim [for pair embedding]. Defaults to 128.
        c (int, optional): hidden [for MSAStack] dim. Defaults to 8.
        dropout (float, optional): dropout ratio. Defaults to 0.15.
        msa_chunk_size (int, optional): chunk size for msa. Defaults to 2048.
        msa_max_size (int, optional): max size for msa. Defaults to 16384.
    """

    def __init__(
        self,
        c_m: int = 64,
        c_z: int = 128,
        c: int = 8,
        dropout: float = 0.15,
        msa_chunk_size: Optional[int] = 2048,
        msa_max_size: Optional[int] = 16384,
    ) -> None:
        super(MSAStack, self).__init__()
        self.c = c
        self.msa_pair_weighted_averaging = MSAPairWeightedAveraging(
            c_m=c_m, c=self.c, c_z=c_z
        )
        self.dropout_row = DropoutRowwise(dropout)
        self.p_drop = dropout
        self.transition_m = Transition(c_in=c_m, n=4)
        self.msa_chunk_size = msa_chunk_size
        self.msa_max_size = msa_max_size

    def forward(self, m: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            m (torch.Tensor): msa embedding
                [...,n_msa_sampled, n_token, c_m]
            z (torch.Tensor): pair embedding
                [...,n_token, n_token, c_z]

        Returns:
            torch.Tensor: updated msa embedding
                [...,n_msa_sampled, n_token, c_m]
        """
        chunk_size = self.msa_chunk_size
        if self.training:
            # Padded m to avoid static graph change in DDP training, which will raise
            # RuntimeError: Your training graph has changed in this iteration,
            # e.g., one parameter is unused in first iteration, but then got used in the second iteration.
            # this is not compatible with static_graph set to True
            m_new = pad_at_dim(
                m, dim=-3, pad_length=(0, self.msa_max_size - m.shape[-3]), value=0
            )
            msa_pair_weighted = self.chunk_forward(
                self.msa_pair_weighted_averaging, m_new, z, chunk_size
            )
            m = dropout_add_rowwise(m, msa_pair_weighted[: m.shape[-3], :, :], self.p_drop, self.training)
            m_new = pad_at_dim(
                m, dim=-3, pad_length=(0, self.msa_max_size - m.shape[-3]), value=0
            )
            m_transition = self.chunk_forward(
                self.transition_m, m_new, None, chunk_size
            )
            m = m + m_transition[: m.shape[-3], :, :]
            if (not self.training) and (z.shape[-2] > 2000 or m.shape[-3] > 5120):
                del msa_pair_weighted, m_transition
        else:
            m = self.inference_forward(m, z, chunk_size)
        return m

    def chunk_forward(
        self,
        module: nn.Module,
        m: torch.Tensor,
        z: torch.Tensor,
        chunk_size: int = 2048,
    ) -> torch.Tensor:
        """
        Args:
            m (torch.Tensor): msa embedding
                [..., n_msa_sampled, n_token, c_m]
            z (torch.Tensor): pair embedding
                [..., n_token, n_token, c_z]
            chunk_size (int): size of each chunk for gradient checkpointing

        Returns:
            torch.Tensor: updated msa embedding
                [..., n_msa_sampled, n_token, c_m]
        """

        def fixed_length_chunk(m, chunk_length, dim=0):
            dim_size = m.size(dim)
            chunk_num = (dim_size + chunk_length - 1) // chunk_length
            chunks = []

            for i in range(chunk_num):
                start = i * chunk_length
                end = min(start + chunk_length, dim_size)
                chunk = m.narrow(dim, start, end - start)
                chunks.append(chunk)

            return chunks

        checkpoint_fn = get_checkpoint_fn()
        # Split the tensor `m` into chunks along the first dimension
        # m_chunks = torch.chunk(m, chunk_size, dim=0)
        m_chunks = fixed_length_chunk(m, chunk_size, dim=0)

        # Process each chunk with gradient checkpointing
        if z is not None:
            processed_chunks = [checkpoint_fn(module, chunk, z) for chunk in m_chunks]
        else:
            processed_chunks = [checkpoint_fn(module, chunk) for chunk in m_chunks]
        if (not self.training) and m.shape[-3] > 5120:
            del m_chunks
        # Concatenate the processed chunks back together
        m = torch.cat(processed_chunks, dim=0)
        if (not self.training) and m.shape[-3] > 5120:
            del processed_chunks
        return m

    def inference_forward(
        self, m: torch.Tensor, z: torch.Tensor, chunk_size: int = 2048
    ) -> torch.Tensor:
        """Inplace slice forward for saving memory
        Args:
            m (torch.Tensor): msa embedding
                [..., n_msa_sampled, n_token, c_m]
            z (torch.Tensor): pair embedding
                [..., n_token, n_token, c_z]
            chunk_num (int): size of each chunk for gradient checkpointing

        Returns:
            torch.Tensor: updated msa embedding
                [..., n_msa_sampled, n_token, c_m]
        """
        num_msa = m.shape[-3]
        no_chunks = num_msa // chunk_size + (num_msa % chunk_size != 0)
        for i in range(no_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, num_msa)
            # Use inplace to save memory
            m[start:end, :, :] += self.msa_pair_weighted_averaging(
                m[start:end, :, :], z
            )
            m[start:end, :, :] += self.transition_m(m[start:end, :, :])
        return m



class MSABlock(nn.Module):
    """
    Base MSA Block, Line6-Line13 in Algorithm 8

    Args:
        c_m (int, optional): hidden dim [for msa embedding]. Defaults to 64.
        c_z (int, optional): hidden dim [for pair embedding]. Defaults to 128.
        c_hidden (int, optional): hidden dim [for MSABlock]. Defaults to 32.
        is_last_block (bool, optional): if this is the last block of MSAModule. Defaults to False.
        msa_dropout (float, optional): dropout ratio for msa block. Defaults to 0.15.
        pair_dropout (float, optional): dropout ratio for pair stack. Defaults to 0.25.
        msa_chunk_size (int, optional): chunk size for msa. Defaults to 2048.
        msa_max_size (int, optional): max size for msa. Defaults to 16384.
        hidden_scale_up (bool, optional): whether scale up the hidden if c_z scales. Defaults to False.
    """

    def __init__(
        self,
        c_m: int = 64,
        c_z: int = 128,
        c_hidden: int = 32,
        is_last_block: bool = False,
        msa_dropout: float = 0.15,
        pair_dropout: float = 0.25,
        msa_chunk_size: Optional[int] = 2048,
        msa_max_size: Optional[int] = 16384,
        hidden_scale_up: bool = False,
    ) -> None:
        super(MSABlock, self).__init__()
        self.c_m = c_m
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.is_last_block = is_last_block
        # Communication
        self.outer_product_mean_msa = OuterProductMean(
            c_m=self.c_m, c_z=self.c_z, c_hidden=self.c_hidden
        )
        if not self.is_last_block:
            # MSA stack
            self.msa_stack = MSAStack(
                c_m=self.c_m,
                c_z=self.c_z,
                dropout=msa_dropout,
                msa_chunk_size=msa_chunk_size,
                msa_max_size=msa_max_size,
            )
        # Pair stack
        self.pair_stack = PairformerBlock(
            c_z=c_z, c_s=0, dropout=pair_dropout, hidden_scale_up=hidden_scale_up
        )

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        pair_mask: Optional[torch.Tensor] = None,
    ) -> tuple[Optional[torch.Tensor], torch.Tensor]:
        """One MSA block: MSA→pair communication + intra-MSA + pair stack."""
        z = z + self.outer_product_mean_msa(m, inplace_safe=True)
        if not self.is_last_block:
            m = self.msa_stack(m, z)
        _, z = self.pair_stack(s=None, z=z, pair_mask=pair_mask)
        return (m if not self.is_last_block else None), z



class MSAModule(nn.Module):
    """
    Implements Algorithm 8 [MSAModule] in AF3

    Args:
        n_blocks (int, optional): number of blocks [for MSAModule]. Defaults to 4.
        c_m (int, optional): hidden dim [for msa embedding]. Defaults to 64.
        c_z (int, optional): hidden dim [for pair embedding]. Defaults to 128.
        c_s_inputs (int, optional):
            hidden dim for single embedding from InputFeatureEmbedder. Defaults to 449.
        msa_dropout (float, optional): dropout ratio for msa block. Defaults to 0.15.
        pair_dropout (float, optional): dropout ratio for pair stack. Defaults to 0.25.
        blocks_per_ckpt: number of MSAModule blocks in each activation checkpoint. Defaults to 1.
        msa_chunk_size (int, optional): chunk size for msa. Defaults to 2048.
        msa_max_size (int, optional): max size for msa. Defaults to 16384.
        msa_configs (dict, optional): a dictionary containing keys: "enable", "strategy", etc. Defaults to None.
        hidden_scale_up (bool, optional): whether scale up the hidden if c_z scales. Defaults to False.
    """

    def __init__(
        self,
        n_blocks: int = 4,
        c_m: int = 64,
        c_z: int = 128,
        c_s_inputs: int = 449,
        msa_dropout: float = 0.15,
        pair_dropout: float = 0.25,
        blocks_per_ckpt: Optional[int] = 1,
        msa_chunk_size: Optional[int] = 2048,
        msa_max_size: Optional[int] = 16384,
        msa_configs: Optional[dict[str, Any]] = None,
        hidden_scale_up: bool = False,
    ) -> None:
        super(MSAModule, self).__init__()
        self.n_blocks = n_blocks
        self.c_m = c_m
        self.c_s_inputs = c_s_inputs
        self.blocks_per_ckpt = blocks_per_ckpt
        self.msa_chunk_size = msa_chunk_size
        self.msa_max_size = msa_max_size
        self.input_feature = {
            "msa": 32,
            "has_deletion": 1,
            "deletion_value": 1,
        }

        self.msa_configs = {
            "enable": msa_configs.get("enable", False),
            "strategy": msa_configs.get("strategy", "random"),
        }
        if "sample_cutoff" in msa_configs:
            self.msa_configs["train_cutoff"] = msa_configs["sample_cutoff"].get(
                "train", 512
            )
            self.msa_configs["test_cutoff"] = msa_configs["sample_cutoff"].get(
                "test", 16384
            )
            # the default msa_max_size is 16384 if not specified
            self.msa_max_size = self.msa_configs["train_cutoff"]
        if "min_size" in msa_configs:
            self.msa_configs["train_lowerb"] = msa_configs["min_size"].get("train", 1)
            self.msa_configs["test_lowerb"] = msa_configs["min_size"].get("test", 1)

        self.linear_no_bias_m = LinearNoBias(
            in_features=32 + 1 + 1, out_features=self.c_m
        )

        self.linear_no_bias_s = LinearNoBias(
            in_features=self.c_s_inputs, out_features=self.c_m
        )
        self.blocks = nn.ModuleList()

        for i in range(n_blocks):
            block = MSABlock(
                c_m=self.c_m,
                c_z=c_z,
                is_last_block=(i + 1 == n_blocks),
                msa_dropout=msa_dropout,
                pair_dropout=pair_dropout,
                msa_chunk_size=self.msa_chunk_size,
                msa_max_size=self.msa_max_size,
                hidden_scale_up=hidden_scale_up,
            )
            self.blocks.append(block)

    def one_hot_fp32(
        self, tensor: torch.Tensor, num_classes: int, dtype=torch.float32
    ) -> torch.Tensor:
        """like F.one_hot, but output dtype is float32.

        Args:
            tensor (torch.Tensor): the input tensor
            num_classes (int): num_classes
            dtype (torch.float32, optional): the output dtype. Defaults to torch.float32.

        Returns:
            torch.Tensor: the one-hot encoded tensor with shape
                [..., n_msa_sampled, N_token, num_classes]
        """
        shape = tensor.shape
        one_hot_tensor = torch.zeros(
            *shape, num_classes, dtype=dtype, device=tensor.device
        )
        one_hot_tensor.scatter_(len(shape), tensor.unsqueeze(-1), 1)
        return one_hot_tensor

    def forward(
        self,
        input_feature_dict: dict[str, Any],
        z: torch.Tensor,
        s_inputs: torch.Tensor,
        pair_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Algorithm 8 — MSAModule.

        Args:
            input_feature_dict: must contain ``msa``, ``has_deletion``, ``deletion_value``.
            z:        [..., N_token, N_token, c_z]  pair representation.
            s_inputs: [..., N_token, c_s_inputs]    single feature.
            pair_mask: [..., N_token, N_token]      optional pair mask.

        Returns:
            updated ``z``, same shape.
        """
        ##########################################################################
        # TODO: Algorithm 8 — MSAModule.                                         #
        #   Runs ``n_blocks`` MSABlocks. Each block does: OuterProductMean       #
        #   (MSA -> z), MSAPairWeightedAveraging (z -> MSA) + Transition,        #
        #   and one PairformerBlock pair update.                                 #
        #                                                                        #
        #   Step 1 — Cheap early-exits when there is nothing to do:              #
        #       if self.n_blocks < 1 or "msa" not in input_feature_dict:         #
        #           return z                                                     #
        #       if input_feature_dict["msa"].dim() < 2:                          #
        #           return z                                                     #
        #                                                                        #
        #   Step 2 — Sub-sample the MSA without replacement (uses train/test     #
        #     cutoffs configured at __init__ time). All three MSA features       #
        #     (``msa``, ``has_deletion``, ``deletion_value``) are sub-sampled    #
        #     along their N_msa axis (dim=-2):                                   #
        #       msa_feat = sample_msa_feature_dict_random_without_replacement(   #
        #           feat_dict   = input_feature_dict,                            #
        #           dim_dict    = {feat_name: -2 for feat_name in                #
        #                            self.input_feature},                        #
        #           cutoff      = self.msa_configs["train_cutoff"]               #
        #                          if self.training                              #
        #                          else self.msa_configs["test_cutoff"],         #
        #           lower_bound = self.msa_configs["train_lowerb"]               #
        #                          if self.training                              #
        #                          else self.msa_configs["test_lowerb"],        #
        #           strategy    = self.msa_configs["strategy"],                  #
        #       )                                                                #
        #                                                                        #
        #   Step 3 — One-hot the residue type (``msa``) into 32 classes. For     #
        #     very long inference sequences keep the one-hot in fp32 to halve    #
        #     peak memory (see ``one_hot_fp32`` helper); otherwise use           #
        #     ``F.one_hot`` which returns int64:                                 #
        #       if not self.training and z.shape[-2] > 2000:                     #
        #           msa_feat["msa"] = self.one_hot_fp32(                         #
        #               msa_feat["msa"], num_classes=self.input_feature["msa"]) #
        #       else:                                                            #
        #           msa_feat["msa"] = F.one_hot(                                 #
        #               msa_feat["msa"], num_classes=self.input_feature["msa"]) #
        #                                                                        #
        #   Step 4 — Concatenate ``msa``, ``has_deletion``, ``deletion_value``   #
        #     along the channel dim (sizes 32 / 1 / 1 -> 34):                    #
        #       target_shape = msa_feat["msa"].shape[:-1]                        #
        #       msa_sample = torch.cat(                                         #
        #           [ msa_feat[name].reshape(*target_shape, d)                  #
        #             for name, d in self.input_feature.items() ],              #
        #           dim=-1,                                                      #
        #       )   # [*, N_msa_sample, N_token, 34]                            #
        #       if not self.training: del msa_feat                              #
        #                                                                        #
        #   Step 5 — Linear-project the concatenated features to ``c_m`` and    #
        #     add the broadcast contribution from the single inputs (so the    #
        #     MSA channel is conditioned on the per-token single embedding):    #
        #       msa_sample = self.linear_no_bias_m(msa_sample)                  #
        #       msa_sample = msa_sample + self.linear_no_bias_s(s_inputs)       #
        #                                                                        #
        #   Step 6 — Run all MSABlocks. Each returns updated (msa_sample, z);   #
        #     the very last block returns ``msa_sample = None`` since the MSA   #
        #     stream is no longer needed downstream:                            #
        #       for block in self.blocks:                                       #
        #           msa_sample, z = block(msa_sample, z, pair_mask=pair_mask)   #
        #   Return ``z``.                                                       #
        #                                                                        #
        # TODO: 算法 8 —— MSAModule。                                            #
        #   依次跑 ``n_blocks`` 个 MSABlock。每块做                                #
        #   OuterProductMean (MSA -> z)、MSAPairWeightedAveraging (z -> MSA)     #
        #   + Transition，最后一次 PairformerBlock 的 pair 更新。                #
        #                                                                        #
        #   步骤 1 — 廉价提前返回:                                                  #
        #       if self.n_blocks < 1 or "msa" not in input_feature_dict:         #
        #           return z                                                     #
        #       if input_feature_dict["msa"].dim() < 2:                          #
        #           return z                                                     #
        #                                                                        #
        #   步骤 2 — 不放回随机采样 MSA 子集 (训练/推理 cutoff 在 __init__ 中配置)。 #
        #     三个 MSA 特征 (``msa``, ``has_deletion``, ``deletion_value``) 均    #
        #     沿 N_msa 轴 (dim=-2) 一起采样:                                       #
        #       msa_feat = sample_msa_feature_dict_random_without_replacement(   #
        #           feat_dict   = input_feature_dict,                            #
        #           dim_dict    = {feat_name: -2 for feat_name in                #
        #                            self.input_feature},                        #
        #           cutoff      = self.msa_configs["train_cutoff"]               #
        #                          if self.training                              #
        #                          else self.msa_configs["test_cutoff"],         #
        #           lower_bound = self.msa_configs["train_lowerb"]               #
        #                          if self.training                              #
        #                          else self.msa_configs["test_lowerb"],        #
        #           strategy    = self.msa_configs["strategy"],                  #
        #       )                                                                #
        #                                                                        #
        #   步骤 3 — 残基类型 (``msa``) one-hot 到 32 类。长序列推理时用 fp32      #
        #     版本减半显存:                                                       #
        #       if not self.training and z.shape[-2] > 2000:                     #
        #           msa_feat["msa"] = self.one_hot_fp32(                         #
        #               msa_feat["msa"], num_classes=self.input_feature["msa"]) #
        #       else:                                                            #
        #           msa_feat["msa"] = F.one_hot(                                 #
        #               msa_feat["msa"], num_classes=self.input_feature["msa"]) #
        #                                                                        #
        #   步骤 4 — 沿通道维拼接 ``msa`` / ``has_deletion`` / ``deletion_value`` #
        #     (32 / 1 / 1 -> 34):                                                #
        #       target_shape = msa_feat["msa"].shape[:-1]                        #
        #       msa_sample = torch.cat(                                         #
        #           [ msa_feat[name].reshape(*target_shape, d)                  #
        #             for name, d in self.input_feature.items() ],              #
        #           dim=-1,                                                      #
        #       )   # [*, N_msa_sample, N_token, 34]                            #
        #       if not self.training: del msa_feat                              #
        #                                                                        #
        #   步骤 5 — 投到 c_m 并叠加单序列广播贡献                                  #
        #     (使 MSA 通道以每 token 的 single 嵌入为条件):                       #
        #       msa_sample = self.linear_no_bias_m(msa_sample)                  #
        #       msa_sample = msa_sample + self.linear_no_bias_s(s_inputs)       #
        #                                                                        #
        #   步骤 6 — 跑完所有 MSABlock。最后一块返回 ``msa_sample = None``        #
        #     (后续不再需要 MSA 流):                                              #
        #       for block in self.blocks:                                       #
        #           msa_sample, z = block(msa_sample, z, pair_mask=pair_mask)   #
        #   返回 ``z``。                                                          #
        ##########################################################################

        if self.n_blocks < 1 or "msa" not in input_feature_dict:
            return z
        if input_feature_dict["msa"].dim() < 2:
            return z
        msa_feat = sample_msa_feature_dict_random_without_replacement(
            feat_dict=input_feature_dict,
            dim_dict={feat_name: -2 for feat_name in self.input_feature},
            cutoff=(
                self.msa_configs["train_cutoff"]
                if self.training
                else self.msa_configs["test_cutoff"]
            ),
            lower_bound=(
                self.msa_configs["train_lowerb"]
                if self.training
                else self.msa_configs["test_lowerb"]
            ),
            strategy=self.msa_configs["strategy"],
        )
        # pylint: disable=E1102
        if not self.training and z.shape[-2] > 2000:
            # msa_feat["msa"] is torch.int64, we convert it
            # to torch.float32 for saving half of the CUDA memory
            msa_feat["msa"] = self.one_hot_fp32(
                msa_feat["msa"],
                num_classes=self.input_feature["msa"],
            )
        else:
            msa_feat["msa"] = torch.nn.functional.one_hot(
                msa_feat["msa"],
                num_classes=self.input_feature["msa"],
            )

        target_shape = msa_feat["msa"].shape[:-1]
        msa_sample = torch.cat(
            [
                msa_feat[name].reshape(*target_shape, d)
                for name, d in self.input_feature.items()
            ],
            dim=-1,
        )  # [..., N_msa_sample, N_token, 32 + 1 + 1]
        # Msa_feat is very large, if N_MSA=16384 and N_token=4000,
        # msa_feat["msa"] consumes about 16G CUDA memory, so we
        # need to clear cache to avoid OOM
        if not self.training:
            del msa_feat
        # Line2
        msa_sample = self.linear_no_bias_m(msa_sample)

        # Auto broadcast [...,n_msa_sampled, n_token, c_m]
        msa_sample = msa_sample + self.linear_no_bias_s(s_inputs)
        for block in self.blocks:
            msa_sample, z = block(msa_sample, z, pair_mask=pair_mask)

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################
        return z

