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
        attn_out = self.drop_path(
            self.attention_pair_bias(a=a, s=s, z=z, n_queries=n_queries, n_keys=n_keys)
        )
        attn_out = attn_out + a
        ff_out = self.drop_path(self.conditioned_transition_block(a=attn_out, s=s))
        return ff_out + attn_out, s, z



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
        for block in self.blocks:
            a, s, z = block(a, s, z, n_queries=n_queries, n_keys=n_keys)
        return a



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
        a = self.adaln(a, s)
        b = F.silu((self.linear_nobias_a1(a))) * self.linear_nobias_a2(a)
        # Output projection (from adaLN-Zero [27])
        a = torch.sigmoid(self.linear_s(s)) * self.linear_nobias_b(b)
        return a

