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
from pairformer.pair_stack import PairformerStack
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



class TemplateEmbedder(nn.Module):
    """
    Implements Algorithm 16 in AF3

    Args:
        n_blocks (int, optional): number of blocks for TemplateEmbedder. Defaults to 2.
        c (int, optional): hidden dim of TemplateEmbedder. Defaults to 64.
        c_z (int, optional): hidden dim [for pair embedding]. Defaults to 128.
        num_intermediate_factor (int, optional): number of intermediate factor for transition. Defaults to 2.
        dropout (float, optional): dropout ratio for PairformerStack. Defaults to 0.25.
            Note this value is missed in Algorithm 16, so we use default ratio for Pairformer
        blocks_per_ckpt (int, optional): number of TemplateEmbedder/Pairformer blocks in each activation
            checkpoint. Defaults to None.
        hidden_scale_up (bool, optional): whether scale up the hidden if c_z scales. Defaults to False.
    """

    def __init__(
        self,
        n_blocks: int = 2,
        c: int = 64,
        c_z: int = 128,
        num_intermediate_factor: int = 2,
        dropout: float = 0.25,
        blocks_per_ckpt: Optional[int] = None,
        hidden_scale_up: bool = False,
    ) -> None:
        super(TemplateEmbedder, self).__init__()
        self.n_blocks = n_blocks
        self.c = c
        self.c_z = c_z
        self.input_feature1 = {
            "template_distogram": 39,
            "template_backbone_frame_mask": 1,
            "template_unit_vector": 3,
            "template_pseudo_beta_mask": 1,
        }
        self.input_feature2 = {
            "template_restype_i": 32,
            "template_restype_j": 32,
        }
        self.distogram = {"max_bin": 50.75, "min_bin": 3.25, "no_bins": 39}
        self.inf = 100000.0

        self.linear_no_bias_z = LinearNoBias(in_features=self.c_z, out_features=self.c)
        self.layernorm_z = LayerNorm(self.c_z)
        self.linear_no_bias_a = LinearNoBias(
            in_features=sum(self.input_feature1.values())
            + sum(self.input_feature2.values()),
            out_features=self.c,
        )
        self.pairformer_stack = PairformerStack(
            c_s=0,
            c_z=c,
            n_blocks=self.n_blocks,
            num_intermediate_factor=num_intermediate_factor,
            dropout=dropout,
            blocks_per_ckpt=blocks_per_ckpt,
            hidden_scale_up=hidden_scale_up,
        )
        self.layernorm_v = LayerNorm(self.c)
        self.relu = nn.ReLU()
        self.linear_no_bias_u = LinearNoBias(in_features=self.c, out_features=self.c_z)

    def forward(
        self,
        input_feature_dict: dict[str, Any],
        z: torch.Tensor,
        pair_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Algorithm 16 — TemplateEmbedder.

        Returns 0 (a scalar that broadcasts) if templates aren't provided or
        if the module was built with ``n_blocks=0``.
        """
        ##########################################################################
        # TODO: Algorithm 16 — TemplateEmbedder. Folds structural templates    #
        #   into the pair representation.                                       #
        #                                                                        #
        #   Step 1 — Cheap early-exit when there is nothing to do:               #
        #       if ("template_aatype" not in input_feature_dict                  #
        #           or self.n_blocks < 1):                                       #
        #           return 0                  # broadcasts as a scalar           #
        #                                                                        #
        #   Step 2 — Build the per-pair “same-chain” mask from ``asym_id``       #
        #     (templates only inform pairs within a single chain) and default    #
        #     the pair mask:                                                     #
        #       asym_id = input_feature_dict["asym_id"]                          #
        #       multichain_mask = (asym_id[:, None] == asym_id[None, :])        #
        #                            .to(z.dtype)         # [N_tok, N_tok]      #
        #       num_residues     = z.shape[0]                                    #
        #       num_templates    = input_feature_dict["template_aatype"]         #
        #                                .shape[0]                              #
        #       query_num_channels = z.shape[-1]                                 #
        #       if pair_mask is None:                                           #
        #           pair_mask = z.new_ones(z.shape[:-1])                        #
        #                                                                        #
        #   Step 3 — Pre-LayerNorm the pair representation once. The same       #
        #     normalized ``z`` feeds every template-specific branch:             #
        #       z = self.layernorm_z(z)                                         #
        #                                                                        #
        #   Step 4 — Sum the per-template embeddings produced by                 #
        #     ``self.single_template_forward`` (which concatenates the           #
        #     template distogram / frames / unit-vector / aatype features,       #
        #     adds them to ``self.linear_no_bias_z(z)``, runs a small             #
        #     PairformerStack, and ``layernorm_v`` ’s the result):                #
        #       u = 0                                                            #
        #       for template_id in range(num_templates):                         #
        #           u = u + self.single_template_forward(                        #
        #               template_id        = template_id,                        #
        #               input_feature_dict = input_feature_dict,                 #
        #               z                  = z,                                  #
        #               pair_mask          = pair_mask,                          #
        #               multichain_mask    = multichain_mask,                    #
        #           )                                                            #
        #                                                                        #
        #   Step 5 — Average across templates (with ``+1e-7`` to be safe when    #
        #     ``num_templates == 0``), ReLU, then project back to ``c_z``:       #
        #       u = u / (1e-7 + num_templates)                                   #
        #       u = self.linear_no_bias_u(self.relu(u))                          #
        #       assert u.shape == (num_residues, num_residues,                   #
        #                          query_num_channels)                          #
        #   Return ``u`` (the caller adds this to ``z`` outside).                #
        #                                                                        #
        # TODO: 算法 16 —— TemplateEmbedder。把结构模板信息汇入 pair 表示。      #
        #                                                                        #
        #   步骤 1 — 廉价提前返回:                                                  #
        #       if ("template_aatype" not in input_feature_dict                  #
        #           or self.n_blocks < 1):                                       #
        #           return 0                  # 当作标量广播                      #
        #                                                                        #
        #   步骤 2 — 用 ``asym_id`` 构造“同链 pair 掩码” (模板只贡献同链内 pair)   #
        #     并设置默认 pair mask:                                                #
        #       asym_id = input_feature_dict["asym_id"]                          #
        #       multichain_mask = (asym_id[:, None] == asym_id[None, :])        #
        #                            .to(z.dtype)         # [N_tok, N_tok]      #
        #       num_residues     = z.shape[0]                                    #
        #       num_templates    = input_feature_dict["template_aatype"]         #
        #                                .shape[0]                              #
        #       query_num_channels = z.shape[-1]                                 #
        #       if pair_mask is None:                                           #
        #           pair_mask = z.new_ones(z.shape[:-1])                        #
        #                                                                        #
        #   步骤 3 — 对 z 做一次 Pre-LayerNorm，同一份归一化后的 ``z`` 喂给         #
        #     所有模板分支:                                                       #
        #       z = self.layernorm_z(z)                                         #
        #                                                                        #
        #   步骤 4 — 累加每个模板的嵌入 (``single_template_forward`` 内部把模板    #
        #     特征 distogram / frames / unit-vector / aatype 拼接，叠加          #
        #     ``self.linear_no_bias_z(z)``，跑小型 PairformerStack，再过           #
        #     ``layernorm_v``):                                                   #
        #       u = 0                                                            #
        #       for template_id in range(num_templates):                         #
        #           u = u + self.single_template_forward(                        #
        #               template_id        = template_id,                        #
        #               input_feature_dict = input_feature_dict,                 #
        #               z                  = z,                                  #
        #               pair_mask          = pair_mask,                          #
        #               multichain_mask    = multichain_mask,                    #
        #           )                                                            #
        #                                                                        #
        #   步骤 5 — 沿模板维取均值 (+1e-7 保护 0 个模板的情况)、ReLU，再投回      #
        #     ``c_z``:                                                           #
        #       u = u / (1e-7 + num_templates)                                   #
        #       u = self.linear_no_bias_u(self.relu(u))                          #
        #       assert u.shape == (num_residues, num_residues,                   #
        #                          query_num_channels)                          #
        #   返回 ``u`` (调用方负责把它加回 ``z``)。                                 #
        ##########################################################################

        if "template_aatype" not in input_feature_dict or self.n_blocks < 1:
            return 0  # type: ignore[return-value]
        asym_id = input_feature_dict["asym_id"]
        multichain_mask = (asym_id[:, None] == asym_id[None, :]).to(z.dtype)

        num_residues = z.shape[0]
        num_templates = input_feature_dict["template_aatype"].shape[0]
        query_num_channels = z.shape[-1]

        if pair_mask is None:
            pair_mask = z.new_ones(z.shape[:-1])

        z = self.layernorm_z(z)
        u = 0
        for template_id in range(num_templates):
            u = u + self.single_template_forward(
                template_id=template_id,
                input_feature_dict=input_feature_dict,
                z=z,
                pair_mask=pair_mask,
                multichain_mask=multichain_mask,
            )
        u = u / (1e-7 + num_templates)
        u = self.linear_no_bias_u(self.relu(u))
        assert u.shape == (num_residues, num_residues, query_num_channels)

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################
        return u

    def single_template_forward(
        self,
        template_id: int,
        input_feature_dict: dict[str, Any],
        z: torch.Tensor,
        pair_mask: Optional[torch.Tensor] = None,
        multichain_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        to_concat = []

        dgram = input_feature_dict["template_distogram"][
            template_id
        ]  # [N_token, N_token, 39]
        pseudo_beta_mask_2d = input_feature_dict["template_pseudo_beta_mask"][
            template_id
        ]
        dgram = dgram * multichain_mask[..., None] * pair_mask[..., None]
        pseudo_beta_mask_2d = (
            pseudo_beta_mask_2d * multichain_mask * pair_mask
        )  # [N_token, N_token]
        to_concat.append(dgram)
        to_concat.append(pseudo_beta_mask_2d.unsqueeze(-1))

        aatype = input_feature_dict["template_aatype"][template_id]  # [N_token]
        aatype = F.one_hot(aatype, num_classes=len(STD_RESIDUES_WITH_GAP))
        to_concat.append(expand_at_dim(aatype, dim=-3, n=z.shape[0]))
        to_concat.append(expand_at_dim(aatype, dim=-2, n=z.shape[0]))

        unit_vector = input_feature_dict["template_unit_vector"][template_id]
        unit_vector = (
            unit_vector * multichain_mask[..., None] * pair_mask[..., None]
        )  # [N_token, N_token, 3]
        to_concat.append(unit_vector)

        backbone_mask_2d = input_feature_dict["template_backbone_frame_mask"][
            template_id
        ]
        backbone_mask_2d = backbone_mask_2d * multichain_mask * pair_mask
        to_concat.append(backbone_mask_2d.unsqueeze(-1))

        at = torch.concat(to_concat, dim=-1)
        v = self.linear_no_bias_z(z) + self.linear_no_bias_a(at)
        _, v = self.pairformer_stack(s=None, z=v, pair_mask=pair_mask)
        v = self.layernorm_v(v)
        return v
