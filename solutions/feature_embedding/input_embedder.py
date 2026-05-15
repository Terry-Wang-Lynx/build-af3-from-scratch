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

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from attention.linear import LinearNoBias
from feature_embedding.atom_attention import AtomAttentionEncoder
from runtime.logger import get_logger

logger = get_logger(__name__)



class InputFeatureEmbedder(nn.Module):
    """
    Implements Algorithm 2 in AF3

    Args:
        c_atom (int, optional): atom embedding dim. Defaults to 128.
        c_atompair (int, optional): atom pair embedding dim. Defaults to 16.
        c_token (int, optional): token embedding dim. Defaults to 384.
        esm_configs (dict[str, Any], optional): esm config. Defaults to {}.
    """

    def __init__(
        self,
        c_atom: int = 128,
        c_atompair: int = 16,
        c_token: int = 384,
        esm_configs: dict[str, Any] = {},
    ) -> None:
        super(InputFeatureEmbedder, self).__init__()
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.c_token = c_token
        ##########################################################################
        # TODO: InputFeatureEmbedder (Algorithm 2) — wire up:                    #
        #                                                                        #
        #   Step 1 — Inner atom encoder (Algorithm 5) configured WITHOUT       #
        #     coordinates (we're producing the trunk single from atom features  #
        #     alone here):                                                       #
        #       self.atom_attention_encoder = AtomAttentionEncoder(             #
        #           c_atom=c_atom,                                              #
        #           c_atompair=c_atompair,                                      #
        #           c_token=c_token,                                            #
        #           has_coords=False,                                            #
        #       )                                                                #
        #                                                                        #
        #   Step 2 — Optional ESM bypass. When the PLM-variant ckpt is in use,  #
        #     we project a precomputed ESM embedding to the same output width   #
        #     as ``s_inputs`` and add it. Zero-initialise the projection so      #
        #     non-ESM checkpoints behave identically:                           #
        #       self.esm_configs = {                                             #
        #           "enable":        esm_configs.get("enable", False),          #
        #           "embedding_dim": esm_configs.get("embedding_dim", 2560),    #
        #       }                                                                #
        #       if self.esm_configs["enable"]:                                   #
        #           self.linear_esm = LinearNoBias(                              #
        #               self.esm_configs["embedding_dim"],                       #
        #               self.c_token + 32 + 32 + 1,                              #
        #           )                                                            #
        #           nn.init.zeros_(self.linear_esm.weight)                       #
        #                                                                        #
        #   Step 3 — Declare which per-token features feed the cat in forward.  #
        #     Order and widths must match exactly: restype 32 + profile 32 +    #
        #     deletion_mean 1:                                                  #
        #       self.input_feature = {                                           #
        #           "restype": 32, "profile": 32, "deletion_mean": 1}            #
        #                                                                        #
        # TODO: InputFeatureEmbedder (算法 2) —— 装配:                            #
        #                                                                        #
        #   步骤 1 — 内部原子编码器 (算法 5)，**不带坐标** (这里只用原子特征产生   #
        #     trunk single):                                                     #
        #       self.atom_attention_encoder = AtomAttentionEncoder(             #
        #           c_atom=c_atom,                                              #
        #           c_atompair=c_atompair,                                      #
        #           c_token=c_token,                                            #
        #           has_coords=False,                                            #
        #       )                                                                #
        #                                                                        #
        #   步骤 2 — 可选 ESM 旁路 (PLM 变体的 ckpt 用)。投到与 ``s_inputs`` 同宽   #
        #     再加上去。零初始化使非-PLM ckpt 完全不受影响:                        #
        #       self.esm_configs = {                                             #
        #           "enable":        esm_configs.get("enable", False),          #
        #           "embedding_dim": esm_configs.get("embedding_dim", 2560),    #
        #       }                                                                #
        #       if self.esm_configs["enable"]:                                   #
        #           self.linear_esm = LinearNoBias(                              #
        #               self.esm_configs["embedding_dim"],                       #
        #               self.c_token + 32 + 32 + 1,                              #
        #           )                                                            #
        #           nn.init.zeros_(self.linear_esm.weight)                       #
        #                                                                        #
        #   步骤 3 — 声明 forward 里要 cat 的每 token 特征。顺序和宽度必须严格       #
        #     一致: restype 32 + profile 32 + deletion_mean 1:                   #
        #       self.input_feature = {                                           #
        #           "restype": 32, "profile": 32, "deletion_mean": 1}            #
        ##########################################################################

        self.atom_attention_encoder = AtomAttentionEncoder(
            c_atom=c_atom,
            c_atompair=c_atompair,
            c_token=c_token,
            has_coords=False,
        )

        self.esm_configs = {
            "enable": esm_configs.get("enable", False),
            "embedding_dim": esm_configs.get("embedding_dim", 2560),
        }
        if self.esm_configs["enable"]:
            self.linear_esm = LinearNoBias(
                self.esm_configs["embedding_dim"],
                self.c_token + 32 + 32 + 1,
            )
            nn.init.zeros_(self.linear_esm.weight)

        # Line2
        self.input_feature = {"restype": 32, "profile": 32, "deletion_mean": 1}

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

    def forward(
        self,
        input_feature_dict: dict[str, Any],
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        """Algorithm 2 — assemble the per-token single embedding.

        Returns:
            s_inputs: [..., N_token, c_token + 32 + 32 + 1 = 449].
        """
        del inplace_safe
        ##########################################################################
        # TODO: Algorithm 2 — InputFeatureEmbedder. Build the per-token single   #
        #   representation ``s_inputs`` that the trunk consumes. The output      #
        #   width is ``c_token + 32 + 32 + 1 = 449``                             #
        #   (= aggregated AtomAttentionEncoder output + restype + profile +      #
        #    deletion_mean).                                                     #
        #                                                                        #
        #   Step 1 — Run the atom-level encoder with ``has_coords=False`` (set   #
        #     in ``__init__``), aggregating atom features up to tokens. Each    #
        #     positional argument corresponds to one entry in the AF3 feature   #
        #     dict; the encoder returns ``(a, q, c, p)`` but we only want the   #
        #     token activation ``a`` of shape [..., N_token, c_token]:           #
        #       a, _, _, _ = self.atom_attention_encoder(                       #
        #           input_feature_dict["atom_to_token_idx"],                    #
        #           input_feature_dict["ref_pos"],                              #
        #           input_feature_dict["ref_charge"],                           #
        #           input_feature_dict["ref_mask"],                             #
        #           input_feature_dict["ref_atom_name_chars"],                  #
        #           input_feature_dict["ref_element"],                          #
        #           input_feature_dict["d_lm"],                                 #
        #           input_feature_dict["v_lm"],                                 #
        #           input_feature_dict["pad_info"],                             #
        #       )                                                                #
        #                                                                        #
        #   Step 2 — Concatenate ``a`` with each per-token feature, reshaping    #
        #     to the common token batch shape so the cat aligns cleanly:        #
        #       batch_shape = input_feature_dict["restype"].shape[:-1]           #
        #       s_inputs = torch.cat(                                            #
        #           [a]                                                          #
        #           + [ input_feature_dict[name].reshape(*batch_shape, d)        #
        #               for name, d in self.input_feature.items() ],            #
        #           dim=-1,                                                      #
        #       )                                                                #
        #     ``self.input_feature`` is ordered exactly                          #
        #     ``{"restype": 32, "profile": 32, "deletion_mean": 1}`` — keep      #
        #     this order to match the checkpoint.                                #
        #                                                                        #
        #   Step 3 — Optional ESM bypass. When ``esm_configs["enable"]`` is       #
        #     True a separate zero-initialized projection of the precomputed     #
        #     ESM embedding is added back to the same channel layout:            #
        #       if self.esm_configs["enable"]:                                   #
        #           esm_emb = self.linear_esm(                                   #
        #               input_feature_dict["esm_token_embedding"])              #
        #           s_inputs = s_inputs + esm_emb                                #
        #   Return ``s_inputs``.                                                 #
        #                                                                        #
        # TODO: 算法 2 —— InputFeatureEmbedder。                                  #
        #   生成主干使用的每 token 单序列表示 ``s_inputs``。                       #
        #   输出宽度 ``c_token + 32 + 32 + 1 = 449``                              #
        #   (= AtomAttentionEncoder 聚合的 token 激活 + restype + profile +       #
        #    deletion_mean)。                                                    #
        #                                                                        #
        #   步骤 1 — ``has_coords=False`` 跑 AtomAttentionEncoder (__init__        #
        #     中已配)。每个位置参数对应 AF3 特征字典中的一项；编码器返回           #
        #     ``(a, q, c, p)``，这里只需 token 激活 ``a``，形状                    #
        #     [..., N_token, c_token]:                                            #
        #       a, _, _, _ = self.atom_attention_encoder(                       #
        #           input_feature_dict["atom_to_token_idx"],                    #
        #           input_feature_dict["ref_pos"],                              #
        #           input_feature_dict["ref_charge"],                           #
        #           input_feature_dict["ref_mask"],                             #
        #           input_feature_dict["ref_atom_name_chars"],                  #
        #           input_feature_dict["ref_element"],                          #
        #           input_feature_dict["d_lm"],                                 #
        #           input_feature_dict["v_lm"],                                 #
        #           input_feature_dict["pad_info"],                             #
        #       )                                                                #
        #                                                                        #
        #   步骤 2 — 把 ``a`` 与每 token 特征沿通道维拼接，先 reshape 到统一的       #
        #     token batch 形状:                                                  #
        #       batch_shape = input_feature_dict["restype"].shape[:-1]           #
        #       s_inputs = torch.cat(                                            #
        #           [a]                                                          #
        #           + [ input_feature_dict[name].reshape(*batch_shape, d)        #
        #               for name, d in self.input_feature.items() ],            #
        #           dim=-1,                                                      #
        #       )                                                                #
        #     ``self.input_feature`` 的顺序必须保持                                #
        #     ``{"restype": 32, "profile": 32, "deletion_mean": 1}`` —— 与权重    #
        #     一致。                                                              #
        #                                                                        #
        #   步骤 3 — ESM 旁路 (可选)。``esm_configs["enable"]`` 为 True 时，       #
        #     用零初始化的 ``linear_esm`` 投影预算好的 ESM 嵌入并叠加:             #
        #       if self.esm_configs["enable"]:                                   #
        #           esm_emb = self.linear_esm(                                   #
        #               input_feature_dict["esm_token_embedding"])              #
        #           s_inputs = s_inputs + esm_emb                                #
        #   返回 ``s_inputs``。                                                   #
        ##########################################################################

        a, _, _, _ = self.atom_attention_encoder(
            input_feature_dict["atom_to_token_idx"],
            input_feature_dict["ref_pos"],
            input_feature_dict["ref_charge"],
            input_feature_dict["ref_mask"],
            input_feature_dict["ref_atom_name_chars"],
            input_feature_dict["ref_element"],
            input_feature_dict["d_lm"],
            input_feature_dict["v_lm"],
            input_feature_dict["pad_info"],
        )
        batch_shape = input_feature_dict["restype"].shape[:-1]
        s_inputs = torch.cat(
            [a]
            + [
                input_feature_dict[name].reshape(*batch_shape, d)
                for name, d in self.input_feature.items()
            ],
            dim=-1,
        )

        if self.esm_configs["enable"]:
            esm_emb = self.linear_esm(input_feature_dict["esm_token_embedding"])
            s_inputs = s_inputs + esm_emb

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################
        return s_inputs

