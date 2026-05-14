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
        # TODO: Algorithm 2. Build the per-token single representation.          #
        #   1. Run AtomAttentionEncoder on the per-atom features (without coords)#
        #      to get a [..., N_token, c_token] token-level activation ``a``.    #
        #   2. Concatenate ``a`` with per-token features (restype / profile /    #
        #      deletion_mean) along the channel dim.                             #
        #   3. If ESM features are enabled, add the projected ESM embedding.    #
        # TODO: Algorithm 2。生成每 token 的 single 表示。                       #
        #   1. 跑 AtomAttentionEncoder (不带坐标) 得到 token 级激活 a。          #
        #   2. 把 a 和每 token 特征 (restype / profile / deletion_mean) 沿通道  #
        #      维拼起来。                                                        #
        #   3. 若启用了 ESM，加上投影后的 ESM embedding。                       #
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

