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



class RelativePositionEncoding(nn.Module):
    """
    Implements Algorithm 3 in AF3

    Args:
        r_max (int, optional): Relative position indices clip value. Defaults to 32.
        s_max (int, optional): Relative chain indices clip value. Defaults to 2.
        c_z (int, optional): hidden dim [for pair embedding]. Defaults to 128.
    """

    def __init__(self, r_max: int = 32, s_max: int = 2, c_z: int = 128) -> None:
        super(RelativePositionEncoding, self).__init__()
        self.r_max = r_max
        self.s_max = s_max
        self.c_z = c_z
        self.linear_no_bias = LinearNoBias(
            in_features=(4 * self.r_max + 2 * self.s_max + 7), out_features=self.c_z
        )
        self.input_feature = {
            "asym_id": 1,
            "residue_index": 1,
            "entity_id": 1,
            "sym_id": 1,
            "token_index": 1,
        }

    def generate_relp(self, input_feature_dict: dict[str, Any]) -> dict[str, Any]:
        """asym_id, residue_index, entity_id, token_index, sym_id"""
        with torch.no_grad():
            asym_id = input_feature_dict["asym_id"]
            residue_index = input_feature_dict["residue_index"]
            entity_id = input_feature_dict["entity_id"]
            token_index = input_feature_dict["token_index"]
            sym_id = input_feature_dict["sym_id"]

            b_same_chain = (
                asym_id[..., :, None] == asym_id[..., None, :]
            ).long()  # [..., N_token, N_token]
            b_same_residue = (
                residue_index[..., :, None] == residue_index[..., None, :]
            ).long()  # [..., N_token, N_token]
            b_same_entity = (
                entity_id[..., :, None] == entity_id[..., None, :]
            ).long()  # [..., N_token, N_token]
            d_residue = torch.clip(
                input=residue_index[..., :, None]
                - residue_index[..., None, :]
                + self.r_max,
                min=0,
                max=2 * self.r_max,
            ) * b_same_chain + (1 - b_same_chain) * (
                2 * self.r_max + 1
            )  # [..., N_token, N_token]
            a_rel_pos = F.one_hot(d_residue, 2 * (self.r_max + 1))
            d_token = torch.clip(
                input=token_index[..., :, None]
                - token_index[..., None, :]
                + self.r_max,
                min=0,
                max=2 * self.r_max,
            ) * b_same_chain * b_same_residue + (1 - b_same_chain * b_same_residue) * (
                2 * self.r_max + 1
            )  # [..., N_token, N_token]
            a_rel_token = F.one_hot(d_token, 2 * (self.r_max + 1))
            d_chain = torch.clip(
                input=sym_id[..., :, None] - sym_id[..., None, :] + self.s_max,
                min=0,
                max=2 * self.s_max,
            ) * b_same_entity + (1 - b_same_entity) * (
                2 * self.s_max + 1
            )  # [..., N_token, N_token]
            a_rel_chain = F.one_hot(d_chain, 2 * (self.s_max + 1))

            relp = torch.cat(
                [a_rel_pos, a_rel_token, b_same_entity[..., None], a_rel_chain],
                dim=-1,
            ).float()
            input_feature_dict["relp"] = relp
        return input_feature_dict

    def forward(self, relp_feature) -> torch.Tensor:
        """
        Args:
            asym_id / residue_index / entity_id / sym_id / token_index
                [..., N_tokens]
        Returns:
            torch.Tensor: relative position encoding
                [..., N_token, N_token, c_z]
        """
        return self.linear_no_bias(relp_feature)



class FourierEmbedding(nn.Module):
    """
    Implements Algorithm 22 in AF3

    Args:
        c (int): embedding dim.
        seed (int, optional): random seed. Defaults to 42.
    """

    def __init__(self, c: int, seed: int = 42) -> None:
        super(FourierEmbedding, self).__init__()
        self.c = c
        self.seed = seed
        generator = torch.Generator()
        generator.manual_seed(seed)
        w_value = torch.randn(size=(c,), generator=generator)
        self.w = nn.Parameter(w_value, requires_grad=False)
        b_value = torch.randn(size=(c,), generator=generator)
        self.b = nn.Parameter(b_value, requires_grad=False)

    def forward(self, t_hat_noise_level: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t_hat_noise_level (torch.Tensor): the noise level
                [..., N_sample]

        Returns:
            torch.Tensor: the output fourier embedding
                [..., N_sample, c]
        """
        return torch.cos(
            input=2 * torch.pi * (t_hat_noise_level.unsqueeze(dim=-1) * self.w + self.b)
        )

